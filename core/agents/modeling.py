"""Modeling agent — log-log + semi-log + LightGBM iterative comparison.

For each eligible PPG (from ``ppg_selection.json``):

1. Split the refined feature frame 80/20 chronologically.
2. Fit log-log OLS on the train half.
3. If the own-price elasticity from log-log has the wrong sign, also fit
   semi-log OLS (sign-retry; records `sign_retry_fired = True`).
4. Fit LightGBM on the train half and recover an average own-price
   elasticity by numerical bump.
5. Pick the winner: lowest hold-out WAPE among sign-correct candidates.
   If no candidate has the right sign, fall back to the lowest-WAPE
   wrong-sign candidate and surface `sign_ok = False` for the
   results-reasoning stage to flag.

Outputs:
- ``modeling_results.json`` per-PPG: every fit attempted + the winner.
- ``elasticity_per_ppg.json`` compact summary the UI / downstream
  agents read.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from core.agents.base import Agent
from core.features.engineering import ENGINEERED_COLUMNS, TARGET
from core.models.base import ElasticityFit
from core.models.bayes_hier import shrink, to_payload
from core.models.lightgbm_model import fit_lightgbm
from core.models.loglog_ols import fit_loglog
from core.models.metrics import chronological_split
from core.models.predictor import build_predictor
from core.models.semilog_ols import fit_semilog
from core.orchestrator.state import AgentResult, ArtifactRef, RunState


SYSTEM_PROMPT = """You are the pricing-elasticity analyst. You receive a JSON
table of per-PPG model candidates with own-price elasticity, std error,
p-value, R², hold-out WAPE, and whether the sign-retry fired. Return
STRICT JSON of the form:
{"narrative": "<=320 chars explaining which elasticities are trustworthy
and where the model still struggles",
 "concerns": [{"ppg_id": "...", "issue": "..."}, ...]}
JSON only, no prose. Cite only PPGs present in the input."""


def _load_features(run_dir: Path) -> pd.DataFrame:
    pq = run_dir / "features.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except (ImportError, ValueError):
            pass
    csv = run_dir / "features.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise RuntimeError("features artifact missing — feature_engineering must run first")


def _load_eligible_ppgs(run_dir: Path) -> list[str]:
    sel_path = run_dir / "ppg_selection.json"
    if not sel_path.exists():
        raise RuntimeError("ppg_selection.json missing — ppg_selection must run first")
    blob = json.loads(sel_path.read_text())
    rows = blob if isinstance(blob, list) else blob.get("ppgs", [])
    return [str(r["ppg_id"]) for r in rows if r.get("eligible")]


def _load_controls(run_dir: Path) -> list[str]:
    refine_path = run_dir / "feature_refine.json"
    if refine_path.exists():
        blob = json.loads(refine_path.read_text())
        kept = blob.get("kept") or []
        return [c for c in kept if c not in (TARGET, "log_price")]
    return [c for c in ENGINEERED_COLUMNS if c not in (TARGET, "log_price")]


# In log-price space the standard deviation IS the relative price
# variation (a std of 0.05 ≈ ±5% swings). PPGs whose log-price barely
# moves can't identify an elasticity, so we skip them with a clear
# reason rather than producing a meaningless coefficient.
LOG_PRICE_STD_FLOOR = 0.01
MIN_ROWS_FOR_FIT = 20


def _log_price_std(slice_: pd.DataFrame) -> float:
    if "log_price" not in slice_.columns or slice_["log_price"].empty:
        return 0.0
    return float(slice_["log_price"].std(ddof=0))


def _gate_slice(slice_: pd.DataFrame) -> str | None:
    """Return a skip-reason string if the slice fails any pre-fit gate, else None."""
    if len(slice_) < MIN_ROWS_FOR_FIT:
        return f"insufficient rows ({len(slice_)})"
    std = _log_price_std(slice_)
    if std < LOG_PRICE_STD_FLOOR:
        return f"price_variance_below_floor (std_log_price={std:.4f})"
    return None


def _test_wape(fit: ElasticityFit) -> float:
    val = fit.diagnostics.get("test_wape")
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return float("inf")
    return float(val)


# Hard cap on the |elasticity| any candidate may have to enter the
# winner pool. Beyond this we have no business operating on the number
# (price-ladder optimisation will produce absurd recommendations). 8 is
# 2x the toothpaste benchmark high and well outside any cited CPG band.
WINNER_MAGNITUDE_CEILING = 8.0


def _pick_winner(attempts: list[ElasticityFit]) -> ElasticityFit:
    """Lowest test WAPE among sign-correct AND in-magnitude fits; else
    relax constraints in order.

    Selection order:
      1. sign_ok AND |ε| <= ceiling   (the safe pool)
      2. sign_ok only                 (in-band but huge)
      3. anything                     (nothing was sane; pick lowest WAPE)
    """
    sign_ok = [a for a in attempts if a.sign_ok]
    in_band = [a for a in sign_ok if abs(a.own_elasticity) <= WINNER_MAGNITUDE_CEILING]
    pool = in_band or sign_ok or attempts
    return min(pool, key=_test_wape)


def _fit_one_ppg(ppg_id: str, frame: pd.DataFrame, controls: list[str]) -> dict:
    train, test = chronological_split(frame, test_ratio=0.2)
    attempts: list[ElasticityFit] = []

    loglog = fit_loglog(ppg_id, train, controls, test=test)
    attempts.append(loglog)

    retried = False
    if not loglog.sign_ok:
        retried = True
        attempts.append(fit_semilog(ppg_id, train, controls, test=test))

    attempts.append(fit_lightgbm(ppg_id, train, controls, test=test))

    winner = _pick_winner(attempts)
    return {
        "ppg_id": ppg_id,
        "winner_model": winner.model,
        "sign_retry_fired": retried,
        "attempts": [a.to_dict() for a in attempts],
        "winner": winner.to_dict(),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }


class ModelingAgent(Agent):
    name = "modeling"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        run_dir = Path(run.run_dir)

        feats = await asyncio.to_thread(_load_features, run_dir)
        eligible = _load_eligible_ppgs(run_dir)
        controls = _load_controls(run_dir)
        has_grain_unit = "grain_unit" in feats.columns

        await self.emit(
            run,
            "tool_called",
            {
                "tool": "load_inputs",
                "eligible_ppgs": len(eligible),
                "controls": len(controls),
                "grain_units": (
                    int(feats["grain_unit"].nunique()) if has_grain_unit else 1
                ),
            },
        )

        per_ppg: list[dict] = []
        preflight: list[dict] = []
        # When grain_unit is present (store-level grain), we iterate one
        # model per (grain_unit, ppg) cell. At the historical PPG_WEEK
        # grain there's no grain_unit column and we keep the existing
        # behaviour (one model per PPG).
        for ppg_id in eligible:
            ppg_slice = feats[feats["ppg_id"] == ppg_id]
            if ppg_slice.empty:
                continue
            if has_grain_unit:
                cells = [(unit, sub) for unit, sub in ppg_slice.groupby("grain_unit")]
            else:
                cells = [(None, ppg_slice)]

            for unit_id, slice_ in cells:
                cell_id = ppg_id if unit_id is None else f"{ppg_id}@{unit_id}"
                reason = _gate_slice(slice_)
                preflight.append(
                    {
                        "ppg_id": ppg_id,
                        "grain_unit": unit_id,
                        "cell_id": cell_id,
                        "n_rows": int(len(slice_)),
                        "std_log_price": _log_price_std(slice_),
                        "competitor_coverage": (
                            float(slice_["log_competitor_price"].notna().mean())
                            if "log_competitor_price" in slice_.columns and len(slice_)
                            else 0.0
                        ),
                        "skip_reason": reason,
                    }
                )
                if reason is not None:
                    per_ppg.append(
                        {
                            "ppg_id": ppg_id,
                            "grain_unit": unit_id,
                            "winner_model": "skipped",
                            "sign_retry_fired": False,
                            "attempts": [],
                            "winner": None,
                            "skip_reason": reason,
                        }
                    )
                    continue
                row = await asyncio.to_thread(_fit_one_ppg, ppg_id, slice_, controls)
                row["grain_unit"] = unit_id
                per_ppg.append(row)
                await self.emit(
                    run,
                    "tool_called",
                    {
                        "tool": "fit_candidates",
                        "ppg_id": ppg_id,
                        "grain_unit": unit_id,
                        "winner": row["winner_model"],
                        "n_candidates": len(row["attempts"]),
                        "retried": row["sign_retry_fired"],
                    },
                )

        n_correct = sum(1 for r in per_ppg if r["winner"] and r["winner"]["sign_ok"])
        n_retries = sum(1 for r in per_ppg if r["sign_retry_fired"])
        n_skipped = sum(1 for r in per_ppg if r["winner_model"] == "skipped")
        n_robust_refit = sum(
            1 for r in per_ppg
            if r.get("winner")
            and "robust_refit" in (r["winner"].get("diagnostics") or {})
        )
        skip_reason_counts: dict[str, int] = {}
        for r in per_ppg:
            if r["winner_model"] == "skipped" and r.get("skip_reason"):
                key = r["skip_reason"].split(" (")[0]
                skip_reason_counts[key] = skip_reason_counts.get(key, 0) + 1

        posterior_payload = _compute_hierarchical(per_ppg)
        posterior_path = run_dir / "hierarchical_posterior.json"
        posterior_path.write_text(json.dumps(posterior_payload, indent=2, default=float))
        result.artifacts.append(
            ArtifactRef(
                path=str(posterior_path),
                mime="application/json",
                agent=self.name,
                name=posterior_path.name,
            )
        )

        shap_rows = _collect_shap(per_ppg)
        shap_path = run_dir / "shap_per_ppg.json"
        shap_path.write_text(json.dumps(shap_rows, indent=2, default=float))
        result.artifacts.append(
            ArtifactRef(
                path=str(shap_path),
                mime="application/json",
                agent=self.name,
                name=shap_path.name,
            )
        )

        fva = _collect_fitted_vs_actual(per_ppg, feats, controls)
        fva_path = run_dir / "fitted_vs_actual.json"
        fva_path.write_text(json.dumps(fva, indent=2, default=float))
        result.artifacts.append(
            ArtifactRef(
                path=str(fva_path),
                mime="application/json",
                agent=self.name,
                name=fva_path.name,
            )
        )

        preflight_path = run_dir / "modeling_preflight.json"
        preflight_path.write_text(
            json.dumps(
                {
                    "n_total": len(preflight),
                    "n_skipped": sum(1 for p in preflight if p["skip_reason"]),
                    "skip_reasons": skip_reason_counts,
                    "thresholds": {
                        "min_rows_for_fit": MIN_ROWS_FOR_FIT,
                        "log_price_std_floor": LOG_PRICE_STD_FLOOR,
                    },
                    "per_ppg": preflight,
                },
                indent=2,
                default=str,
            )
        )
        result.artifacts.append(
            ArtifactRef(
                path=str(preflight_path),
                mime="application/json",
                agent=self.name,
                name=preflight_path.name,
            )
        )

        results_blob = {
            "controls_used": controls,
            "per_ppg": per_ppg,
            "n_correct_sign": n_correct,
            "n_retries": n_retries,
            "n_skipped": n_skipped,
            "n_robust_refit": n_robust_refit,
            "skip_reasons": skip_reason_counts,
            "n_total": len(per_ppg),
            "model_pool": ["loglog_ols", "semilog_ols", "lightgbm"],
        }
        results_path = run_dir / "modeling_results.json"
        results_path.write_text(json.dumps(results_blob, indent=2))
        result.artifacts.append(
            ArtifactRef(
                path=str(results_path),
                mime="application/json",
                agent=self.name,
                name=results_path.name,
            )
        )

        compact = [
            {
                "ppg_id": r["ppg_id"],
                "grain_unit": r.get("grain_unit"),
                "model": r["winner_model"],
                "own_elasticity": r["winner"]["own_elasticity"] if r["winner"] else None,
                "std_err": r["winner"]["std_err"] if r["winner"] else None,
                "r_squared": r["winner"]["r_squared"] if r["winner"] else None,
                "test_wape": (r["winner"]["diagnostics"].get("test_wape") if r["winner"] else None),
                "n_obs": r["winner"]["n_obs"] if r["winner"] else 0,
                "sign_ok": r["winner"]["sign_ok"] if r["winner"] else False,
                "sign_retry_fired": r["sign_retry_fired"],
            }
            for r in per_ppg
        ]
        # At a store-grain we need a chain-level view for downstream agents
        # (decomposition / validation / optimization all assume one row per
        # PPG). Inverse-variance pool the per-store estimates back to PPG
        # using the same DerSimonian-Laird machinery already in core.models.
        # Per-store rows stay in elasticity_per_ppg.json for the UI drill-down.
        compact_pooled = (
            _pool_per_store_to_ppg(per_ppg) if has_grain_unit else None
        )
        compact_path = run_dir / "elasticity_per_ppg.json"
        compact_path.write_text(json.dumps(compact, indent=2))
        result.artifacts.append(
            ArtifactRef(
                path=str(compact_path),
                mime="application/json",
                agent=self.name,
                name=compact_path.name,
            )
        )
        if compact_pooled is not None:
            pooled_path = run_dir / "elasticity_per_ppg_pooled.json"
            pooled_path.write_text(json.dumps(compact_pooled, indent=2))
            result.artifacts.append(
                ArtifactRef(
                    path=str(pooled_path),
                    mime="application/json",
                    agent=self.name,
                    name=pooled_path.name,
                )
            )

        narrative, concerns = self._narrate(result, compact, n_correct, n_retries, len(per_ppg))

        result.outputs = {
            "n_total": len(per_ppg),
            "n_correct_sign": n_correct,
            "n_retries": n_retries,
            "n_skipped": n_skipped,
            "n_robust_refit": n_robust_refit,
            "skip_reasons": skip_reason_counts,
            "winners_by_family": _winners_by_family(per_ppg),
            "n_shap": len(shap_rows),
            "n_shrunk": int(posterior_payload["n_studies"]),
            "tau_squared": float(posterior_payload["tau_squared"]),
        }
        result.reasoning = narrative or (
            f"Recovered correct elasticity sign for {n_correct}/{len(per_ppg)} eligible PPGs; "
            f"semi-log retry fired on {n_retries}."
        )
        result.confidence = (n_correct / len(per_ppg)) if per_ppg else 0.0
        if concerns:
            result.outputs["concerns"] = concerns

    def _narrate(
        self,
        result: AgentResult,
        compact: list[dict],
        n_correct: int,
        n_retries: int,
        n_total: int,
    ) -> tuple[str, list[dict]]:
        user_payload = json.dumps(
            {
                "per_ppg": compact,
                "n_correct_sign": n_correct,
                "n_retries": n_retries,
                "n_total": n_total,
            }
        )
        try:
            resp = self.call_llm(
                result,
                system=SYSTEM_PROMPT,
                user=user_payload,
                max_tokens=600,
                label="elasticity-narrative",
            )
            if self._is_dry_run(resp):
                return "", []
            blob = json.loads(resp.text)
            return str(blob.get("narrative", "")), list(blob.get("concerns", []))
        except (json.JSONDecodeError, ValueError) as exc:
            self.log.warning("modeling: LLM narrative parse failed: %s", exc)
            return "", []


def _winners_by_family(per_ppg: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in per_ppg:
        counts[r["winner_model"]] = counts.get(r["winner_model"], 0) + 1
    return counts


def _pool_per_store_to_ppg(per_ppg: list[dict]) -> list[dict]:
    """Inverse-variance pool per-store cell estimates back to PPG.

    For each PPG, take every cell whose winner has a finite (elasticity,
    std_err), pool via inverse-variance weighting, and emit one chain-
    level row with the pooled point estimate, pooled SE, and the count
    of stores contributing. Downstream agents (decomposition / validation
    / optimization) consume this so they don't need to reason about
    store-level rows.
    """
    by_ppg: dict[str, list[tuple[float, float, int, str]]] = {}
    for r in per_ppg:
        winner = r.get("winner")
        if not winner:
            continue
        e = winner.get("own_elasticity")
        s = winner.get("std_err")
        if e is None or s is None:
            continue
        try:
            e_f, s_f = float(e), float(s)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(e_f) and math.isfinite(s_f)) or s_f <= 0:
            continue
        by_ppg.setdefault(r["ppg_id"], []).append(
            (e_f, s_f, int(winner.get("n_obs") or 0), winner.get("model", ""))
        )

    pooled: list[dict] = []
    for ppg_id, cells in sorted(by_ppg.items()):
        weights = [1.0 / (s * s) for _, s, _, _ in cells]
        sum_w = sum(weights)
        if sum_w <= 0:
            continue
        pooled_e = sum(w * e for (e, _, _, _), w in zip(cells, weights)) / sum_w
        pooled_se = (1.0 / sum_w) ** 0.5
        # Median test-wape across cells is a more robust headline than a mean.
        test_wapes_sorted = sorted(c[2] for c in cells)
        pooled.append(
            {
                "ppg_id": ppg_id,
                "grain_unit": "pooled",
                "model": "store_inverse_variance_pool",
                "own_elasticity": pooled_e,
                "std_err": pooled_se,
                "r_squared": None,
                "test_wape": None,
                "n_obs": sum(c[2] for c in cells),
                "n_stores_pooled": len(cells),
                "sign_ok": pooled_e < 0,
                "sign_retry_fired": False,
            }
        )
    return pooled


def _compute_hierarchical(per_ppg: list[dict]) -> dict:
    """Pool the per-PPG winners with empirical-Bayes shrinkage.

    Only fits that report a finite std err contribute to the prior — that
    drops LightGBM winners (whose std err is the across-row dispersion of
    a numerical-derivative elasticity, not a sampling SE) and any skipped
    PPGs. The agent surfaces both the OLS point estimate and the shrunken
    posterior so the UI can render before/after.

    At a store-grain we'd otherwise feed N=stores×PPGs studies into the
    prior, which doesn't match the "one study per PPG" framing the UI
    assumes. Pre-collapse to one row per PPG by picking the cell with the
    tightest SE so the shrinkage operates on PPG-level studies.
    """
    triples: list[tuple[str, float, float]] = []
    best_by_ppg: dict[str, tuple[float, float]] = {}
    for r in per_ppg:
        winner = r.get("winner")
        if not winner:
            continue
        if winner["model"] not in ("loglog_ols", "semilog_ols"):
            continue
        ppg_id = r["ppg_id"]
        e = float(winner["own_elasticity"])
        s = float(winner["std_err"])
        if not (math.isfinite(e) and math.isfinite(s)) or s <= 0:
            continue
        prev = best_by_ppg.get(ppg_id)
        if prev is None or s < prev[1]:
            best_by_ppg[ppg_id] = (e, s)
    for ppg_id, (e, s) in best_by_ppg.items():
        triples.append((ppg_id, e, s))
    return to_payload(shrink(triples))


def _collect_fitted_vs_actual(
    per_ppg: list[dict], feats: pd.DataFrame, controls: list[str]
) -> list[dict]:
    """Predict on each PPG's full feature frame using its winning model.

    Emits one row per PPG containing parallel arrays of observed log_units,
    predicted log_units, and a train/test split flag aligned to the same
    chronological 80/20 the modelling agent used. The UI renders this as
    a fitted-vs-actual scatter with train/test colouring.
    """
    rows: list[dict] = []
    for row in per_ppg:
        winner = row.get("winner")
        if not winner or row["winner_model"] == "skipped":
            continue
        ppg_id = row["ppg_id"]
        unit_id = row.get("grain_unit")
        if unit_id is not None and "grain_unit" in feats.columns:
            slice_ = feats[(feats["ppg_id"] == ppg_id) & (feats["grain_unit"] == unit_id)]
        else:
            slice_ = feats[feats["ppg_id"] == ppg_id]
        slice_ = slice_.sort_values("week_start").reset_index(drop=True)
        if slice_.empty or "log_units" not in slice_.columns:
            continue
        try:
            predictor = build_predictor(row, slice_, controls, test_ratio=0.2)
            pred_log = predictor.predict_log(slice_)
        except (KeyError, ValueError, RuntimeError) as exc:
            import logging
            logging.getLogger("agent.modeling").warning(
                "modeling: skipping fitted-vs-actual for %s: %s", ppg_id, exc
            )
            continue
        observed_log = slice_["log_units"].astype(float).to_numpy()
        n_train = int(row.get("n_train") or len(slice_))
        split = ["train"] * n_train + ["test"] * max(0, len(slice_) - n_train)
        split = split[: len(slice_)]
        rows.append(
            {
                "ppg_id": ppg_id,
                "model": row["winner_model"],
                "weeks": slice_["week_start"].astype(str).tolist()
                if "week_start" in slice_.columns
                else [str(i) for i in range(len(slice_))],
                "observed_log": observed_log.tolist(),
                "predicted_log": [float(v) for v in pred_log],
                "observed_units": [float(v) for v in np.exp(observed_log)],
                "predicted_units": [float(v) for v in np.exp(pred_log)],
                "split": split,
                "n_train": n_train,
                "n_test": int(row.get("n_test") or max(0, len(slice_) - n_train)),
            }
        )
    return rows


def _collect_shap(per_ppg: list[dict]) -> list[dict]:
    """Strip the winner's SHAP block out of diagnostics into a per-PPG list.

    The candidates table can render mean |SHAP| for every PPG that produced
    a real fit; we omit skipped PPGs and any winner whose fitter didn't emit
    a SHAP block (defensive — every current fitter does).
    """
    rows: list[dict] = []
    for r in per_ppg:
        winner = r.get("winner")
        if not winner:
            continue
        shap = winner.get("diagnostics", {}).get("shap")
        if not shap:
            continue
        rows.append(
            {
                "ppg_id": r["ppg_id"],
                "model": winner["model"],
                "shap": shap,
            }
        )
    return rows
