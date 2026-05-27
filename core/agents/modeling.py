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

import core.models.library  # noqa: F401 — register model plugins
from core.agents.base import Agent
from core.config import get_settings
from core.features.engineering import ENGINEERED_COLUMNS, TARGET
from core.models.base import ElasticityFit
from core.models.bayes_hier import shrink, to_payload
from core.models.library import registry as model_registry
from core.models.library.base import FitContext
from core.models.library.diagnostics import profile as data_profile
from core.models.lightgbm_model import fit_lightgbm
from core.models.loglog_ols import fit_loglog
from core.models.metrics import chronological_split
from core.models.predictor import build_predictor
from core.models.result import ProblemType, to_elasticity_fit
from core.models.router.escalation import run_escalation, run_forecast_escalation
from core.models.router.llm_router import LLMRouter
from core.models.router.rules import DeterministicRouter
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


def _resolve_config(run: RunState, settings):
    """Merge per-run ``run.options['modeling']`` overrides over the global
    model-library / router config, validating the result. Lets the rerun loop
    re-model with a different enabled set, router mode, or problem type."""
    opts: dict = {}
    if isinstance(run.options, dict):
        raw = run.options.get("modeling")
        if isinstance(raw, dict):
            opts = raw
    lib_fields = type(settings.model_library).model_fields
    rtr_fields = type(settings.router).model_fields
    lib_updates = {k: v for k, v in opts.items() if k in lib_fields}
    rtr_updates = {k: v for k, v in opts.items() if k in rtr_fields}
    lib = settings.model_library
    rtr = settings.router
    if lib_updates:
        lib = type(lib).model_validate({**lib.model_dump(), **lib_updates})
    if rtr_updates:
        rtr = type(rtr).model_validate({**rtr.model_dump(), **rtr_updates})
    return lib, rtr


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


def _gate_slice(
    slice_: pd.DataFrame,
    min_rows: int = MIN_ROWS_FOR_FIT,
    std_floor: float = LOG_PRICE_STD_FLOOR,
) -> str | None:
    """Return a skip-reason string if the slice fails any pre-fit gate, else None."""
    if len(slice_) < min_rows:
        return f"insufficient rows ({len(slice_)})"
    std = _log_price_std(slice_)
    if std < std_floor:
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


def _effective_enabled(lib) -> set[str]:
    """Available registry keys narrowed by the config allow/deny lists."""
    avail = model_registry.available_keys()
    enabled = set(lib.enabled_models)
    out = (avail & enabled) if enabled else set(avail)
    return out - set(lib.disabled_models)


def _fit_one_ppg_routed(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    candidates: list[str],
    *,
    grain: str,
    problem: ProblemType,
    hparams: dict[str, dict],
    max_candidates: int,
    magnitude_ceiling: float,
    wape_floor: float,
    rng_seed: int,
) -> dict:
    """Run the router-chosen candidate list through the escalation loop and
    shape the result like ``_fit_one_ppg`` so every downstream consumer is
    unaffected by which path produced the row."""
    train, test = chronological_split(frame, test_ratio=0.2)
    ctx = FitContext(
        ppg_id=ppg_id,
        controls=controls,
        test=test,
        grain=grain,
        problem_type=problem,
        rng_seed=rng_seed,
    )
    esc = run_escalation(
        candidates,
        train,
        ctx,
        max_candidates=max_candidates,
        magnitude_ceiling=magnitude_ceiling,
        wape_floor=wape_floor,
        hparams=hparams,
    )
    row: dict = {
        "ppg_id": ppg_id,
        "winner_model": esc.winner.model if esc.winner else "no_fit",
        "sign_retry_fired": False,
        "attempts": [a.to_dict() for a in esc.attempts],
        "winner": esc.winner.to_dict() if esc.winner else None,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "router": {
            "problem_type": problem.value,
            "candidates": esc.candidates,
            "non_scalar": esc.non_scalar,
            "errors": esc.errors,
        },
    }
    if esc.winner is None:
        row["skip_reason"] = "router_no_scalar_fit"
    return row


def _forecast_one_ppg_routed(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    candidates: list[str],
    *,
    grain: str,
    hparams: dict[str, dict],
    max_candidates: int,
    rng_seed: int,
) -> tuple[dict, dict | None]:
    """Forecast-problem variant: rank candidates by hold-out forecast WAPE and
    return (modeling row, forecast entry). The winner may carry an elasticity
    (ARIMAX/SARIMAX/state-space) or be forecast-only (ETS/Holt-Winters)."""
    train, test = chronological_split(frame, test_ratio=0.2)
    ctx = FitContext(
        ppg_id=ppg_id,
        controls=controls,
        test=test,
        grain=grain,
        problem_type=ProblemType.FORECAST,
        rng_seed=rng_seed,
    )
    fc = run_forecast_escalation(
        candidates, train, ctx, max_candidates=max_candidates, hparams=hparams
    )
    winner = fc.winner
    ef = to_elasticity_fit(winner) if winner else None
    row: dict = {
        "ppg_id": ppg_id,
        "winner_model": winner.model if winner else "no_forecast",
        "sign_retry_fired": False,
        "attempts": [
            {"model": a.model, "diagnostics": {"test_wape": a.diagnostics.get("test_wape")}}
            for a in fc.attempts
        ],
        "winner": ef.to_dict() if ef else None,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "router": {
            "problem_type": ProblemType.FORECAST.value,
            "candidates": fc.candidates,
            "errors": fc.errors,
        },
    }
    if winner is None:
        row["skip_reason"] = "router_no_forecast"
    entry: dict | None = None
    if winner is not None and winner.forecast is not None:
        entry = {
            "ppg_id": ppg_id,
            "model": winner.model,
            "test_wape": winner.diagnostics.get("test_wape"),
            "own_elasticity": winner.own_elasticity,
            "forecast": winner.forecast.to_dict(),
        }
    return row, entry


def _system_one_ppg_routed(
    ppg_id: str,
    feats: pd.DataFrame,
    controls: list[str],
    candidates: list[str],
    *,
    grain: str,
    hparams: dict[str, dict],
    rng_seed: int,
) -> tuple[dict, dict | None]:
    """DEMAND_SYSTEM variant: fit one cross-price equation for the target PPG
    against EVERY PPG's price (the full ``feats`` frame), returning (modeling
    row, the PPG's cross-price row)."""
    key = next(
        (k for k in candidates if model_registry.has(k) and model_registry.get(k).is_available()),
        None,
    )
    base_router = {"problem_type": ProblemType.DEMAND_SYSTEM.value, "candidates": candidates}
    if key is None:
        return {
            "ppg_id": ppg_id,
            "winner_model": "no_system",
            "sign_retry_fired": False,
            "attempts": [],
            "winner": None,
            "n_train": 0,
            "n_test": 0,
            "skip_reason": "router_no_system",
            "router": {**base_router, "errors": {}},
        }, None

    ctx = FitContext(
        ppg_id=ppg_id,
        controls=controls,
        test=None,
        grain=grain,
        problem_type=ProblemType.DEMAND_SYSTEM,
        hparams=hparams.get(key, {}),
        rng_seed=rng_seed,
    )
    try:
        result = model_registry.get(key).fit(feats, ctx)
    except Exception as exc:  # noqa: BLE001
        return {
            "ppg_id": ppg_id,
            "winner_model": "no_system",
            "sign_retry_fired": False,
            "attempts": [],
            "winner": None,
            "n_train": 0,
            "n_test": 0,
            "skip_reason": "system_error",
            "router": {**base_router, "errors": {key: f"{type(exc).__name__}: {exc}"}},
        }, None

    ef = to_elasticity_fit(result)
    row: dict = {
        "ppg_id": ppg_id,
        "winner_model": key,
        "sign_retry_fired": False,
        "attempts": [ef.to_dict()] if ef else [],
        "winner": ef.to_dict() if ef else None,
        "n_train": int(result.n_obs),
        "n_test": int(result.diagnostics.get("n_test", 0)),
        "router": {**base_router, "errors": {}},
    }
    cross_entry: dict | None = None
    if result.cross_price and ppg_id in result.cross_price:
        cross_entry = {
            "ppg_id": ppg_id,
            "model": key,
            "own_elasticity": result.own_elasticity,
            "test_wape": result.diagnostics.get("test_wape"),
            "cross": result.cross_price[ppg_id],
        }
    return row, cross_entry


class ModelingAgent(Agent):
    name = "modeling"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        run_dir = Path(run.run_dir)

        settings = get_settings()
        lib, rtr = _resolve_config(run, settings)
        hparams = settings.model_hparams.model_dump()
        routed = lib.router_enabled
        grain_str = settings.modelling_grain.value
        dry_run = self.llm.provider.value == "dry_run"
        problem = ProblemType(rtr.default_problem_type)
        is_forecast = routed and problem == ProblemType.FORECAST
        is_demand_system = routed and problem == ProblemType.DEMAND_SYSTEM
        eff_enabled = _effective_enabled(lib) if routed else set()
        router_decisions: list[dict] = []
        forecasts: list[dict] = []
        cross_price_rows: list[dict] = []

        feats = await asyncio.to_thread(_load_features, run_dir)
        eligible = _load_eligible_ppgs(run_dir)
        controls = _load_controls(run_dir)
        has_grain_unit = "grain_unit" in feats.columns

        # When the features frame's ppg_id column carries brand /
        # category labels (non-PPG grain), the ppg_selection eligibility
        # list is keyed by PPG_AUTO_xx and won't match anything here.
        # Fall back to every distinct unit observed in the features
        # frame; the per-cell gates downstream filter out the
        # under-sized ones.
        feature_ppg_ids = set(feats["ppg_id"].astype(str).unique())
        if eligible and not (set(eligible) & feature_ppg_ids):
            self.log.info(
                "modeling: ppg_selection ids don't match features grain — "
                "using all %d units from features.csv",
                len(feature_ppg_ids),
            )
            eligible = sorted(feature_ppg_ids)

        # Cells the modelling loop will visit (after restricting to eligible
        # PPGs). At chain grain this is just len(eligible); at store grain
        # it's the number of distinct (ppg_id, grain_unit) keys actually
        # observed in the feature frame.
        if has_grain_unit:
            total_cells = int(
                feats[feats["ppg_id"].isin(eligible)]
                .groupby(["ppg_id", "grain_unit"])
                .ngroups
            )
        else:
            total_cells = int(
                feats[feats["ppg_id"].isin(eligible)]["ppg_id"].nunique()
            )

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
                "total_cells": total_cells,
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
                reason = _gate_slice(slice_, lib.min_rows_for_fit, lib.log_price_std_floor)
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
                    await self.emit(
                        run,
                        "tool_called",
                        {
                            "tool": "fit_skipped",
                            "ppg_id": ppg_id,
                            "grain_unit": unit_id,
                            "reason": reason.split(" (")[0],
                        },
                    )
                    continue
                if routed:
                    prof = data_profile(
                        slice_,
                        controls,
                        grain=grain_str,
                        test_ratio=0.2,
                        seasonality_min_length=rtr.seasonality_min_length,
                    )
                    candidates, router_used = self._route(
                        result, problem, prof, eff_enabled, dry_run, rtr
                    )
                    router_decisions.append(
                        {
                            "cell_id": cell_id,
                            "ppg_id": ppg_id,
                            "grain_unit": unit_id,
                            "router": router_used,
                            "problem_type": problem.value,
                            "candidates": candidates,
                            "profile": prof.to_dict(),
                        }
                    )
                    if is_forecast:
                        row, fc_entry = await asyncio.to_thread(
                            _forecast_one_ppg_routed,
                            ppg_id,
                            slice_,
                            controls,
                            candidates,
                            grain=grain_str,
                            hparams=hparams,
                            max_candidates=lib.max_candidates,
                            rng_seed=lib.rng_seed,
                        )
                        if fc_entry is not None:
                            fc_entry["grain_unit"] = unit_id
                            forecasts.append(fc_entry)
                    elif is_demand_system:
                        row, x_entry = await asyncio.to_thread(
                            _system_one_ppg_routed,
                            ppg_id,
                            feats,
                            controls,
                            candidates,
                            grain=grain_str,
                            hparams=hparams,
                            rng_seed=lib.rng_seed,
                        )
                        if x_entry is not None:
                            cross_price_rows.append(x_entry)
                    else:
                        row = await asyncio.to_thread(
                            _fit_one_ppg_routed,
                            ppg_id,
                            slice_,
                            controls,
                            candidates,
                            grain=grain_str,
                            problem=problem,
                            hparams=hparams,
                            max_candidates=lib.max_candidates,
                            magnitude_ceiling=lib.winner_magnitude_ceiling,
                            wape_floor=lib.wape_escalate_floor,
                            rng_seed=lib.rng_seed,
                        )
                else:
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
                        "min_rows_for_fit": lib.min_rows_for_fit,
                        "log_price_std_floor": lib.log_price_std_floor,
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

        if is_forecast:
            forecasts_path = run_dir / "forecasts.json"
            forecasts_path.write_text(
                json.dumps(
                    {"problem_type": problem.value, "n_ppg": len(forecasts), "per_ppg": forecasts},
                    indent=2,
                    default=float,
                )
            )
            result.artifacts.append(
                ArtifactRef(
                    path=str(forecasts_path),
                    mime="application/json",
                    agent=self.name,
                    name=forecasts_path.name,
                )
            )

        if is_demand_system:
            matrix = {r["ppg_id"]: r["cross"] for r in cross_price_rows}
            own = {r["ppg_id"]: r["own_elasticity"] for r in cross_price_rows}
            ppg_order = sorted(matrix)
            cross_path = run_dir / "cross_price_matrix.json"
            cross_path.write_text(
                json.dumps(
                    {
                        "problem_type": problem.value,
                        "ppgs": ppg_order,
                        "own_elasticity": own,
                        "matrix": matrix,
                    },
                    indent=2,
                    default=float,
                )
            )
            result.artifacts.append(
                ArtifactRef(
                    path=str(cross_path),
                    mime="application/json",
                    agent=self.name,
                    name=cross_path.name,
                )
            )

        if routed:
            router_path = run_dir / "router_decision.json"
            router_path.write_text(
                json.dumps(
                    {
                        "router_mode": rtr.mode,
                        "problem_type": problem.value,
                        "enabled_models": sorted(eff_enabled),
                        "decisions": router_decisions,
                    },
                    indent=2,
                    default=str,
                )
            )
            result.artifacts.append(
                ArtifactRef(
                    path=str(router_path),
                    mime="application/json",
                    agent=self.name,
                    name=router_path.name,
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
            "router_enabled": routed,
            "model_pool": (
                sorted(eff_enabled) if routed else ["loglog_ols", "semilog_ols", "lightgbm"]
            ),
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

    def _route(
        self, result, problem, prof, enabled, dry_run, rtr
    ) -> tuple[list[str], str]:
        """Pick an ordered candidate set. Deterministic rules in dry-run or
        ``mode='rules'``; otherwise the LLM router with rules fallback."""
        rules = DeterministicRouter(rtr)
        if rtr.mode == "rules" or dry_run:
            return rules.select(problem, prof, enabled=enabled), "rules"
        system, user = LLMRouter.build_prompt(problem, prof, sorted(enabled))
        try:
            resp = self.call_llm(
                result, system=system, user=user, max_tokens=400, label="model-router"
            )
        except (ValueError, RuntimeError) as exc:
            self.log.warning("modeling: router LLM call failed: %s", exc)
            return rules.select(problem, prof, enabled=enabled), "rules"
        if self._is_dry_run(resp):
            return rules.select(problem, prof, enabled=enabled), "rules"
        cands = LLMRouter(rules, rtr).select(
            problem, prof, enabled=enabled, llm_text=resp.text
        )
        return cands, "llm"


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
