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

import pandas as pd

from core.agents.base import Agent
from core.features.engineering import ENGINEERED_COLUMNS, TARGET
from core.models.base import ElasticityFit
from core.models.lightgbm_model import fit_lightgbm
from core.models.loglog_ols import fit_loglog
from core.models.metrics import chronological_split
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


def _test_wape(fit: ElasticityFit) -> float:
    val = fit.diagnostics.get("test_wape")
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return float("inf")
    return float(val)


def _pick_winner(attempts: list[ElasticityFit]) -> ElasticityFit:
    """Lowest test WAPE among sign-correct fits; else lowest WAPE overall."""
    sign_ok = [a for a in attempts if a.sign_ok]
    pool = sign_ok or attempts
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

        await self.emit(
            run,
            "tool_called",
            {"tool": "load_inputs", "eligible_ppgs": len(eligible), "controls": len(controls)},
        )

        per_ppg: list[dict] = []
        for ppg_id in eligible:
            slice_ = feats[feats["ppg_id"] == ppg_id]
            if len(slice_) < 20:
                per_ppg.append(
                    {
                        "ppg_id": ppg_id,
                        "winner_model": "skipped",
                        "sign_retry_fired": False,
                        "attempts": [],
                        "winner": None,
                        "skip_reason": f"insufficient rows ({len(slice_)})",
                    }
                )
                continue
            row = await asyncio.to_thread(_fit_one_ppg, ppg_id, slice_, controls)
            per_ppg.append(row)
            await self.emit(
                run,
                "tool_called",
                {
                    "tool": "fit_candidates",
                    "ppg_id": ppg_id,
                    "winner": row["winner_model"],
                    "n_candidates": len(row["attempts"]),
                    "retried": row["sign_retry_fired"],
                },
            )

        n_correct = sum(1 for r in per_ppg if r["winner"] and r["winner"]["sign_ok"])
        n_retries = sum(1 for r in per_ppg if r["sign_retry_fired"])
        n_skipped = sum(1 for r in per_ppg if r["winner_model"] == "skipped")

        results_blob = {
            "controls_used": controls,
            "per_ppg": per_ppg,
            "n_correct_sign": n_correct,
            "n_retries": n_retries,
            "n_skipped": n_skipped,
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

        narrative, concerns = self._narrate(result, compact, n_correct, n_retries, len(per_ppg))

        result.outputs = {
            "n_total": len(per_ppg),
            "n_correct_sign": n_correct,
            "n_retries": n_retries,
            "n_skipped": n_skipped,
            "winners_by_family": _winners_by_family(per_ppg),
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
            if resp.raw.get("dry_run"):
                return "", []
            blob = json.loads(resp.text)
            return str(blob.get("narrative", "")), list(blob.get("concerns", []))
        except (json.JSONDecodeError, ValueError):
            return "", []


def _winners_by_family(per_ppg: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in per_ppg:
        counts[r["winner_model"]] = counts.get(r["winner_model"], 0) + 1
    return counts
