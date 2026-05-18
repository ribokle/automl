"""Modeling agent — log-log OLS with semi-log sign-retry.

Phase 3 first slice. For each eligible PPG (from ppg_selection.json):

1. Fit log-log OLS on the refined feature frame.
2. If the own-price elasticity has the wrong sign (β >= 0), refit semi-log
   on the same controls. The semi-log β is converted to an elasticity at
   the mean price so both fits are comparable.
3. Pick the winning fit: prefer log-log when its sign is correct; otherwise
   fall back to semi-log and record that the retry fired.

LightGBM + PyMC land in the next slice; the iterative-retry scaffolding is
designed so adding a third candidate is a one-line registry change.

Outputs:
- modeling_results.json   per-PPG: every fit attempted + the winner
- elasticity_per_ppg.json compact summary the UI / downstream agents read
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from core.agents.base import Agent
from core.features.engineering import ENGINEERED_COLUMNS, TARGET
from core.models.base import ElasticityFit
from core.models.loglog_ols import fit_loglog
from core.models.semilog_ols import fit_semilog
from core.orchestrator.state import AgentResult, ArtifactRef, RunState


SYSTEM_PROMPT = """You are the pricing-elasticity analyst. You receive a JSON
table of per-PPG OLS fits with their own-price elasticity, std error,
p-value, R², and whether the sign-retry from log-log to semi-log fired.
Return STRICT JSON of the form:
{"narrative": "<=320 chars explaining which PPGs the elasticities are
trustworthy for and where the model still struggles",
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
    """Refined kept-feature list minus the target and the price variable
    (which is always the elasticity primary, not a control)."""
    refine_path = run_dir / "feature_refine.json"
    if refine_path.exists():
        blob = json.loads(refine_path.read_text())
        kept = blob.get("kept") or []
        return [c for c in kept if c not in (TARGET, "log_price")]
    return [c for c in ENGINEERED_COLUMNS if c not in (TARGET, "log_price")]


def _fit_one_ppg(ppg_id: str, frame: pd.DataFrame, controls: list[str]) -> dict:
    """Run log-log; if sign is wrong, retry with semi-log. Return both
    attempts + a pointer to the winner."""
    attempts: list[ElasticityFit] = []
    primary = fit_loglog(ppg_id, frame, controls)
    attempts.append(primary)

    retried = False
    winner = primary
    if not primary.sign_ok:
        retried = True
        semilog = fit_semilog(ppg_id, frame, controls)
        attempts.append(semilog)
        if semilog.sign_ok:
            winner = semilog

    return {
        "ppg_id": ppg_id,
        "winner_model": winner.model,
        "sign_retry_fired": retried,
        "attempts": [a.to_dict() for a in attempts],
        "winner": winner.to_dict(),
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
            per_ppg.append(
                await asyncio.to_thread(_fit_one_ppg, ppg_id, slice_, controls)
            )
            await self.emit(
                run,
                "tool_called",
                {
                    "tool": "fit_elasticity",
                    "ppg_id": ppg_id,
                    "model": per_ppg[-1]["winner_model"],
                    "retried": per_ppg[-1]["sign_retry_fired"],
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
