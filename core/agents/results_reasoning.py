"""Results-reasoning agent.

Runs after the modelling agent. Re-reads ``modeling_results.json``, runs a
deterministic battery of sanity checks (sign, magnitude band, R², hold-out
WAPE), then asks the LLM to narrate the model-choice rationale per PPG and
flag PPGs that need analyst review before downstream simulation.

Outputs:
- ``results_reasoning.json`` per-PPG verdicts (pass / warn / fail per
  check, plus a one-line narrative).
- ``model_choice_summary.json`` compact summary the UI table renders
  (one row per PPG: PPG, winner, elasticity, WAPE, verdict).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from core.agents.base import Agent
from core.orchestrator.state import AgentResult, ArtifactRef, RunState


SYSTEM_PROMPT = """You are the modelling-results reviewer. You receive the
modelling_results.json blob: every candidate model fitted per PPG plus the
winner and its hold-out WAPE. For each PPG, return one sentence (<=140
chars) explaining WHY the winning model was chosen over the other
candidates, citing only numbers present in the inputs. Return STRICT JSON:
{"per_ppg": [{"ppg_id": "...", "rationale": "..."}, ...],
 "headline": "<=240 chars exec summary"}
JSON only, no prose."""


ELASTICITY_LOW = 0.3
ELASTICITY_HIGH = 6.0
R2_FLOOR = 0.30
WAPE_CEILING = 0.30


def _verdicts(per_ppg: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in per_ppg:
        ppg_id = row["ppg_id"]
        winner = row.get("winner")
        checks: list[dict[str, Any]] = []
        verdict = "pass"

        if row.get("winner_model") == "skipped" or winner is None:
            checks.append(
                {
                    "name": "fitted",
                    "status": "fail",
                    "detail": row.get("skip_reason", "no winning model"),
                }
            )
            out.append({"ppg_id": ppg_id, "verdict": "fail", "checks": checks})
            continue

        sign_ok = bool(winner["sign_ok"])
        elasticity_abs = abs(float(winner["own_elasticity"]))
        r2 = float(winner.get("r_squared", 0.0))
        test_wape = winner.get("diagnostics", {}).get("test_wape")

        checks.append(
            {
                "name": "sign_correct",
                "status": "pass" if sign_ok else "fail",
                "detail": f"own_elasticity={winner['own_elasticity']:.3f}",
            }
        )
        if not sign_ok:
            verdict = "fail"

        if ELASTICITY_LOW <= elasticity_abs <= ELASTICITY_HIGH:
            checks.append(
                {"name": "magnitude_band", "status": "pass", "detail": f"|ε|={elasticity_abs:.2f}"}
            )
        else:
            checks.append(
                {
                    "name": "magnitude_band",
                    "status": "warn",
                    "detail": f"|ε|={elasticity_abs:.2f} outside [{ELASTICITY_LOW},{ELASTICITY_HIGH}]",
                }
            )
            verdict = verdict if verdict == "fail" else "warn"

        checks.append(
            {
                "name": "r_squared_floor",
                "status": "pass" if r2 >= R2_FLOOR else "warn",
                "detail": f"R²={r2:.2f}",
            }
        )
        if r2 < R2_FLOOR:
            verdict = verdict if verdict == "fail" else "warn"

        if test_wape is None:
            checks.append(
                {"name": "holdout_wape", "status": "info", "detail": "no hold-out WAPE recorded"}
            )
        else:
            wape = float(test_wape)
            status = "pass" if wape <= WAPE_CEILING else "warn"
            checks.append(
                {"name": "holdout_wape", "status": status, "detail": f"WAPE={wape:.3f}"}
            )
            if status == "warn":
                verdict = verdict if verdict == "fail" else "warn"

        out.append(
            {
                "ppg_id": ppg_id,
                "verdict": verdict,
                "checks": checks,
                "winner_model": row["winner_model"],
                "own_elasticity": winner["own_elasticity"],
                "test_wape": test_wape,
            }
        )
    return out


class ResultsReasoningAgent(Agent):
    name = "results_reasoning"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        run_dir = Path(run.run_dir)
        results_path = run_dir / "modeling_results.json"
        if not results_path.exists():
            raise RuntimeError(
                "modeling_results.json missing — modelling must run before results_reasoning"
            )
        modelling = json.loads(results_path.read_text())
        per_ppg = modelling.get("per_ppg", [])

        verdicts = await asyncio.to_thread(_verdicts, per_ppg)
        await self.emit(run, "tool_called", {"tool": "compute_verdicts", "n": len(verdicts)})

        narrative_rationales, headline = self._narrate(result, modelling)

        rationale_by_id = {r["ppg_id"]: r["rationale"] for r in narrative_rationales}
        for v in verdicts:
            v["rationale"] = rationale_by_id.get(v["ppg_id"], "")

        n_pass = sum(1 for v in verdicts if v["verdict"] == "pass")
        n_warn = sum(1 for v in verdicts if v["verdict"] == "warn")
        n_fail = sum(1 for v in verdicts if v["verdict"] == "fail")

        verdict_blob = {
            "thresholds": {
                "elasticity_low": ELASTICITY_LOW,
                "elasticity_high": ELASTICITY_HIGH,
                "r_squared_floor": R2_FLOOR,
                "wape_ceiling": WAPE_CEILING,
            },
            "per_ppg": verdicts,
            "headline": headline,
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
        }
        verdict_path = run_dir / "results_reasoning.json"
        verdict_path.write_text(json.dumps(verdict_blob, indent=2))
        result.artifacts.append(
            ArtifactRef(
                path=str(verdict_path),
                mime="application/json",
                agent=self.name,
                name=verdict_path.name,
            )
        )

        summary_rows = [
            {
                "ppg_id": v["ppg_id"],
                "winner": v.get("winner_model", "skipped"),
                "own_elasticity": v.get("own_elasticity"),
                "test_wape": v.get("test_wape"),
                "verdict": v["verdict"],
                "rationale": v["rationale"],
            }
            for v in verdicts
        ]
        summary_path = run_dir / "model_choice_summary.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        result.artifacts.append(
            ArtifactRef(
                path=str(summary_path),
                mime="application/json",
                agent=self.name,
                name=summary_path.name,
            )
        )

        result.outputs = {
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
            "n_total": len(verdicts),
        }
        result.reasoning = headline or (
            f"{n_pass}/{len(verdicts)} PPGs pass all checks; "
            f"{n_warn} flagged warn, {n_fail} flagged fail."
        )
        result.confidence = (n_pass / len(verdicts)) if verdicts else 0.0

    def _narrate(
        self, result: AgentResult, modelling: dict
    ) -> tuple[list[dict], str]:
        user_payload = json.dumps(modelling)
        try:
            resp = self.call_llm(
                result,
                system=SYSTEM_PROMPT,
                user=user_payload,
                max_tokens=900,
                label="model-choice-rationale",
            )
            if resp.raw.get("dry_run"):
                return [], ""
            blob = json.loads(resp.text)
            return list(blob.get("per_ppg", [])), str(blob.get("headline", ""))
        except (json.JSONDecodeError, ValueError):
            return [], ""
