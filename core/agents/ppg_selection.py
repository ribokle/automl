"""PPG-selection agent.

Reads the mapping artefact from the previous stage, scores each PPG for
modelling eligibility (volume / coverage / price variation / promo variation),
and decides which PPGs the downstream modelling stage should attempt. The LLM
provides a one-line business rationale per excluded PPG.

Outputs:
- ppg_selection.json  (per-PPG score + eligible flag + reasoning)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from core.agents.base import Agent
from core.orchestrator.state import AgentResult, ArtifactRef, RunState
from core.ppg.score import score_ppgs


SYSTEM_PROMPT = """You are the PPG-selection analyst. You receive an eligibility table
for every Price-Pack Group with a 0-1 score and a short metric-based reasoning. For each
EXCLUDED PPG, return a one-line business rationale (<=140 chars) suitable for an exec
summary. Do NOT change the eligible flag. Return STRICT JSON:
{"per_ppg": [{"ppg_id": "...", "exec_rationale": "..."}, ...]}. JSON only."""


def _dry_run_exec_rationales(scores: pd.DataFrame) -> dict:
    out = []
    for _, row in scores.iterrows():
        if not row["eligible"]:
            out.append(
                {
                    "ppg_id": row["ppg_id"],
                    "exec_rationale": f"Hold from modelling: {row['reasoning']}.",
                }
            )
    return {"per_ppg": out}


class PPGSelectionAgent(Agent):
    name = "ppg_selection"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        duckdb_path = Path(run.duckdb_path)
        run_dir = Path(run.run_dir)

        mapping_path = run_dir / "ppg_mapping.json"
        if not mapping_path.exists():
            raise RuntimeError("ppg_mapping.json not found - ppg_mapping stage must run first")
        mapping_blob = json.loads(mapping_path.read_text())
        assignments = pd.DataFrame(mapping_blob["assignments"])

        scores = await asyncio.to_thread(score_ppgs, duckdb_path, assignments)
        await self.emit(run, "tool_called", {"tool": "score_ppgs", "n_ppgs": int(len(scores))})

        llm_resp = await asyncio.to_thread(
            self.call_llm,
            result,
            system=SYSTEM_PROMPT,
            user=scores.to_json(orient="records"),
            max_tokens=900,
        )
        exec_rationales: dict
        try:
            exec_rationales = (
                json.loads(llm_resp.text)
                if not llm_resp.raw.get("dry_run")
                else _dry_run_exec_rationales(scores)
            )
        except (json.JSONDecodeError, ValueError):
            exec_rationales = _dry_run_exec_rationales(scores)
        rationale_lookup = {r["ppg_id"]: r.get("exec_rationale", "") for r in exec_rationales.get("per_ppg", [])}

        scores["exec_rationale"] = scores["ppg_id"].map(rationale_lookup).fillna("")
        out_path = run_dir / "ppg_selection.json"
        out_path.write_text(scores.to_json(orient="records", indent=2))
        result.artifacts.append(
            ArtifactRef(path=str(out_path), mime="application/json", agent=self.name, name="ppg_selection.json")
        )

        n_eligible = int(scores["eligible"].sum())
        result.outputs = {
            "n_ppgs": int(len(scores)),
            "n_eligible": n_eligible,
            "mean_score": round(float(scores["score"].mean()), 3),
        }
        result.reasoning = (
            f"Scored {len(scores)} PPGs on volume / coverage / price variation / promo "
            f"variation. {n_eligible} eligible for modelling, "
            f"{len(scores) - n_eligible} held out. Mean score {result.outputs['mean_score']:.2f}."
        )
        result.confidence = round(float(scores["score"].mean()), 3)
