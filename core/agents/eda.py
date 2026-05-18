"""EDA agent.

Reads the canonical mart, rolls up to PPG × week, and produces an EDA artefact
covering panel coverage, numeric distributions, missingness, the target-vs-candidate
relationship table, and the pairwise correlation matrix among candidate features.
The LLM narrates which signals look strongest; everything else is deterministic.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.agents.base import Agent
from core.features.eda import (
    missingness,
    numeric_summary,
    pairwise_corr,
    panel_overview,
    ppg_week_aggregate,
    target_relationship,
)
from core.orchestrator.state import AgentResult, ArtifactRef, RunState


SYSTEM_PROMPT = """You are a pricing EDA analyst. Given a panel overview, the target's
relationship to each candidate, and the candidates' pairwise correlations, return
STRICT JSON of the form
{"findings": ["bullet 1 (<=140 chars)", "bullet 2", ...], "narrative": "<=320 chars"}.
Focus on signals that matter for elasticity modelling. JSON only."""


_NUMERIC_COLS = [
    "units", "price", "base_price", "discount_depth", "tpr_share",
    "display_share", "feature_share", "distribution_acv", "competitor_price",
]


def _dry_run_summary(rel: list[dict], corr: dict[str, dict[str, float]]) -> dict:
    top = rel[:3]
    bullets = [
        f"Strongest target relationship: {top[0]['feature']} (ρ={top[0]['spearman']:+.2f})." if top else "No candidates passed the n>=30 floor.",
    ]
    if len(top) > 1:
        bullets.append(f"Next strongest: {top[1]['feature']} (ρ={top[1]['spearman']:+.2f}).")
    if corr:
        cols = list(corr.keys())
        worst = ("", "", 0.0)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                r = corr[a][b]
                if abs(r) > abs(worst[2]):
                    worst = (a, b, r)
        if worst[0]:
            bullets.append(f"Highest off-diagonal |corr|: {worst[0]} ~ {worst[1]} (r={worst[2]:+.2f}).")
    return {
        "findings": bullets,
        "narrative": f"Panel ready for feature engineering; {len(rel)} candidates ranked by target relationship.",
    }


class EDAAgent(Agent):
    name = "eda"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        duckdb_path = Path(run.duckdb_path)
        run_dir = Path(run.run_dir)

        overview = await asyncio.to_thread(panel_overview, duckdb_path)
        await self.emit(run, "tool_called", {"tool": "panel_overview", "rows": int(overview.get("rows", 0))})

        panel = await asyncio.to_thread(ppg_week_aggregate, duckdb_path)
        await self.emit(run, "tool_called", {"tool": "ppg_week_aggregate", "rows": int(len(panel))})

        candidates_path = run_dir / "feature_candidates.json"
        if not candidates_path.exists():
            raise RuntimeError("feature_candidates.json not found — feature_selection must run first")
        cand_blob = json.loads(candidates_path.read_text())

        numeric_cols = [c for c in _NUMERIC_COLS if c in panel.columns]
        summary = numeric_summary(panel, numeric_cols)
        rel = target_relationship(panel, "units", [c for c in numeric_cols if c != "units"])
        corr = pairwise_corr(panel, [c for c in numeric_cols if c != "units"])
        miss = missingness(panel)

        user_prompt = json.dumps(
            {
                "overview": overview,
                "target": "units",
                "candidates": cand_blob.get("candidates", []),
                "target_relationship": rel,
                "pairwise_corr": corr,
            }
        )
        try:
            llm_resp = await asyncio.to_thread(
                self.call_llm,
                result,
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=400,
            )
            data = json.loads(llm_resp.text or "{}")
            if not isinstance(data.get("findings"), list):
                raise ValueError("missing findings")
        except Exception:  # noqa: BLE001
            data = _dry_run_summary(rel, corr)

        artifact = {
            "overview": overview,
            "numeric_summary": summary,
            "target_relationship": rel,
            "pairwise_corr": corr,
            "missingness": miss,
            "findings": data["findings"],
            "narrative": data["narrative"],
        }
        path = run_dir / "eda_report.json"
        path.write_text(json.dumps(artifact, indent=2))

        result.artifacts.append(ArtifactRef(path=str(path), agent=self.name, name=path.name))
        result.outputs = {
            "n_features_summarised": len(summary),
            "n_candidates_ranked": len(rel),
            "top_feature": rel[0]["feature"] if rel else None,
            "top_spearman": rel[0]["spearman"] if rel else None,
        }
        result.reasoning = data["narrative"]
        result.confidence = 0.9
