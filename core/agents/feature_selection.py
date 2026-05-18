"""Feature-selection agent.

Reads the panel mart, decides which raw columns are candidate features for
the modelling stage, and emits a per-candidate role (target / numeric / flag /
identifier / temporal). The LLM contributes a short narrative on why a few
borderline columns were included or excluded; everything else is deterministic
so the agent runs without an API key.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import duckdb

from core.agents.base import Agent
from core.orchestrator.state import AgentResult, ArtifactRef, RunState


SYSTEM_PROMPT = """You are a pricing-analytics feature analyst. Given the panel
columns and their roles, return STRICT JSON with a short rationale (<=240 chars)
explaining the chosen feature set: {"rationale": "..."}. JSON only."""


_TARGETS = {"units"}
_IDENTIFIERS = {"sku", "store_id", "region", "category", "brand", "pack_size", "segment", "ppg_id"}
_TEMPORAL = {"week_start", "holiday"}
_FLAGS = {"tpr_flag", "display_flag", "feature_flag"}


def _classify(column: str, dtype: str) -> str:
    if column in _TARGETS:
        return "target"
    if column in _IDENTIFIERS:
        return "identifier"
    if column in _TEMPORAL:
        return "temporal"
    if column in _FLAGS:
        return "flag"
    if "INT" in dtype or "DOUBLE" in dtype or "DECIMAL" in dtype or "FLOAT" in dtype:
        return "numeric"
    return "categorical"


def _describe_panel(duckdb_path: Path) -> list[dict[str, str]]:
    con = duckdb.connect(str(duckdb_path))
    try:
        cols = con.execute("DESCRIBE main.panel").df()
    finally:
        con.close()
    out: list[dict[str, str]] = []
    for _, row in cols.iterrows():
        out.append(
            {
                "column": str(row["column_name"]),
                "dtype": str(row["column_type"]),
                "role": _classify(str(row["column_name"]), str(row["column_type"])),
            }
        )
    return out


class FeatureSelectionAgent(Agent):
    name = "feature_selection"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        duckdb_path = Path(run.duckdb_path)
        run_dir = Path(run.run_dir)

        columns = await asyncio.to_thread(_describe_panel, duckdb_path)
        await self.emit(run, "tool_called", {"tool": "describe_panel", "columns": len(columns)})

        roles: dict[str, list[str]] = {}
        for c in columns:
            roles.setdefault(c["role"], []).append(c["column"])

        candidates = roles.get("numeric", []) + roles.get("flag", [])

        user_prompt = json.dumps(
            {
                "columns": columns,
                "candidates": candidates,
                "target": "units",
            }
        )
        try:
            llm_resp = await asyncio.to_thread(
                self.call_llm,
                result,
                system=SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=240,
            )
            data = json.loads(llm_resp.text or "{}")
            rationale = data.get("rationale", "")
        except Exception:  # noqa: BLE001
            rationale = ""
        if not rationale:
            rationale = (
                f"Selected {len(candidates)} numeric / flag columns as modelling candidates; "
                f"held {len(roles.get('identifier', []))} identifiers and {len(roles.get('temporal', []))} temporal columns aside."
            )

        artifact = {
            "target": "units",
            "candidates": candidates,
            "roles": roles,
            "columns": columns,
            "rationale": rationale,
        }
        path = run_dir / "feature_candidates.json"
        path.write_text(json.dumps(artifact, indent=2))

        result.artifacts.append(ArtifactRef(path=str(path), agent=self.name, name=path.name))
        result.outputs = {
            "n_candidates": len(candidates),
            "n_identifiers": len(roles.get("identifier", [])),
            "n_temporal": len(roles.get("temporal", [])),
            "target": "units",
        }
        result.reasoning = rationale
        result.confidence = 0.9
