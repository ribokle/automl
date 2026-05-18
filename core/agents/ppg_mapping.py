"""PPG-mapping agent.

1. Aggregates per-SKU features from the canonical panel.
2. Runs the deterministic PPG clustering algorithm.
3. Asks the LLM to review the result and emit a one-paragraph rationale per
   PPG. The algorithm is the source of truth for the mapping; the LLM only
   produces explanatory text and flags any group it thinks looks suspicious.

Outputs:
- ppg_mapping.json         (the full per-SKU assignment + per-PPG rationale)
- ppg_mapping_table.json   (a flat list for the UI)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.agents.base import Agent
from core.data.charts import (
    ppg_price_box,
    ppg_scatter_behaviour,
    ppg_scatter_facet,
    ppg_scatter_tier,
)
from core.orchestrator.state import AgentResult, ArtifactRef, RunState
from core.ppg.cluster import ClusterParams, cluster_ppgs
from core.ppg.features import aggregate_sku_features


SYSTEM_PROMPT = """You are the Price-Pack Group (PPG) mapping reviewer for a CPG analytics platform.
You receive a deterministic PPG mapping produced by a clustering algorithm grouped by
(brand, category) with optional within-group price-pack subclustering. For each PPG you
must return one short, business-grade rationale (<=200 chars) explaining why the SKUs
in that group are substitutable. Flag any PPG whose member SKUs look heterogeneous
(very different price tiers or pack sizes).

Return STRICT JSON: {"per_ppg": [{"ppg_id": "...", "rationale": "...", "flag": bool}, ...]}.
Do not change ppg_id assignments. Reply with JSON only."""


def _dry_run_rationales(per_ppg_summary: list[dict]) -> dict:
    return {
        "per_ppg": [
            {
                "ppg_id": p["ppg_id"],
                "rationale": (
                    f"{p['brand']} {p['category']} ({p['n_skus']} SKUs, "
                    f"price ${p['min_price']:.2f}-${p['max_price']:.2f}); "
                    f"shared brand and category make these direct substitutes."
                ),
                "flag": (p["max_price"] / max(p["min_price"], 1e-6)) > 2.0,
            }
            for p in per_ppg_summary
        ]
    }


class PPGMappingAgent(Agent):
    name = "ppg_mapping"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        duckdb_path = Path(run.duckdb_path)
        run_dir = Path(run.run_dir)

        sku_features = await asyncio.to_thread(aggregate_sku_features, duckdb_path)
        await self.emit(run, "tool_called", {"tool": "aggregate_sku_features", "n_skus": int(len(sku_features))})

        assignments = await asyncio.to_thread(cluster_ppgs, sku_features, ClusterParams())
        await self.emit(
            run,
            "tool_called",
            {"tool": "cluster_ppgs", "n_skus": int(len(assignments)), "n_ppgs": int(assignments["ppg_id"].nunique())},
        )

        # Per-PPG summary for LLM rationale.
        per_ppg_summary: list[dict] = []
        for ppg_id, grp in assignments.groupby("ppg_id"):
            per_ppg_summary.append(
                {
                    "ppg_id": ppg_id,
                    "brand": grp["brand"].iloc[0],
                    "category": grp["category"].iloc[0],
                    "n_skus": int(len(grp)),
                    "pack_sizes": sorted(grp["pack_size"].unique().tolist()),
                    "min_price": float(grp["median_price"].min()),
                    "max_price": float(grp["median_price"].max()),
                    "mean_confidence": round(float(grp["confidence"].mean()), 3),
                }
            )

        llm_resp = await asyncio.to_thread(
            self.call_llm,
            result,
            system=SYSTEM_PROMPT,
            user=json.dumps({"per_ppg": per_ppg_summary}),
            max_tokens=1200,
        )
        rationales: dict
        try:
            rationales = json.loads(llm_resp.text) if not llm_resp.raw.get("dry_run") else _dry_run_rationales(per_ppg_summary)
        except (json.JSONDecodeError, ValueError):
            rationales = _dry_run_rationales(per_ppg_summary)
        rationale_lookup = {item["ppg_id"]: item for item in rationales.get("per_ppg", [])}

        # Build UI-friendly table.
        table_rows: list[dict] = []
        for ppg_id, grp in assignments.groupby("ppg_id"):
            r = rationale_lookup.get(ppg_id, {})
            ppg_rationale = r.get("rationale", "")
            flag = bool(r.get("flag", False))
            for _, row in grp.iterrows():
                table_rows.append(
                    {
                        "ppg_id": ppg_id,
                        "sku": row["sku"],
                        "brand": row["brand"],
                        "category": row["category"],
                        "pack_size": row["pack_size"],
                        "median_price": float(row["median_price"]),
                        "confidence": float(row["confidence"]),
                        "rationale": ppg_rationale,
                        "flagged": flag,
                    }
                )

        mapping_blob = {
            "assignments": assignments.to_dict(orient="records"),
            "per_ppg": per_ppg_summary,
            "rationales": rationale_lookup,
            "table": table_rows,
        }
        mapping_path = run_dir / "ppg_mapping.json"
        mapping_path.write_text(json.dumps(mapping_blob, indent=2, default=str))
        table_path = run_dir / "ppg_mapping_table.json"
        table_path.write_text(json.dumps(table_rows, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(path=str(mapping_path), mime="application/json", agent=self.name, name="ppg_mapping.json")
        )
        result.artifacts.append(
            ArtifactRef(path=str(table_path), mime="application/json", agent=self.name, name="ppg_mapping_table.json")
        )

        for chart_name, blob in (
            ("ppg_scatter_tier.json", ppg_scatter_tier(assignments)),
            (
                "ppg_scatter_behaviour.json",
                await asyncio.to_thread(ppg_scatter_behaviour, assignments, duckdb_path),
            ),
            ("ppg_scatter_facet.json", ppg_scatter_facet(assignments)),
            ("ppg_price_box.json", await asyncio.to_thread(ppg_price_box, assignments, duckdb_path)),
        ):
            chart_path = run_dir / chart_name
            chart_path.write_text(json.dumps(blob, indent=2, default=str))
            result.artifacts.append(
                ArtifactRef(
                    path=str(chart_path),
                    mime="application/json",
                    agent=self.name,
                    name=chart_name,
                )
            )

        n_ppgs = int(assignments["ppg_id"].nunique())
        flagged = sum(1 for r in rationale_lookup.values() if r.get("flag"))
        result.outputs = {
            "n_skus": int(len(assignments)),
            "n_ppgs": n_ppgs,
            "n_flagged": flagged,
            "mean_confidence": round(float(assignments["confidence"].mean()), 3),
        }
        result.reasoning = (
            f"Clustered {len(assignments)} SKUs into {n_ppgs} PPGs via "
            f"(brand, category) hard partition + within-group price-pack subclustering. "
            f"Mean per-SKU confidence {result.outputs['mean_confidence']:.2f}. "
            f"{flagged} groups flagged for review."
        )
        result.confidence = float(assignments["confidence"].mean())
