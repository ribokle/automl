"""Feature-engineering agent.

Builds the PPG × week feature frame downstream modelling will consume, writes
it as a parquet artefact, and surfaces a one-line summary. LLM narrates the
intent of the engineered set.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from core.agents.base import Agent
from core.data.charts import feature_histograms
from core.features.eda import ppg_week_aggregate
from core.features.engineering import ENGINEERED_COLUMNS, TARGET, build_features
from core.orchestrator.state import AgentResult, ArtifactRef, RunState


SYSTEM_PROMPT = """You are a pricing feature engineer. Given the list of engineered
columns and brief stats, return STRICT JSON of the form
{"narrative": "<=320 chars on what the feature set captures"}. JSON only."""


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError):
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)


class FeatureEngineeringAgent(Agent):
    name = "feature_engineering"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        duckdb_path = Path(run.duckdb_path)
        run_dir = Path(run.run_dir)

        panel = await asyncio.to_thread(ppg_week_aggregate, duckdb_path)
        await self.emit(run, "tool_called", {"tool": "ppg_week_aggregate", "rows": int(len(panel))})

        feats = await asyncio.to_thread(build_features, panel)
        await self.emit(
            run,
            "tool_called",
            {"tool": "build_features", "rows": int(len(feats)), "columns": len(ENGINEERED_COLUMNS)},
        )

        features_path = run_dir / "features.parquet"
        await asyncio.to_thread(_to_parquet, feats, features_path)
        if not features_path.exists():
            features_path = features_path.with_suffix(".csv")

        summary = {
            "target": TARGET,
            "rows": int(len(feats)),
            "columns": ENGINEERED_COLUMNS,
            "ppg_ids": sorted(feats["ppg_id"].unique().tolist()),
            "week_min": str(feats["week_start"].min()),
            "week_max": str(feats["week_start"].max()),
        }

        try:
            llm_resp = await asyncio.to_thread(
                self.call_llm,
                result,
                system=SYSTEM_PROMPT,
                user=json.dumps(summary),
                max_tokens=240,
            )
            data = json.loads(llm_resp.text or "{}")
            narrative = data.get("narrative", "")
        except Exception:  # noqa: BLE001
            narrative = ""
        if not narrative:
            narrative = (
                f"Engineered {len(ENGINEERED_COLUMNS)} features at the PPG×week grain: "
                f"log_price + log_units target, promo shares, lag1/lag4 price, lag1 units, "
                f"competitor + distribution scaled by log, and seasonal week-of-year sin/cos."
            )

        summary_path = run_dir / "feature_engineering.json"
        summary_path.write_text(json.dumps({**summary, "narrative": narrative}, indent=2))

        hist_cols = [c for c in ENGINEERED_COLUMNS if c in feats.columns and c not in ("week_start", "ppg_id")]
        hist_blob = feature_histograms(feats, hist_cols)
        hist_path = run_dir / "feature_histograms.json"
        hist_path.write_text(json.dumps(hist_blob, indent=2))

        result.artifacts.append(ArtifactRef(path=str(features_path), agent=self.name, name=features_path.name))
        result.artifacts.append(ArtifactRef(path=str(summary_path), agent=self.name, name=summary_path.name))
        result.artifacts.append(ArtifactRef(path=str(hist_path), agent=self.name, name=hist_path.name))
        result.outputs = {
            "rows": int(len(feats)),
            "n_features": len(ENGINEERED_COLUMNS),
            "n_ppgs": int(feats["ppg_id"].nunique()),
            "format": features_path.suffix.lstrip("."),
        }
        result.reasoning = narrative
        result.confidence = 0.9
