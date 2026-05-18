"""Feature-refine agent.

Loads the engineered feature frame, runs VIF + correlation pruning with
`log_price` protected, and writes the kept / dropped manifest plus final
diagnostics. Verification anchor for Phase 2 (max VIF < 10 and no |corr| > 0.95).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from core.agents.base import Agent
from core.data.charts import corr_refined
from core.features.engineering import ENGINEERED_COLUMNS, TARGET
from core.features.refine import refine_features
from core.orchestrator.state import AgentResult, ArtifactRef, RunState


SYSTEM_PROMPT = """You are a pricing feature analyst. Given the kept features, the
dropped features with their reason, and the final VIF + max |corr|, return STRICT
JSON of the form {"narrative": "<=320 chars on what the refined set captures and
why"}. JSON only."""


PROTECTED: list[str] = ["log_price"]


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


class FeatureRefineAgent(Agent):
    name = "feature_refine"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        run_dir = Path(run.run_dir)

        feats = await asyncio.to_thread(_load_features, run_dir)
        await self.emit(run, "tool_called", {"tool": "load_features", "rows": int(len(feats))})

        candidates = [c for c in ENGINEERED_COLUMNS if c != TARGET and c in feats.columns]
        refined = await asyncio.to_thread(
            refine_features,
            feats,
            candidates,
            10.0,
            0.95,
            PROTECTED,
        )
        await self.emit(
            run,
            "tool_called",
            {"tool": "refine_features", "kept": len(refined["kept"]), "dropped": len(refined["dropped"])},
        )

        try:
            llm_resp = await asyncio.to_thread(
                self.call_llm,
                result,
                system=SYSTEM_PROMPT,
                user=json.dumps({**refined, "target": TARGET, "protected": PROTECTED}),
                max_tokens=240,
            )
            data = json.loads(llm_resp.text or "{}")
            narrative = data.get("narrative", "")
        except Exception:  # noqa: BLE001
            narrative = ""
        if not narrative:
            narrative = (
                f"Kept {len(refined['kept'])} features after dropping "
                f"{len(refined['dropped'])} collinear / high-VIF columns; log_price retained "
                f"as the primary elasticity variable. Max VIF {refined['max_vif']:.2f}, "
                f"max |corr| {refined['max_abs_corr']:.2f}."
            )

        passes = refined["max_vif"] < 10.0 and refined["max_abs_corr"] <= 0.95

        artifact = {
            **refined,
            "target": TARGET,
            "protected": PROTECTED,
            "vif_threshold": 10.0,
            "corr_threshold": 0.95,
            "passes_thresholds": passes,
            "narrative": narrative,
        }
        path = run_dir / "feature_refine.json"
        path.write_text(json.dumps(artifact, indent=2))
        result.artifacts.append(ArtifactRef(path=str(path), agent=self.name, name=path.name))

        corr_chart = corr_refined(feats, refined["kept"])
        corr_path = run_dir / "corr_refined.json"
        corr_path.write_text(json.dumps(corr_chart, indent=2))
        result.artifacts.append(
            ArtifactRef(path=str(corr_path), agent=self.name, name=corr_path.name)
        )
        result.outputs = {
            "n_kept": len(refined["kept"]),
            "n_dropped": len(refined["dropped"]),
            "max_vif": round(refined["max_vif"], 2),
            "max_abs_corr": round(refined["max_abs_corr"], 2),
            "passes_thresholds": passes,
        }
        result.reasoning = narrative
        result.confidence = 0.9 if passes else 0.5
