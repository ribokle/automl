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
from core.config import get_settings
from core.data.charts import feature_histograms
from core.features.competitor import compute_competitor_proxy
from core.features.eda import aggregate_features
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

        # Resolve the modelling grain: per-run option override > global setting.
        # Default is the historical ppg_week so existing tests stay green.
        explicit_grain = run.options.get("modelling_grain")
        grain = (
            getattr(explicit_grain, "value", str(explicit_grain))
            if explicit_grain
            else get_settings().modelling_grain.value
        )

        panel = await asyncio.to_thread(
            aggregate_features, duckdb_path, grain=grain
        )
        await self.emit(
            run,
            "tool_called",
            {"tool": "aggregate_features", "grain": grain, "rows": int(len(panel))},
        )

        # If the loader didn't ship a competitor series, fall back to the
        # within-PPG-week mean price of the other SKUs (Hoch et al. use a
        # similar substitution; see core/features/competitor.py). At the
        # store-grain we use the same chain-level proxy — it's still the
        # cleanest signal of "what did the rest of the PPG cost this week".
        comp_coverage = float(panel["competitor_price"].notna().mean()) if len(panel) else 0.0
        if comp_coverage < 0.5:
            self.log.warning(
                "feature_engineering: competitor_price coverage %.0f%% — filling from within-PPG-week proxy",
                comp_coverage * 100,
            )
            proxy = await asyncio.to_thread(compute_competitor_proxy, duckdb_path, grain="ppg_week")
            if len(proxy):
                proxy_idx = proxy.set_index(["ppg_id", "week_start"])["competitor_price"]
                key_idx = pd.MultiIndex.from_arrays(
                    [panel["ppg_id"], pd.to_datetime(panel["week_start"])]
                )
                mapped = pd.Series(proxy_idx.reindex(key_idx).to_numpy(), index=panel.index)
                panel["competitor_price"] = panel["competitor_price"].fillna(mapped)

        feats = await asyncio.to_thread(build_features, panel)
        # At the historical chain grain we drop grain_unit so feature_refine /
        # modeling / validation see the same schema they always have.
        # At a store-grain we keep it so modeling can loop over cells.
        if grain == "ppg_week" and "grain_unit" in feats.columns:
            feats = feats.drop(columns=["grain_unit"])
        has_grain_unit = "grain_unit" in feats.columns
        await self.emit(
            run,
            "tool_called",
            {"tool": "build_features", "rows": int(len(feats)), "columns": len(ENGINEERED_COLUMNS)},
        )

        # Flag columns that came out of build_features as constants — these
        # will be dropped by feature_refine (correlation undefined on a
        # constant), and seeing them here means the upstream loader didn't
        # carry a real signal (display_flag / feature_flag / acv on
        # Dominick's, for example). Surface them so the dashboard can
        # render a "constant-by-design" badge instead of letting the drop
        # happen silently.
        constant_engineered: list[str] = []
        for col in ENGINEERED_COLUMNS:
            if col in feats.columns and feats[col].dropna().nunique() <= 1:
                constant_engineered.append(col)
        if constant_engineered:
            self.log.warning(
                "feature_engineering: %d engineered columns are constant on this panel: %s",
                len(constant_engineered),
                ", ".join(constant_engineered),
            )

        features_path = run_dir / "features.parquet"
        await asyncio.to_thread(_to_parquet, feats, features_path)
        if not features_path.exists():
            features_path = features_path.with_suffix(".csv")

        summary = {
            "target": TARGET,
            "grain": grain,
            "rows": int(len(feats)),
            "columns": ENGINEERED_COLUMNS,
            "constant_columns": constant_engineered,
            "ppg_ids": sorted(feats["ppg_id"].unique().tolist()),
            "n_grain_units": int(feats["grain_unit"].nunique()) if has_grain_unit else 1,
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
            "n_grain_units": summary["n_grain_units"],
            "grain": grain,
            "format": features_path.suffix.lstrip("."),
        }
        result.reasoning = narrative
        result.confidence = 0.9
