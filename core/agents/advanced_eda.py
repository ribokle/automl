"""Advanced EDA agent.

Runs after the lightweight ``eda`` agent. Produces a suite of artefacts that
surface time-series structure, structural anomalies, change points, pre-model
promo lift sketches, cross-PPG correlation within category, Pareto / ABC
classification, price ladders, a promo calendar, holiday lift, and a
cardinality report.

Compute caps (set via ``run.options["advanced_eda"]``):
- ``max_series``: top-K PPGs by revenue for STL / ACF / isolation forest
  (default 50)
- ``corr_cap``: per-category cap for cross-PPG correlation (default 20)

The LLM narrates the top findings; every block has a deterministic dry-run
fallback so the pipeline runs without an API key.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from core.agents.base import Agent
from core.data.advanced_charts import (
    acf_chart,
    anomaly_timeline,
    cross_ppg_heatmap,
    promo_calendar_chart,
    stl_chart,
)
from core.features.advanced_eda import (
    cardinality_report,
    cross_ppg_correlation,
    detect_change_points,
    detect_forward_buy,
    detect_pantry_loading,
    detect_stockouts,
    distribution_report,
    holiday_lift_table,
    isolation_forest_anomalies,
    pareto_abc,
    price_ladder,
    promo_calendar,
    promo_lift_sketches,
    store_variability,
    time_series_diagnostics,
    top_k_ppgs_by_revenue,
)
from core.features.eda import ppg_week_aggregate
from core.orchestrator.state import AgentResult, ArtifactRef, RunState

SYSTEM_PROMPT = """You are a CPG pricing EDA analyst. Given the advanced-EDA
summary (stationarity verdicts, anomalies, change points, top lifters, ABC
class, and category correlation hot spots), return STRICT JSON of the form
{"findings": ["bullet 1 (<=160 chars)", "bullet 2", ...], "narrative": "<=380 chars"}.
Focus on what the modelling agent needs to know upfront: which PPGs have
structural breaks, which lift sketches look anomalous, which categories
show suspicious co-movement. JSON only."""


_NUMERIC_COLS = (
    "units", "price", "base_price", "discount_depth", "tpr_share",
    "display_share", "feature_share", "distribution_acv", "competitor_price",
)


def _read_raw_panel(duckdb_path: Path) -> pd.DataFrame:
    con = duckdb.connect(str(duckdb_path))
    try:
        df = con.execute(
            """
            SELECT
              sku, week_start, store_id, region,
              ppg_id, category, brand, pack_size, segment,
              units, price, base_price, discount_depth,
              tpr_flag, display_flag, feature_flag,
              distribution_acv, competitor_price, holiday
            FROM main.panel
            """
        ).df()
    finally:
        con.close()
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def _ppg_category_lookup(raw: pd.DataFrame) -> dict[str, str]:
    if "ppg_id" not in raw.columns or "category" not in raw.columns:
        return {}
    return (
        raw[["ppg_id", "category"]]
        .dropna()
        .drop_duplicates()
        .set_index("ppg_id")["category"]
        .to_dict()
    )


def _dry_run_summary(
    stationarity_pass_rate: float,
    anomaly_count: int,
    change_point_count: int,
    top_lifters: list[dict[str, Any]],
    abc_a: int,
) -> dict[str, Any]:
    bullets = [
        f"Stationarity: {stationarity_pass_rate * 100:.0f}% of top-K PPGs reject ADF — modelling can use levels for those.",
        f"Anomalies flagged: {anomaly_count} rows across stockout / pantry / forward-buy / isolation-forest.",
        f"Change points: {change_point_count} baseline-price shifts detected — flag these in the modelling agent.",
        f"Pareto: {abc_a} PPGs carry the top 80% of revenue (ABC=A).",
    ]
    if top_lifters:
        t = top_lifters[0]
        bullets.append(f"Highest TPR lift sketch: {t['ppg_id']} (×{t['lift']:.2f}); rerun controlled in modelling.")
    return {
        "findings": bullets,
        "narrative": "Advanced EDA produced 10 artefacts; modelling agent should consult change points + isolation-forest flags first.",
    }


class AdvancedEDAAgent(Agent):
    name = "advanced_eda"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        duckdb_path = Path(run.duckdb_path)
        run_dir = Path(run.run_dir)
        opts = (run.options or {}).get("advanced_eda", {}) if isinstance(run.options, dict) else {}
        max_series = int(opts.get("max_series", 50))
        corr_cap = int(opts.get("corr_cap", 20))

        panel = await asyncio.to_thread(ppg_week_aggregate, duckdb_path)
        await self.emit(run, "tool_called", {"tool": "ppg_week_aggregate", "rows": int(len(panel))})

        raw = await asyncio.to_thread(_read_raw_panel, duckdb_path)
        await self.emit(run, "tool_called", {"tool": "read_raw_panel", "rows": int(len(raw))})

        top_ppgs = top_k_ppgs_by_revenue(panel, max_series)

        diagnostics = await asyncio.to_thread(time_series_diagnostics, panel, top_ppgs)
        await self.emit(run, "tool_called", {"tool": "time_series_diagnostics", "n_ppgs": len(diagnostics)})

        variability = await asyncio.to_thread(store_variability, raw)

        stockouts = await asyncio.to_thread(detect_stockouts, panel)
        pantry = await asyncio.to_thread(detect_pantry_loading, panel)
        forward = await asyncio.to_thread(detect_forward_buy, panel)
        iso = await asyncio.to_thread(isolation_forest_anomalies, panel, top_ppgs)
        anomalies = [*stockouts, *pantry, *forward, *iso]
        await self.emit(run, "tool_called", {"tool": "anomaly_detection", "n_anomalies": len(anomalies)})

        change_points = await asyncio.to_thread(detect_change_points, panel, top_ppgs)
        await self.emit(run, "tool_called", {"tool": "change_points", "n_points": len(change_points)})

        dist_cols = [c for c in _NUMERIC_COLS if c in panel.columns]
        dist = await asyncio.to_thread(distribution_report, panel, dist_cols)

        promo_lift = await asyncio.to_thread(promo_lift_sketches, panel)
        await self.emit(run, "tool_called", {"tool": "promo_lift_sketches", "n_ppgs": len(promo_lift)})

        cat_lookup = _ppg_category_lookup(raw)
        cross = await asyncio.to_thread(cross_ppg_correlation, panel, cat_lookup, corr_cap)

        pareto = await asyncio.to_thread(pareto_abc, raw, panel)
        ladder = await asyncio.to_thread(price_ladder, panel)
        calendar = await asyncio.to_thread(promo_calendar, panel)
        holiday_lift = await asyncio.to_thread(holiday_lift_table, raw)
        cardinality = await asyncio.to_thread(
            cardinality_report, raw, ["category", "brand", "pack_size", "segment", "region"]
        )

        artifacts: list[tuple[str, Any]] = [
            (
                "time_series_diagnostics.json",
                {
                    "max_series": max_series,
                    "n_ppgs_top_k": len(top_ppgs),
                    "diagnostics": diagnostics,
                    "store_variability": variability,
                },
            ),
            (
                "temporal_anomalies.json",
                {
                    "n_total": len(anomalies),
                    "by_type": {
                        t: sum(1 for a in anomalies if a["anomaly_type"] == t)
                        for t in ("stockout", "pantry_loading", "forward_buy", "isolation_forest")
                    },
                    "rows": anomalies,
                },
            ),
            ("change_points.json", {"n_total": len(change_points), "rows": change_points}),
            ("distribution_report.json", {"columns": dist}),
            ("promo_lift_sketches.json", {"per_ppg": promo_lift}),
            ("cross_ppg_correlation.json", cross),
            ("pareto_abc.json", pareto),
            ("price_ladder.json", {"per_ppg": ladder}),
            ("promo_calendar.json", calendar),
            ("holiday_lift.json", {"rows": holiday_lift}),
            ("cardinality_report.json", {"columns": cardinality}),
        ]

        weeks = sorted({str(w) for w in panel["week_start"].astype(str).unique()}) if not panel.empty else []
        chart_specs: list[tuple[str, Any]] = [
            (
                "advanced_eda_charts.json",
                {
                    "stl": [stl_chart(d) for d in diagnostics if d.get("stl", {}).get("available")],
                    "acf_pacf": [acf_chart(d) for d in diagnostics if d.get("acf_pacf", {}).get("available")],
                    "anomaly_timeline": anomaly_timeline(weeks, anomalies),
                    "cross_ppg": cross_ppg_heatmap(cross),
                    "promo_calendar": promo_calendar_chart(calendar),
                },
            ),
        ]

        stationarity_pass_rate = (
            sum(1 for d in diagnostics if d.get("stationarity", {}).get("verdict") == "stationary")
            / max(1, len(diagnostics))
        )
        top_lifters = sorted(
            [
                {"ppg_id": p["ppg_id"], "lift": p["lifts"].get("tpr", {}).get("lift", 0.0)}
                for p in promo_lift
                if p.get("lifts", {}).get("tpr", {}).get("lift", 0.0)
            ],
            key=lambda r: r["lift"],
            reverse=True,
        )
        abc_counts = {"A": 0, "B": 0, "C": 0}
        for row in pareto.get("ppg_abc", []):
            abc_counts[row["abc_class"]] = abc_counts.get(row["abc_class"], 0) + 1

        summary_for_llm = {
            "n_ppgs_total": int(panel["ppg_id"].nunique()) if not panel.empty else 0,
            "n_ppgs_top_k": len(top_ppgs),
            "stationarity_pass_rate": round(stationarity_pass_rate, 3),
            "n_anomalies": len(anomalies),
            "anomaly_breakdown": {
                t: sum(1 for a in anomalies if a["anomaly_type"] == t)
                for t in ("stockout", "pantry_loading", "forward_buy", "isolation_forest")
            },
            "n_change_points": len(change_points),
            "abc_counts": abc_counts,
            "top_lifters": top_lifters[:5],
            "categories_with_correlation": [c["category"] for c in cross.get("categories", [])],
        }

        try:
            llm_resp = await asyncio.to_thread(
                self.call_llm,
                result,
                system=SYSTEM_PROMPT,
                user=json.dumps(summary_for_llm),
                max_tokens=400,
            )
            data = json.loads(llm_resp.text or "{}")
            if not isinstance(data.get("findings"), list):
                raise ValueError("missing findings")
        except Exception:  # noqa: BLE001
            data = _dry_run_summary(
                stationarity_pass_rate, len(anomalies), len(change_points), top_lifters, abc_counts["A"]
            )

        report = {
            "summary": summary_for_llm,
            "findings": data["findings"],
            "narrative": data["narrative"],
            "compute_caps": {"max_series": max_series, "corr_cap": corr_cap},
            "artifact_index": [name for name, _ in artifacts],
        }
        report_path = run_dir / "advanced_eda_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(path=str(report_path), agent=self.name, name=report_path.name, mime="application/json")
        )

        for name, blob in [*artifacts, *chart_specs]:
            path = run_dir / name
            path.write_text(json.dumps(blob, indent=2, default=str))
            result.artifacts.append(
                ArtifactRef(path=str(path), agent=self.name, name=path.name, mime="application/json")
            )

        result.outputs = {
            "n_ppgs_top_k": len(top_ppgs),
            "n_anomalies": len(anomalies),
            "n_change_points": len(change_points),
            "stationarity_pass_rate": round(stationarity_pass_rate, 3),
            "abc_a_count": abc_counts["A"],
            "n_categories_correlated": len(cross.get("categories", [])),
        }
        result.reasoning = data["narrative"]
        result.confidence = 0.85
