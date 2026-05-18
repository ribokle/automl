"""Chart-ready data builders for the Phase 2a data-visibility layer.

Each function reads the canonical `main.panel` mart (and the upstream PPG /
selection artefacts where relevant) and returns a JSON-serialisable shape the
frontend can render directly with ECharts — no further reshaping in TS.

Builders are deliberately small + pure. Each agent calls only the ones it
owns and writes the result alongside its other artefacts.

Real-data graceful degradation: when an expected column is absent (e.g. an
uploaded CSV that doesn't carry `brand`), the builder returns
``{"missing_columns": [...], ...}`` instead of raising. The frontend renders
a placeholder using that field.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from core.data.ingestion_report import CheckResult, IngestionReport


def _connect(duckdb_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(duckdb_path))


def coverage_grid(duckdb_path: Path, table: str = "panel") -> dict[str, Any]:
    """Sparse SKU × week presence matrix.

    Returns ``{skus, weeks, present}`` where ``present[i][j]`` is 1 if any
    row exists for ``(skus[i], weeks[j])`` and 0 otherwise.
    """
    con = _connect(duckdb_path)
    try:
        df = con.execute(
            f"select sku, week_start, count(*) as n from main.{table} group by 1, 2"
        ).df()
    finally:
        con.close()
    skus = sorted(df["sku"].unique().tolist())
    weeks = sorted({w.isoformat() if hasattr(w, "isoformat") else str(w) for w in df["week_start"]})
    week_idx = {w: i for i, w in enumerate(weeks)}
    sku_idx = {s: i for i, s in enumerate(skus)}
    grid = [[0] * len(weeks) for _ in skus]
    for sku, w, _ in df.itertuples(index=False):
        wkey = w.isoformat() if hasattr(w, "isoformat") else str(w)
        grid[sku_idx[sku]][week_idx[wkey]] = 1
    return {
        "skus": skus,
        "weeks": weeks,
        "present": grid,
        "n_total_cells": len(skus) * len(weeks),
        "n_present_cells": sum(sum(r) for r in grid),
    }


def weekly_trend(duckdb_path: Path, table: str = "panel") -> dict[str, Any]:
    """Panel-wide weekly units / average price / promo share."""
    con = _connect(duckdb_path)
    try:
        df = con.execute(
            f"""
            select
              week_start,
              sum(units) as units,
              avg(price) as price,
              avg(case when tpr_flag = 1 then 1.0 else 0.0 end) as promo_share
            from main.{table}
            group by 1
            order by 1
            """
        ).df()
    finally:
        con.close()
    return {
        "weeks": [w.isoformat() if hasattr(w, "isoformat") else str(w) for w in df["week_start"]],
        "units": [int(u) for u in df["units"]],
        "price": [round(float(p), 4) for p in df["price"]],
        "promo_share": [round(float(p), 4) for p in df["promo_share"]],
    }


def quality_results(report: IngestionReport) -> dict[str, Any]:
    """Flatten the dbt + GE check list into a UI-shaped object.

    Output: ``{summary: {pass, warn, fail}, checks: [{source, name, status,
    severity, message, failing_rows}, ...]}``. Frontend renders the summary
    pills and a per-row list.
    """
    checks: list[dict[str, Any]] = []
    n_pass = n_warn = n_fail = 0
    for c in [*report.dbt, *report.ge]:
        ui_status = c.status
        if c.status == "fail" and c.severity.value == "warn":
            ui_status = "warn"
        if ui_status == "pass":
            n_pass += 1
        elif ui_status == "warn":
            n_warn += 1
        else:
            n_fail += 1
        checks.append(
            {
                "source": c.source,
                "name": c.name,
                "status": ui_status,
                "severity": c.severity.value,
                "message": c.message or "",
                "failing_rows": c.failing_rows,
            }
        )
    return {"summary": {"pass": n_pass, "warn": n_warn, "fail": n_fail}, "checks": checks}


_PPG_COLOURS = [
    "#34d399", "#60a5fa", "#f472b6", "#fbbf24",
    "#22d3ee", "#a78bfa", "#fb7185", "#4ade80",
    "#f59e0b", "#38bdf8", "#e879f9", "#facc15",
]


def _ppg_colour_map(ppg_ids: list[str]) -> dict[str, str]:
    return {ppg: _PPG_COLOURS[i % len(_PPG_COLOURS)] for i, ppg in enumerate(sorted(ppg_ids))}


def ppg_scatter_tier(assignments: pd.DataFrame) -> dict[str, Any]:
    """X = price tier (small/medium/large -> 0/1/2 ordinal jittered),
    Y = log of median price. Tautological by construction (the clusterer
    used these axes) but useful to confirm the partition.
    """
    missing = [c for c in ("ppg_id", "sku", "median_price", "pack_size") if c not in assignments.columns]
    if missing:
        return {"missing_columns": missing}
    tier_lookup = {"small": 0, "medium": 1, "large": 2}
    df = assignments.copy()
    df["tier"] = df["pack_size"].map(tier_lookup).fillna(1).astype(int)
    rng = np.random.default_rng(seed=0)
    df["x"] = df["tier"] + rng.uniform(-0.18, 0.18, size=len(df))
    df["y"] = np.log(df["median_price"].clip(lower=1e-6))
    colours = _ppg_colour_map(df["ppg_id"].unique().tolist())
    points = [
        {
            "ppg_id": r.ppg_id,
            "sku": r.sku,
            "pack": r.pack_size,
            "x": round(float(r.x), 4),
            "y": round(float(r.y), 4),
        }
        for r in df.itertuples(index=False)
    ]
    return {
        "view": "tier",
        "x_label": "pack-size tier",
        "y_label": "log median price",
        "colours": colours,
        "points": points,
    }


def ppg_scatter_behaviour(assignments: pd.DataFrame, duckdb_path: Path, table: str = "panel") -> dict[str, Any]:
    """Behaviour-based scatter: x = log mean weekly units per SKU,
    y = per-SKU correlation of log(units) vs log(price) (a coarse
    elasticity proxy). Tests whether members of the same PPG behave alike.
    """
    if "sku" not in assignments.columns or "ppg_id" not in assignments.columns:
        return {"missing_columns": ["sku", "ppg_id"]}
    con = _connect(duckdb_path)
    try:
        df = con.execute(
            f"select sku, week_start, units, price from main.{table}"
        ).df()
    finally:
        con.close()
    df = df.merge(assignments[["sku", "ppg_id"]], on="sku", how="inner")
    df = df[(df["units"] > 0) & (df["price"] > 0)].copy()
    df["log_units"] = np.log(df["units"])
    df["log_price"] = np.log(df["price"])
    weekly = df.groupby(["sku", "ppg_id", "week_start"], as_index=False).agg(
        log_units=("log_units", "mean"),
        log_price=("log_price", "mean"),
    )
    rows: list[dict[str, Any]] = []
    for (sku, ppg_id), grp in weekly.groupby(["sku", "ppg_id"]):
        if len(grp) < 3 or grp["log_price"].std() < 1e-6:
            continue
        corr = float(grp["log_units"].corr(grp["log_price"]))
        x = float(grp["log_units"].mean())
        rows.append({"sku": sku, "ppg_id": ppg_id, "x": round(x, 4), "y": round(corr, 4)})
    colours = _ppg_colour_map([r["ppg_id"] for r in rows])
    return {
        "view": "behaviour",
        "x_label": "log mean weekly units",
        "y_label": "per-SKU corr(log units, log price)",
        "colours": colours,
        "points": rows,
    }


def ppg_scatter_facet(assignments: pd.DataFrame) -> dict[str, Any]:
    """Faceted scatter: one panel per category, x = brand (ordinal),
    y = pack-size tier. Tests the brand/category/pack-size separation the
    clusterer was supposed to honour.
    """
    needed = ["ppg_id", "sku", "brand", "category", "pack_size"]
    missing = [c for c in needed if c not in assignments.columns]
    if missing:
        return {"missing_columns": missing}
    tier_lookup = {"small": 0, "medium": 1, "large": 2}
    rng = np.random.default_rng(seed=1)
    facets: list[dict[str, Any]] = []
    colours = _ppg_colour_map(assignments["ppg_id"].unique().tolist())
    for category, grp in assignments.groupby("category"):
        brands = sorted(grp["brand"].unique().tolist())
        brand_idx = {b: i for i, b in enumerate(brands)}
        points = []
        for r in grp.itertuples(index=False):
            tier = tier_lookup.get(r.pack_size, 1)
            jitter_x = rng.uniform(-0.18, 0.18)
            jitter_y = rng.uniform(-0.12, 0.12)
            points.append(
                {
                    "ppg_id": r.ppg_id,
                    "sku": r.sku,
                    "brand": r.brand,
                    "pack": r.pack_size,
                    "x": round(brand_idx[r.brand] + jitter_x, 4),
                    "y": round(tier + jitter_y, 4),
                }
            )
        facets.append({"category": category, "brands": brands, "points": points})
    return {
        "view": "facet",
        "x_label": "brand",
        "y_label": "pack-size tier",
        "colours": colours,
        "facets": facets,
    }


def ppg_price_box(assignments: pd.DataFrame, duckdb_path: Path, table: str = "panel") -> dict[str, Any]:
    """Per-PPG price distribution as ``{min, q1, median, q3, max}``."""
    if "sku" not in assignments.columns or "ppg_id" not in assignments.columns:
        return {"missing_columns": ["sku", "ppg_id"]}
    con = _connect(duckdb_path)
    try:
        df = con.execute(f"select sku, price from main.{table}").df()
    finally:
        con.close()
    df = df.merge(assignments[["sku", "ppg_id"]], on="sku", how="inner")
    boxes: list[dict[str, Any]] = []
    colours = _ppg_colour_map(df["ppg_id"].unique().tolist())
    for ppg_id, grp in df.groupby("ppg_id"):
        p = grp["price"].dropna()
        if p.empty:
            continue
        q1, med, q3 = (float(x) for x in p.quantile([0.25, 0.5, 0.75]))
        boxes.append(
            {
                "ppg_id": ppg_id,
                "min": round(float(p.min()), 3),
                "q1": round(q1, 3),
                "median": round(med, 3),
                "q3": round(q3, 3),
                "max": round(float(p.max()), 3),
                "n": int(len(p)),
            }
        )
    boxes.sort(key=lambda b: b["ppg_id"])
    return {"colours": colours, "boxes": boxes}


def eligibility_bars(scores: pd.DataFrame) -> dict[str, Any]:
    """Per-PPG eligibility breakdown by metric.

    Each PPG row has the four normalised contributions (volume / coverage /
    price-variation / promo-variation) so the frontend can render a stacked
    horizontal bar with the threshold line and an eligible / held-out badge.
    """
    needed = ["ppg_id", "score", "eligible", "total_units", "coverage", "price_cv", "promo_weeks_pct"]
    missing = [c for c in needed if c not in scores.columns]
    if missing:
        return {"missing_columns": missing}

    def _sat(v: float, t: float) -> float:
        return float(np.clip(v / t if t > 0 else 0.0, 0.0, 1.0))

    def _band(v: float, low: float, high: float) -> float:
        if low <= v <= high:
            return 1.0
        if v < low:
            return float(max(0.0, v / low))
        span = max(1.0 - high, 1e-6)
        return float(max(0.0, 1.0 - (v - high) / span))

    bars = []
    for r in scores.itertuples(index=False):
        bars.append(
            {
                "ppg_id": r.ppg_id,
                "score": round(float(r.score), 3),
                "eligible": bool(r.eligible),
                "contributions": {
                    "volume": round(_sat(float(r.total_units), 50_000) * 0.25, 4),
                    "coverage": round(float(r.coverage) * 0.25, 4),
                    "price_variation": round(_sat(float(r.price_cv), 0.30) * 0.30, 4),
                    "promo_variation": round(_band(float(r.promo_weeks_pct), 0.05, 0.50) * 0.20, 4),
                },
            }
        )
    return {"threshold": 0.60, "bars": bars}
