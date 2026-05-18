"""EDA helpers that read the panel mart and surface relationship signals.

These are pure functions over a DuckDB connection or a DataFrame so the EDA
agent can compose them without owning IO. Results are JSON-serialisable so the
agent can drop them straight into a run artefact.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


def panel_overview(duckdb_path: Path, table: str = "main.panel") -> dict[str, Any]:
    con = duckdb.connect(str(duckdb_path))
    try:
        agg = con.execute(
            f"""
            SELECT
              COUNT(*) AS rows,
              COUNT(DISTINCT sku) AS n_skus,
              COUNT(DISTINCT store_id) AS n_stores,
              COUNT(DISTINCT ppg_id) AS n_ppgs,
              MIN(week_start)::VARCHAR AS week_min,
              MAX(week_start)::VARCHAR AS week_max,
              SUM(CASE WHEN units IS NULL THEN 1 ELSE 0 END) AS null_units,
              SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) AS null_price
            FROM {table}
            """
        ).fetchone()
        cols = ["rows", "n_skus", "n_stores", "n_ppgs", "week_min", "week_max", "null_units", "null_price"]
        return {k: v for k, v in zip(cols, agg)}
    finally:
        con.close()


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        out.append(
            {
                "column": c,
                "n": int(s.size),
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }
        )
    return out


def pairwise_corr(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    """Pearson correlation matrix as a nested dict {col -> {col -> r}}."""
    sub = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        return {}
    corr = sub.corr()
    return {a: {b: float(corr.loc[a, b]) for b in columns} for a in columns}


def target_relationship(
    df: pd.DataFrame,
    target: str,
    candidates: list[str],
) -> list[dict[str, Any]]:
    """Spearman ρ between each candidate and the target (robust to scale + outliers)."""
    out: list[dict[str, Any]] = []
    y = pd.to_numeric(df[target], errors="coerce")
    for c in candidates:
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 30:
            continue
        rho = float(x[mask].corr(y[mask], method="spearman"))
        out.append({"feature": c, "spearman": rho, "n": int(mask.sum())})
    out.sort(key=lambda r: abs(r["spearman"]), reverse=True)
    return out


def missingness(df: pd.DataFrame) -> dict[str, float]:
    return {c: float(df[c].isna().mean()) for c in df.columns}


def ppg_week_aggregate(duckdb_path: Path, table: str = "main.panel") -> pd.DataFrame:
    """PPG × week roll-up used by the EDA / modelling stages as the canonical
    analytics grain (collapses the store dimension)."""
    con = duckdb.connect(str(duckdb_path))
    try:
        return con.execute(
            f"""
            SELECT
              ppg_id,
              week_start,
              SUM(units) AS units,
              SUM(units * price) / NULLIF(SUM(units), 0) AS price,
              SUM(units * base_price) / NULLIF(SUM(units), 0) AS base_price,
              AVG(discount_depth) AS discount_depth,
              AVG(tpr_flag::DOUBLE) AS tpr_share,
              AVG(display_flag::DOUBLE) AS display_share,
              AVG(feature_flag::DOUBLE) AS feature_share,
              AVG(distribution_acv) AS distribution_acv,
              AVG(competitor_price) AS competitor_price,
              MAX(CASE WHEN holiday IS NULL OR holiday = '' THEN 0 ELSE 1 END) AS is_holiday_week
            FROM {table}
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).df()
    finally:
        con.close()
