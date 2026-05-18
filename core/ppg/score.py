"""PPG eligibility scoring.

A PPG is modelling-eligible when the panel has enough volume, enough weekly
coverage, enough price variation to identify an elasticity, and enough promo
variation to identify lift coefficients. The score is a weighted blend
clipped to [0, 1]; the agent rules-out any PPG below a configurable threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


@dataclass
class ScoreWeights:
    volume: float = 0.25
    coverage: float = 0.25
    price_variation: float = 0.30
    promo_variation: float = 0.20


def score_ppgs(
    duckdb_path: Path,
    ppg_assignments: pd.DataFrame,
    table: str = "panel",
    weights: ScoreWeights | None = None,
    threshold: float = 0.6,
) -> pd.DataFrame:
    """Score each PPG and tag it eligible / not.

    Inputs:
      ppg_assignments: per-SKU dataframe with at least (sku, ppg_id).
    Output columns: ppg_id, n_skus, total_units, coverage, price_cv,
    promo_weeks_pct, score, eligible, reasoning.
    """
    if weights is None:
        weights = ScoreWeights()

    con = duckdb.connect(str(duckdb_path))
    try:
        panel = con.execute(
            f"select sku, week_start, store_id, units, price, tpr_flag from main.{table}"
        ).df()
    finally:
        con.close()

    panel = panel.merge(ppg_assignments[["sku", "ppg_id"]], on="sku", how="left")
    rows: list[dict[str, Any]] = []
    total_weeks = panel["week_start"].nunique()

    for ppg_id, grp in panel.groupby("ppg_id"):
        n_skus = grp["sku"].nunique()
        total_units = float(grp["units"].sum())
        weeks_covered = grp["week_start"].nunique()
        coverage = weeks_covered / total_weeks if total_weeks else 0.0
        # Price variation: coefficient of variation across all (sku, week) cells.
        price = grp["price"].dropna()
        price_cv = float(price.std() / price.mean()) if len(price) and price.mean() > 0 else 0.0
        promo_weeks_pct = float(grp["tpr_flag"].astype(float).mean())

        s_volume = _saturating(total_units, 50_000)
        s_coverage = coverage
        s_price = _saturating(price_cv, 0.30)
        s_promo = _band(promo_weeks_pct, low=0.05, high=0.50)

        score = (
            weights.volume * s_volume
            + weights.coverage * s_coverage
            + weights.price_variation * s_price
            + weights.promo_variation * s_promo
        )
        score = float(np.clip(score, 0.0, 1.0))

        reasons: list[str] = []
        if s_volume < 0.4:
            reasons.append(f"low volume ({int(total_units)} units)")
        if s_coverage < 0.6:
            reasons.append(f"sparse weekly coverage ({coverage:.0%})")
        if s_price < 0.4:
            reasons.append(f"insufficient price variation (cv={price_cv:.2f})")
        if s_promo < 0.4:
            reasons.append(f"promo activity outside identifiable band ({promo_weeks_pct:.0%})")
        if not reasons:
            reasons.append("volume, coverage, and price/promo variation all sufficient")

        rows.append(
            {
                "ppg_id": ppg_id,
                "n_skus": int(n_skus),
                "total_units": total_units,
                "coverage": round(coverage, 3),
                "price_cv": round(price_cv, 3),
                "promo_weeks_pct": round(promo_weeks_pct, 3),
                "score": round(score, 3),
                "eligible": bool(score >= threshold),
                "reasoning": "; ".join(reasons),
            }
        )

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return out


def _saturating(value: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return float(np.clip(value / target, 0.0, 1.0))


def _band(value: float, low: float, high: float) -> float:
    """1.0 inside [low, high], decaying linearly to 0 outside the band."""
    if low <= value <= high:
        return 1.0
    if value < low:
        return float(max(0.0, value / low))
    span = max(1.0 - high, 1e-6)
    return float(max(0.0, 1.0 - (value - high) / span))
