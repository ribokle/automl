"""Synthetic data regimes for cross-dataset model testing.

A single generator hides model-specific failure modes, so this module supplies
several *distinct* regimes — each shaped for the problem a family must handle —
with KNOWN ground truth. Own-price elasticities are drawn from the published
benchmark means (Hoch 1995 / Bijmolt 2005, see
``core/benchmarks/data/elasticity.json``) so the magnitudes are realistic, not
arbitrary.

Each generator returns ``(frame, truth)`` where ``frame`` uses the canonical
modelling schema (``ppg_id``, ``week_start``, ``log_price``, ``log_units`` + the
named controls) and ``truth`` is a dict describing what a correct model should
recover. Real public data (Dominick's) plugs in via
``core/data/loaders/dominicks.py``; these regimes keep CI hermetic and licence-clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.benchmarks.elasticity import lookup_category

CONTROLS = ["log_distribution_acv", "tpr_share"]


@dataclass
class Truth:
    elasticity: dict[str, float] = field(default_factory=dict)
    controls: list[str] = field(default_factory=list)
    instrument: str | None = None
    cross: dict[str, dict[str, float]] = field(default_factory=dict)


def _benchmark_elasticity(category: str, fallback: float) -> float:
    bench = lookup_category(category)
    return float(bench.mean) if bench is not None else fallback


def _weeks(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2019-01-06", periods=n, freq="W")


def clean_panel(
    categories: tuple[str, ...] = ("beer", "toothpaste", "detergent"),
    n_weeks: int = 150,
    seed: int = 0,
) -> tuple[pd.DataFrame, Truth]:
    """Well-behaved log-log panel: one PPG per category, benchmark elasticities,
    two informative controls. The baseline regression regime."""
    rng = np.random.default_rng(seed)
    frames, elas = [], {}
    for cat in categories:
        e = _benchmark_elasticity(cat, -1.8)
        elas[cat] = e
        lp = rng.normal(0.0, 0.22, n_weeks)
        acv = rng.normal(0.0, 1.0, n_weeks)
        tpr = rng.uniform(0.0, 0.3, n_weeks)
        lu = 6.0 + e * lp + 0.3 * acv + 0.8 * tpr + rng.normal(0.0, 0.05, n_weeks)
        frames.append(
            pd.DataFrame(
                {
                    "ppg_id": cat,
                    "week_start": _weeks(n_weeks),
                    "log_price": lp,
                    "log_units": lu,
                    "log_distribution_acv": acv,
                    "tpr_share": tpr,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), Truth(elasticity=elas, controls=CONTROLS)


def confounded_with_instrument(
    category: str = "cereal", n_weeks: int = 400, seed: int = 0
) -> tuple[pd.DataFrame, Truth]:
    """Price endogenous via an unobserved demand confounder; ``log_cost`` is a
    valid instrument. Naive OLS is biased; IV / Double-ML should recover truth."""
    rng = np.random.default_rng(seed)
    e = _benchmark_elasticity(category, -2.0)
    z = rng.normal(0.0, 1.0, n_weeks)  # cost shifter (instrument)
    conf = rng.normal(0.0, 1.0, n_weeks)  # unobserved confounder
    lp = 0.6 * z + 0.7 * conf + rng.normal(0.0, 0.2, n_weeks)
    lu = 6.0 + e * lp + 0.9 * conf + rng.normal(0.0, 0.1, n_weeks)
    frame = pd.DataFrame(
        {
            "ppg_id": category,
            "week_start": _weeks(n_weeks),
            "log_price": lp,
            "log_units": lu,
            "log_cost": z,
        }
    )
    return frame, Truth(elasticity={category: e}, controls=[], instrument="log_cost")


def seasonal_series(
    category: str = "soft_drinks", n_weeks: int = 160, period: int = 52, seed: int = 0
) -> tuple[pd.DataFrame, Truth]:
    """Strong annual seasonality + price effect. For the time-series / deep
    forecast families (and exog models still recover the elasticity)."""
    rng = np.random.default_rng(seed)
    e = _benchmark_elasticity(category, -2.5)
    t = np.arange(n_weeks)
    season = 0.6 * np.sin(2 * np.pi * t / period) + 0.3 * np.cos(2 * np.pi * t / (period / 2))
    lp = rng.normal(0.0, 0.2, n_weeks)
    lu = 6.0 + e * lp + season + rng.normal(0.0, 0.05, n_weeks)
    frame = pd.DataFrame(
        {"ppg_id": category, "week_start": _weeks(n_weeks), "log_price": lp, "log_units": lu}
    )
    return frame, Truth(elasticity={category: e}, controls=[])


def store_panel(
    category: str = "yogurt", n_stores: int = 8, n_weeks: int = 60, seed: int = 0
) -> tuple[pd.DataFrame, Truth]:
    """Multi-store panel with store-level intercepts (entity heterogeneity). For
    panel FE/RE — the within-store elasticity is the truth."""
    rng = np.random.default_rng(seed)
    e = _benchmark_elasticity(category, -2.4)
    rows = []
    for s in range(n_stores):
        intercept = rng.normal(6.0, 0.7)  # store fixed effect
        lp = rng.normal(0.0, 0.2, n_weeks)
        lu = intercept + e * lp + rng.normal(0.0, 0.05, n_weeks)
        rows.append(
            pd.DataFrame(
                {
                    "ppg_id": category,
                    "grain_unit": f"store_{s}",
                    "week_start": _weeks(n_weeks),
                    "log_price": lp,
                    "log_units": lu,
                }
            )
        )
    return pd.concat(rows, ignore_index=True), Truth(elasticity={category: e}, controls=[])


def cross_price_pair(
    categories: tuple[str, str] = ("soft_drinks", "beer"),
    n_weeks: int = 140,
    cross: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, Truth]:
    """Two substitute PPGs: each PPG's units rise with the other's price. For the
    cross-price demand system (own < 0 on the diagonal, cross > 0 off-diagonal)."""
    rng = np.random.default_rng(seed)
    a, b = categories
    ea = _benchmark_elasticity(a, -3.0)
    eb = _benchmark_elasticity(b, -1.3)
    lpa = rng.normal(0.0, 0.2, n_weeks)
    lpb = rng.normal(0.0, 0.2, n_weeks)
    lua = 6.0 + ea * lpa + cross * lpb + rng.normal(0.0, 0.05, n_weeks)
    lub = 6.0 + eb * lpb + cross * lpa + rng.normal(0.0, 0.05, n_weeks)
    wk = _weeks(n_weeks)

    def _blk(ppg, lp, lu):
        return pd.DataFrame({"ppg_id": ppg, "week_start": wk, "log_price": lp, "log_units": lu})

    frame = pd.concat([_blk(a, lpa, lua), _blk(b, lpb, lub)], ignore_index=True)
    truth = Truth(
        elasticity={a: ea, b: eb},
        controls=[],
        cross={a: {a: ea, b: cross}, b: {b: eb, a: cross}},
    )
    return frame, truth


def outlier_promo(
    category: str = "analgesics", n_weeks: int = 150, n_outliers: int = 8, seed: int = 0
) -> tuple[pd.DataFrame, Truth]:
    """Clean log-log relationship contaminated by a handful of extreme promo
    weeks (leverage points). Robust fitters should track the bulk elasticity
    better than plain OLS."""
    rng = np.random.default_rng(seed)
    e = _benchmark_elasticity(category, -1.85)
    lp = rng.normal(0.0, 0.2, n_weeks)
    lu = 6.0 + e * lp + rng.normal(0.0, 0.05, n_weeks)
    idx = rng.choice(n_weeks, size=n_outliers, replace=False)
    lp[idx] -= rng.uniform(0.4, 0.7, n_outliers)  # deep discounts
    lu[idx] += rng.uniform(1.5, 3.0, n_outliers)  # disproportionate spikes
    frame = pd.DataFrame(
        {"ppg_id": category, "week_start": _weeks(n_weeks), "log_price": lp, "log_units": lu}
    )
    return frame, Truth(elasticity={category: e}, controls=[])
