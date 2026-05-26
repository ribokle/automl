"""Vectorised price × promo scenario grid simulator.

Given an OLS coefficient dict from the modelling agent + a representative
"context row" (PPG-week feature values to hold constant while sweeping
price and promo), the simulator predicts units, revenue, and margin
across every cell of a configurable grid. Used both as input to the
optimisation stage and as a standalone what-if surface for the UI.

The math mirrors the decomposition module: ``log_units = α + Σ βᵢ·xᵢ``.
For each grid cell we update the price + promo columns and leave
everything else fixed at the context-row values, then exponentiate to
units. The whole grid is one vector op per coefficient so a 21×2 sweep
for one PPG costs microseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd


DEFAULT_PRICE_MULTIPLIERS: tuple[float, ...] = (
    0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98,
    1.00,
    1.02, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20,
)
DEFAULT_PROMO_STATES: tuple[int, ...] = (0, 1)


@dataclass
class ScenarioGridSpec:
    price_multipliers: tuple[float, ...] = DEFAULT_PRICE_MULTIPLIERS
    promo_states: tuple[int, ...] = DEFAULT_PROMO_STATES
    promo_features: tuple[str, ...] = ("tpr_share",)
    cost_of_goods_pct: float = 0.55  # placeholder margin assumption
    context: dict[str, float] = field(default_factory=dict)


def _context_log_units(
    coefficients: dict[str, float],
    context: dict[str, float],
    excluded: set[str],
) -> float:
    """Log_units contribution from columns held constant in the grid sweep."""
    return float(coefficients.get("const", 0.0)) + sum(
        float(coefficients[c]) * float(context.get(c, 0.0))
        for c in coefficients
        if c != "const" and c not in excluded
    )


def simulate_ols_grid(
    coefficients: dict[str, float],
    base_price: float,
    spec: ScenarioGridSpec,
    *,
    model_kind: str = "loglog_ols",
) -> pd.DataFrame:
    """Sweep the price × promo grid for one PPG using OLS coefficients.

    ``model_kind`` selects how the price column is updated:
    ``loglog_ols`` uses ``log_price`` (and ``log_price_gap`` if present);
    ``semilog_ols`` uses raw ``price``. Promo features are pinned to the
    states in ``spec.promo_states`` (1 = active across the listed
    ``promo_features``, 0 = off everywhere).
    """
    if base_price <= 0:
        raise ValueError("base_price must be positive")

    # semilog is the only raw-price model; every other linear-coefficient
    # model is log-price space.
    is_semilog = model_kind == "semilog_ols"
    swept_cols: set[str] = set()
    if is_semilog:
        swept_cols.update({"price"})
    else:
        swept_cols.update({"log_price", "log_price_gap", "log_base_price"})
    swept_cols.update(spec.promo_features)

    fixed_log = _context_log_units(coefficients, spec.context, swept_cols)

    log_base_price = float(np.log(base_price))
    cog = max(0.0, min(0.95, float(spec.cost_of_goods_pct)))

    rows: list[dict] = []
    for mult, promo in product(spec.price_multipliers, spec.promo_states):
        price = base_price * mult
        log_units = fixed_log

        if is_semilog:
            log_units += float(coefficients.get("price", 0.0)) * price
        else:
            log_price = float(np.log(price))
            log_units += float(coefficients.get("log_price", 0.0)) * log_price
            if "log_price_gap" in coefficients:
                comp_ref = float(spec.context.get("log_competitor_price", log_base_price))
                log_units += float(coefficients["log_price_gap"]) * (log_price - comp_ref)
            if "log_base_price" in coefficients:
                log_units += float(coefficients["log_base_price"]) * log_base_price

        for col in spec.promo_features:
            if col in coefficients:
                log_units += float(coefficients[col]) * float(promo)

        units = float(np.exp(log_units))
        revenue = price * units
        margin = (price - cog * base_price) * units
        rows.append(
            {
                "price_multiplier": float(mult),
                "price": price,
                "promo": int(promo),
                "units": units,
                "revenue": revenue,
                "margin": margin,
            }
        )
    return pd.DataFrame(rows)


def simulate_predictor_grid(
    predictor,
    base_price: float,
    spec: ScenarioGridSpec,
) -> pd.DataFrame:
    """Sweep the price × promo grid for one PPG using a fitted ``Predictor``.

    Works for both OLS and LightGBM winners — the predictor abstracts the
    underlying functional form. The grid row builder mirrors
    :func:`simulate_ols_grid`'s sweep: only ``log_price``,
    ``log_price_gap`` (when in the model), ``log_base_price``, and the
    listed promo features change per cell; everything else is held at the
    context value (or zero if absent).
    """
    if base_price <= 0:
        raise ValueError("base_price must be positive")

    model_kind = predictor.model_kind
    log_base_price = float(np.log(base_price))
    cog = max(0.0, min(0.95, float(spec.cost_of_goods_pct)))

    swept_price_cols: set[str] = (
        {"log_price", "log_price_gap", "log_base_price"}
        if model_kind in ("loglog_ols", "lightgbm")
        else {"price"}
    )

    rows: list[dict] = []
    grid_inputs: list[dict] = []
    for mult, promo in product(spec.price_multipliers, spec.promo_states):
        price = base_price * mult
        row: dict[str, float] = dict(spec.context)
        # Fill any model column not in context with 0.0 — predictor handles
        # missing columns by zeroing them out, which matches log-units context.
        for col in predictor.feature_cols:
            if col not in row and col not in swept_price_cols and col not in spec.promo_features:
                row[col] = 0.0
        if model_kind in ("loglog_ols", "lightgbm"):
            row["log_price"] = float(np.log(price))
            row["log_base_price"] = log_base_price
            row["log_price_gap"] = float(
                np.log(price) - spec.context.get("log_competitor_price", log_base_price)
            )
        else:  # semilog_ols
            row["price"] = float(price)
        for col in spec.promo_features:
            row[col] = float(promo)
        grid_inputs.append({"mult": float(mult), "promo": int(promo), "price": price, **row})

    feature_frame = pd.DataFrame(grid_inputs)
    log_units = predictor.predict_log(feature_frame[predictor.feature_cols])
    units = np.exp(log_units)
    for i, info in enumerate(grid_inputs):
        u = float(units[i])
        rows.append(
            {
                "price_multiplier": info["mult"],
                "price": float(info["price"]),
                "promo": info["promo"],
                "units": u,
                "revenue": float(info["price"]) * u,
                "margin": (float(info["price"]) - cog * base_price) * u,
            }
        )
    return pd.DataFrame(rows)


def grid_summary(grid: pd.DataFrame) -> dict:
    """Highlight the best price/promo cell by each objective."""
    if grid.empty:
        return {}
    best_revenue = grid.loc[grid["revenue"].idxmax()]
    best_margin = grid.loc[grid["margin"].idxmax()]
    return {
        "n_cells": int(len(grid)),
        "best_revenue": {
            "price_multiplier": float(best_revenue["price_multiplier"]),
            "promo": int(best_revenue["promo"]),
            "revenue": float(best_revenue["revenue"]),
            "units": float(best_revenue["units"]),
        },
        "best_margin": {
            "price_multiplier": float(best_margin["price_multiplier"]),
            "promo": int(best_margin["promo"]),
            "margin": float(best_margin["margin"]),
            "units": float(best_margin["units"]),
        },
    }
