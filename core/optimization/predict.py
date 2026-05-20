"""Closed-form unit / revenue / margin prediction for one (price, promo) cell.

Mirrors ``core.simulation.grid.simulate_ols_grid`` but operates on a single
cell so the optimisers can call it as a black-box scalar function.
Keeping this in its own module makes the math sharable between the scipy
continuous solver and the PuLP MILP feasibility checker — and easy to
unit-test against the simulator's grid output.

For LightGBM-winning PPGs the OLS closed-form isn't applicable. The
``cell_metrics_via_predictor`` helper consumes the shared
:class:`core.models.predictor.Predictor` instead, so the MILP can score
LightGBM cells exactly the same way it scores OLS cells.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def predict_units(
    coefficients: dict[str, float],
    base_price: float,
    price: float,
    promo: int,
    *,
    model_kind: str,
    context: dict[str, float],
    promo_features: tuple[str, ...] = ("tpr_share",),
) -> float:
    """Predict units at one (price, promo) cell for one PPG."""
    if base_price <= 0 or price <= 0:
        return 0.0

    swept: set[str] = set(promo_features)
    if model_kind == "loglog_ols":
        swept.update({"log_price", "log_price_gap", "log_base_price"})
    elif model_kind == "semilog_ols":
        swept.add("price")
    else:
        raise ValueError(f"unsupported model_kind={model_kind!r}")

    log_units = float(coefficients.get("const", 0.0))
    for col, beta in coefficients.items():
        if col == "const" or col in swept:
            continue
        log_units += float(beta) * float(context.get(col, 0.0))

    if model_kind == "loglog_ols":
        log_price = math.log(price)
        log_units += float(coefficients.get("log_price", 0.0)) * log_price
        if "log_price_gap" in coefficients:
            comp_ref = float(context.get("log_competitor_price", math.log(base_price)))
            log_units += float(coefficients["log_price_gap"]) * (log_price - comp_ref)
        if "log_base_price" in coefficients:
            log_units += float(coefficients["log_base_price"]) * math.log(base_price)
    else:  # semilog_ols
        log_units += float(coefficients.get("price", 0.0)) * price

    for col in promo_features:
        if col in coefficients:
            log_units += float(coefficients[col]) * float(promo)

    return math.exp(log_units)


def cell_metrics(
    coefficients: dict[str, float],
    base_price: float,
    price: float,
    promo: int,
    *,
    model_kind: str,
    context: dict[str, float],
    cog_pct: float,
) -> dict[str, float]:
    """units / revenue / margin for one (price, promo) cell."""
    units = predict_units(
        coefficients,
        base_price,
        price,
        promo,
        model_kind=model_kind,
        context=context,
    )
    cost_per_unit = max(0.0, cog_pct) * base_price
    revenue = price * units
    margin = (price - cost_per_unit) * units
    return {"units": units, "revenue": revenue, "margin": margin}


def _build_cell_row(
    predictor,
    base_price: float,
    price: float,
    promo: int,
    context: dict[str, float],
    promo_features: tuple[str, ...],
) -> dict[str, float]:
    """One row of feature inputs for a (price, promo) cell.

    Mirrors the grid-sweep row builder in
    :func:`core.simulation.grid.simulate_predictor_grid` but at single-cell
    granularity. Columns absent from ``context`` default to zero; the
    predictor handles missing columns the same way.
    """
    log_base_price = math.log(base_price)
    row: dict[str, float] = dict(context)
    swept_price_cols: set[str] = (
        {"log_price", "log_price_gap", "log_base_price"}
        if predictor.model_kind in ("loglog_ols", "lightgbm")
        else {"price"}
    )
    for col in predictor.feature_cols:
        if col not in row and col not in swept_price_cols and col not in promo_features:
            row[col] = 0.0
    if predictor.model_kind in ("loglog_ols", "lightgbm"):
        log_price = math.log(price)
        row["log_price"] = log_price
        row["log_base_price"] = log_base_price
        row["log_price_gap"] = log_price - context.get("log_competitor_price", log_base_price)
    else:  # semilog_ols
        row["price"] = float(price)
    for col in promo_features:
        row[col] = float(promo)
    return row


def cell_metrics_via_predictor(
    predictor,
    base_price: float,
    price: float,
    promo: int,
    *,
    cog_pct: float,
    context: dict[str, float],
    promo_features: tuple[str, ...] = ("tpr_share",),
) -> dict[str, float]:
    """``cell_metrics`` analogue routed through a ``Predictor``.

    Works for OLS and LightGBM winners alike. The single-cell prediction
    is one call to ``predictor.predict_log`` over a 1-row frame so the
    LightGBM path stays vectorised inside the booster.
    """
    if base_price <= 0 or price <= 0:
        return {"units": 0.0, "revenue": 0.0, "margin": 0.0}
    row = _build_cell_row(predictor, base_price, price, promo, context, promo_features)
    frame = pd.DataFrame([row])[predictor.feature_cols]
    log_units = float(np.asarray(predictor.predict_log(frame))[0])
    units = float(math.exp(log_units))
    cost_per_unit = max(0.0, cog_pct) * base_price
    return {
        "units": units,
        "revenue": float(price * units),
        "margin": float((price - cost_per_unit) * units),
    }


def predict_units_via_predictor(
    predictor,
    base_price: float,
    price: float,
    promo: int,
    *,
    context: dict[str, float],
    promo_features: tuple[str, ...] = ("tpr_share",),
) -> float:
    """Scalar units prediction at one cell for the scipy continuous solver."""
    if base_price <= 0 or price <= 0:
        return 0.0
    row = _build_cell_row(predictor, base_price, price, promo, context, promo_features)
    frame = pd.DataFrame([row])[predictor.feature_cols]
    log_units = float(np.asarray(predictor.predict_log(frame))[0])
    return float(math.exp(log_units))
