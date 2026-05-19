"""Closed-form unit / revenue / margin prediction for one (price, promo) cell.

Mirrors ``core.simulation.grid.simulate_ols_grid`` but operates on a single
cell so the optimisers can call it as a black-box scalar function.
Keeping this in its own module makes the math sharable between the scipy
continuous solver and the PuLP MILP feasibility checker — and easy to
unit-test against the simulator's grid output.
"""
from __future__ import annotations

import math


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
