"""scipy bounded scalar optimisation for the price multiplier.

Continuous relaxation: ignore the discrete ladder, ignore promo (we'll
sweep promo states ourselves), search the continuous price multiplier
interval ``[1 - max_decrease, 1 + max_increase]`` and pin both the
margin-floor lower bound and the competitive-gap window into the search
bounds before handing off to ``scipy.optimize.minimize_scalar``.

The result is the warm-start anchor the MILP rounds onto the nearest
allowed-ladder multiplier. If the MILP is infeasible (constraints
incompatible) the continuous solution is also the fallback the agent
returns with a relaxation note.
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy.optimize import minimize_scalar

from core.optimization.constraints import OptimizationConstraints, PPGOptInputs
from core.optimization.predict import cell_metrics


@dataclass
class ContinuousResult:
    ppg_id: str
    price_multiplier: float
    price: float
    promo: int
    units: float
    revenue: float
    margin: float
    objective_value: float
    bounds_used: tuple[float, float]
    feasible: bool


def _multiplier_bounds(inp: PPGOptInputs, c: OptimizationConstraints) -> tuple[float, float]:
    """Intersect the move guardrail, the margin floor, and the comp gap.

    The margin floor sets a *minimum* price; the comp gap sets a window
    around the competitor's price; the move guardrail clamps the change
    from base. The result is the multiplier interval all three rules
    allow. If the intersection is empty we return the move guardrail
    interval unchanged and flag ``feasible=False`` upstream.
    """
    lo = 1.0 - c.max_decrease
    hi = 1.0 + c.max_increase

    floor_price = c.cog_pct * inp.base_price + c.margin_floor_pct * inp.base_price
    lo = max(lo, floor_price / inp.base_price)

    if inp.competitor_price is not None and inp.competitor_price > 0:
        lo = max(lo, (1.0 - c.comp_gap_pct) * inp.competitor_price / inp.base_price)
        hi = min(hi, (1.0 + c.comp_gap_pct) * inp.competitor_price / inp.base_price)

    return lo, hi


def solve_continuous(
    inp: PPGOptInputs, c: OptimizationConstraints
) -> ContinuousResult:
    """Best continuous price multiplier per promo state, picked across states."""
    lo, hi = _multiplier_bounds(inp, c)
    feasible = hi > lo

    if not feasible:
        # The constraint set is empty — degrade gracefully by returning
        # base price; the MILP step will flag the relaxation.
        mid = max(min(1.0, hi), lo)
        metrics = cell_metrics(
            inp.coefficients,
            inp.base_price,
            inp.base_price * mid,
            promo=int(c.promo_states[0]),
            model_kind=inp.model_kind,
            context=inp.context,
            cog_pct=c.cog_pct,
        )
        return ContinuousResult(
            ppg_id=inp.ppg_id,
            price_multiplier=float(mid),
            price=float(inp.base_price * mid),
            promo=int(c.promo_states[0]),
            units=metrics["units"],
            revenue=metrics["revenue"],
            margin=metrics["margin"],
            objective_value=metrics[c.objective],
            bounds_used=(float(lo), float(hi)),
            feasible=False,
        )

    best: ContinuousResult | None = None
    for promo in c.promo_states:
        def neg_objective(m: float, promo: int = int(promo)) -> float:
            metrics = cell_metrics(
                inp.coefficients,
                inp.base_price,
                inp.base_price * m,
                promo=promo,
                model_kind=inp.model_kind,
                context=inp.context,
                cog_pct=c.cog_pct,
            )
            return -metrics[c.objective]

        res = minimize_scalar(neg_objective, bounds=(lo, hi), method="bounded")
        m_star = float(res.x)
        metrics = cell_metrics(
            inp.coefficients,
            inp.base_price,
            inp.base_price * m_star,
            promo=int(promo),
            model_kind=inp.model_kind,
            context=inp.context,
            cog_pct=c.cog_pct,
        )
        candidate = ContinuousResult(
            ppg_id=inp.ppg_id,
            price_multiplier=m_star,
            price=float(inp.base_price * m_star),
            promo=int(promo),
            units=metrics["units"],
            revenue=metrics["revenue"],
            margin=metrics["margin"],
            objective_value=metrics[c.objective],
            bounds_used=(float(lo), float(hi)),
            feasible=True,
        )
        if best is None or candidate.objective_value > best.objective_value:
            best = candidate

    assert best is not None
    return best
