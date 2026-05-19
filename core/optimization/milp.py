"""PuLP MILP for discrete price-ladder + promo selection per PPG.

Single-PPG problem: choose exactly one (multiplier, promo) cell from the
configured ladder × promo states that maximises revenue or margin
subject to:

- **Margin floor** — chosen price must clear ``cog_pct * base + margin_floor_pct * base``.
- **Competitive gap** — chosen price must sit within ``±comp_gap_pct``
  of the competitor's reference price (skipped if competitor data
  missing).
- **Move guardrail** — chosen multiplier must lie in
  ``[1 - max_decrease, 1 + max_increase]``.

Cell-level values (units / revenue / margin) are precomputed by
``core.optimization.predict.cell_metrics`` and fed in as parameters, so
the MILP itself is linear: pick exactly one cell, maximise the chosen
cell's objective value.

When no cell is feasible, the solver falls back to a **soft-constraint
relaxation**: each constraint is encoded as a slack variable with a
penalty term in the objective, the problem is re-solved, and the
``binding_violations`` list reports which constraint was relaxed and by
how much. The agent uses this to narrate the trade-off ("relaxed margin
floor by 1.2pp to keep PPG_03 within the comp gap").
"""
from __future__ import annotations

from dataclasses import dataclass

import pulp

from core.optimization.constraints import OptimizationConstraints, PPGOptInputs
from core.optimization.predict import cell_metrics


VIOLATION_PENALTY = 1e6  # large enough to dominate any per-cell objective value


@dataclass
class MILPResult:
    ppg_id: str
    price_multiplier: float
    price: float
    promo: int
    units: float
    revenue: float
    margin: float
    objective_value: float
    objective_kind: str
    feasible_strict: bool
    relaxed: bool
    binding_violations: list[dict]
    n_cells_considered: int
    n_cells_feasible: int


def _cell_feasibility(
    inp: PPGOptInputs,
    c: OptimizationConstraints,
    multiplier: float,
) -> dict[str, float]:
    """Return per-cell constraint slack values (>=0 means feasible).

    Each key maps to the amount by which the cell satisfies the
    constraint. Negative values flag the violation magnitude in the same
    units as the slack (currency for price-based constraints).
    """
    price = inp.base_price * multiplier
    floor_price = c.cog_pct * inp.base_price + c.margin_floor_pct * inp.base_price
    slacks: dict[str, float] = {
        "margin_floor": float(price - floor_price),
        "move_lower": float(multiplier - (1.0 - c.max_decrease)),
        "move_upper": float((1.0 + c.max_increase) - multiplier),
    }
    if inp.competitor_price is not None and inp.competitor_price > 0:
        cp_lo = (1.0 - c.comp_gap_pct) * inp.competitor_price
        cp_hi = (1.0 + c.comp_gap_pct) * inp.competitor_price
        slacks["comp_gap_lower"] = float(price - cp_lo)
        slacks["comp_gap_upper"] = float(cp_hi - price)
    return slacks


def _all_cells(
    inp: PPGOptInputs, c: OptimizationConstraints
) -> list[dict]:
    cells: list[dict] = []
    for mult in c.price_ladder:
        for promo in c.promo_states:
            metrics = cell_metrics(
                inp.coefficients,
                inp.base_price,
                inp.base_price * mult,
                promo=int(promo),
                model_kind=inp.model_kind,
                context=inp.context,
                cog_pct=c.cog_pct,
            )
            slacks = _cell_feasibility(inp, c, mult)
            cells.append(
                {
                    "multiplier": float(mult),
                    "promo": int(promo),
                    "price": float(inp.base_price * mult),
                    "metrics": metrics,
                    "slacks": slacks,
                    "feasible": all(v >= -1e-9 for v in slacks.values()),
                }
            )
    return cells


def _solve_strict(
    cells: list[dict], c: OptimizationConstraints, ppg_id: str
) -> tuple[MILPResult | None, list[dict]]:
    """Pick exactly one feasible cell maximising the objective."""
    feasible = [cell for cell in cells if cell["feasible"]]
    if not feasible:
        return None, feasible

    prob = pulp.LpProblem(f"price_milp_{ppg_id}", pulp.LpMaximize)
    x: dict[int, pulp.LpVariable] = {
        idx: pulp.LpVariable(f"x_{idx}", cat="Binary") for idx in range(len(feasible))
    }
    prob += pulp.lpSum(
        feasible[idx]["metrics"][c.objective] * x[idx] for idx in x
    )
    prob += pulp.lpSum(x.values()) == 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None, feasible

    chosen = next((idx for idx, v in x.items() if pulp.value(v) > 0.5), None)
    if chosen is None:
        return None, feasible

    cell = feasible[chosen]
    return (
        MILPResult(
            ppg_id=ppg_id,
            price_multiplier=cell["multiplier"],
            price=cell["price"],
            promo=cell["promo"],
            units=cell["metrics"]["units"],
            revenue=cell["metrics"]["revenue"],
            margin=cell["metrics"]["margin"],
            objective_value=cell["metrics"][c.objective],
            objective_kind=c.objective,
            feasible_strict=True,
            relaxed=False,
            binding_violations=[],
            n_cells_considered=len(cells),
            n_cells_feasible=len(feasible),
        ),
        feasible,
    )


def _solve_relaxed(
    cells: list[dict], c: OptimizationConstraints, ppg_id: str
) -> MILPResult:
    """Soft-constraint fallback: minimise sum-of-violations with the
    objective as a secondary criterion (scaled down by the penalty).

    Encoded as a MILP so the linear-programming view extends cleanly:
    pick one cell, pay the cell's violation magnitudes via positive slack
    variables, maximise ``-VIOLATION_PENALTY * Σ slack + objective_value``.
    """
    prob = pulp.LpProblem(f"price_milp_relaxed_{ppg_id}", pulp.LpMaximize)
    x: dict[int, pulp.LpVariable] = {
        idx: pulp.LpVariable(f"x_{idx}", cat="Binary") for idx in range(len(cells))
    }
    prob += pulp.lpSum(x.values()) == 1

    # Each cell has a fixed total-violation magnitude (sum of negative
    # slack absolute values). Treat the cell's violation as a parameter:
    # objective term per cell = obj - PENALTY * violation_total.
    coeffs: list[float] = []
    for cell in cells:
        violation = sum(max(0.0, -float(v)) for v in cell["slacks"].values())
        coeffs.append(cell["metrics"][c.objective] - VIOLATION_PENALTY * violation)
    prob += pulp.lpSum(coeffs[idx] * x[idx] for idx in x)

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        # No solver can make this infeasible (always exactly-one selectable),
        # so this branch is paranoia: pick the least-violating cell directly.
        idx = min(range(len(cells)), key=lambda i: -coeffs[i])
    else:
        idx = next(i for i, v in x.items() if pulp.value(v) > 0.5)

    cell = cells[idx]
    violations = [
        {"constraint": name, "magnitude": float(-slack)}
        for name, slack in cell["slacks"].items()
        if slack < -1e-9
    ]
    return MILPResult(
        ppg_id=ppg_id,
        price_multiplier=cell["multiplier"],
        price=cell["price"],
        promo=cell["promo"],
        units=cell["metrics"]["units"],
        revenue=cell["metrics"]["revenue"],
        margin=cell["metrics"]["margin"],
        objective_value=cell["metrics"][c.objective],
        objective_kind=c.objective,
        feasible_strict=False,
        relaxed=True,
        binding_violations=violations,
        n_cells_considered=len(cells),
        n_cells_feasible=0,
    )


def solve_milp(inp: PPGOptInputs, c: OptimizationConstraints) -> MILPResult:
    """Strict MILP solve; fall back to soft-constraint relaxation if infeasible."""
    cells = _all_cells(inp, c)
    strict, _feasible = _solve_strict(cells, c, inp.ppg_id)
    if strict is not None:
        return strict
    return _solve_relaxed(cells, c, inp.ppg_id)
