"""LightGBM-backed grid simulator + MILP optimisation cell-scoring.

The closed-form OLS tests already cover the OLS branches; these tests
exercise the predictor-driven path the Phase-4b refactor introduced.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.models.predictor import build_predictor
from core.optimization.constraints import OptimizationConstraints, PPGOptInputs
from core.optimization.milp import solve_milp
from core.optimization.predict import (
    predict_units_via_predictor,
)
from core.simulation.grid import (
    ScenarioGridSpec,
    simulate_predictor_grid,
)


def _toy_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    log_base_price = np.log(3.0) * np.ones(n)
    log_price = log_base_price + np.log(0.8 + 0.4 * rng.random(n))
    return pd.DataFrame(
        {
            "ppg_id": "PPG_L",
            "week_start": pd.date_range("2024-01-01", periods=n, freq="W").astype(str),
            "log_units": 6.5 - 2.0 * log_price + rng.normal(0, 0.05, size=n),
            "log_price": log_price,
            "log_base_price": log_base_price,
            "tpr_share": rng.binomial(1, 0.3, size=n).astype(float),
            "log_distribution_acv": np.log(70 + 20 * rng.random(n)),
        }
    )


def _lightgbm_predictor():
    frame = _toy_frame()
    return build_predictor(
        {"ppg_id": "PPG_L", "winner_model": "lightgbm", "winner": {}},
        frame,
        controls=["tpr_share", "log_distribution_acv"],
        test_ratio=0.0,
    )


def test_lightgbm_grid_units_decrease_with_price() -> None:
    """Elastic DGP (slope -2): the trained booster's predicted units
    should fall monotonically (or close to it) as price rises across the
    grid. We accept some tree-step roughness — assert the trend rather
    than strict monotonicity."""
    predictor = _lightgbm_predictor()
    spec = ScenarioGridSpec(
        promo_states=(0,),
        context={"log_distribution_acv": math.log(85), "tpr_share": 0.0},
    )
    grid = simulate_predictor_grid(predictor, base_price=3.0, spec=spec)
    grid = grid.sort_values("price").reset_index(drop=True)
    # Average slope: linear regression of log(units) on log(price) should
    # be negative.
    slope = np.polyfit(np.log(grid["price"]), np.log(grid["units"]), 1)[0]
    assert slope < -0.5


def test_lightgbm_cell_metrics_match_grid_cell() -> None:
    """Single-cell scoring through the predictor must equal the grid sweep
    cell-by-cell."""
    predictor = _lightgbm_predictor()
    spec = ScenarioGridSpec(
        price_multipliers=(0.9, 1.0, 1.1),
        promo_states=(0, 1),
        context={"log_distribution_acv": math.log(85), "tpr_share": 0.0},
    )
    grid = simulate_predictor_grid(predictor, base_price=3.0, spec=spec)
    for _, row in grid.iterrows():
        u = predict_units_via_predictor(
            predictor,
            base_price=3.0,
            price=float(row["price"]),
            promo=int(row["promo"]),
            context={"log_distribution_acv": math.log(85), "tpr_share": 0.0},
        )
        assert math.isclose(u, float(row["units"]), rel_tol=1e-9)


def test_lightgbm_milp_picks_ladder_cell() -> None:
    predictor = _lightgbm_predictor()
    inp = PPGOptInputs(
        ppg_id="PPG_L",
        model_kind="lightgbm",
        coefficients={},
        base_price=3.0,
        context={"log_distribution_acv": math.log(85), "tpr_share": 0.0},
        predictor=predictor,
    )
    c = OptimizationConstraints(
        price_ladder=(0.90, 0.95, 1.00, 1.05, 1.10),
        margin_floor_pct=0.0,
        comp_gap_pct=1.0,
        objective="revenue",
    )
    res = solve_milp(inp, c)
    assert res.feasible_strict is True
    assert res.price_multiplier in c.price_ladder
    # Elastic demand + revenue objective -> the optimiser should sit at
    # the lower end of the ladder (lowest price = highest revenue when
    # |ε|>1).
    assert res.price_multiplier <= 1.0
