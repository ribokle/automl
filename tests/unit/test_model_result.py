"""ModelResult <-> ElasticityFit adapter contract."""
from __future__ import annotations

import math

from core.models.base import ElasticityFit
from core.models.result import (
    Capability,
    ForecastBlock,
    ModelResult,
    ProblemType,
    from_elasticity_fit,
    to_elasticity_fit,
)


def test_round_trip_elasticity_fit() -> None:
    ef = ElasticityFit(
        ppg_id="P1",
        model="loglog_ols",
        own_elasticity=-1.4,
        std_err=0.2,
        p_value=0.01,
        r_squared=0.8,
        n_obs=100,
        controls=["ctrl"],
        coefficients={"const": 5.0, "log_price": -1.4, "ctrl": 0.3},
        diagnostics={"test_wape": 0.12},
    )
    result = from_elasticity_fit(ef)
    back = to_elasticity_fit(result)
    assert back is not None
    assert back.own_elasticity == -1.4
    assert back.coefficients == ef.coefficients
    assert back.diagnostics["test_wape"] == 0.12
    assert back.sign_ok


def test_forecast_only_has_no_scalar_elasticity() -> None:
    result = ModelResult(
        ppg_id="P1",
        model="sarimax",
        problem_type=ProblemType.FORECAST,
        forecast=ForecastBlock(horizon=4, index=["w1"], mean=[10.0]),
        capabilities=Capability.FORECAST,
    )
    assert to_elasticity_fit(result) is None


def test_cross_price_diagonal_becomes_own_elasticity() -> None:
    result = ModelResult(
        ppg_id="P1",
        model="aids",
        problem_type=ProblemType.CROSS_PRICE,
        cross_price={"P1": {"P1": -1.2, "P2": 0.3}, "P2": {"P1": 0.25, "P2": -0.9}},
        capabilities=Capability.CROSS_PRICE_MATRIX,
    )
    fit = to_elasticity_fit(result)
    assert fit is not None
    assert fit.own_elasticity == -1.2
    assert fit.diagnostics["cross_price_matrix"]["P1"]["P2"] == 0.3


def test_missing_std_err_maps_to_nan() -> None:
    result = ModelResult(ppg_id="P1", model="ridge", own_elasticity=-1.0, std_err=None)
    fit = to_elasticity_fit(result)
    assert fit is not None
    assert math.isnan(fit.std_err)
