"""Nonparametric / Bayesian-linear plugins recover sign + feed the predictor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.library import registry
from core.models.library.base import FitContext
from core.models.predictor import build_predictor
from core.models.result import to_elasticity_fit


def _demand_frame(n: int = 200, beta: float = -1.6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_price = rng.normal(0.0, 0.3, n)
    ctrl = rng.normal(0.0, 1.0, n)
    log_units = 5.0 + beta * log_price + 0.3 * ctrl + rng.normal(0.0, 0.05, n)
    return pd.DataFrame(
        {
            "log_price": log_price,
            "log_units": log_units,
            "ctrl": ctrl,
            "week_start": pd.date_range("2021-01-01", periods=n, freq="W"),
        }
    )


@pytest.mark.parametrize("key", ["bayesian_ridge", "gaussian_process", "svr", "knn"])
def test_recovers_negative_sign_and_predicts(key: str) -> None:
    frame = _demand_frame()
    result = registry.get(key).fit(frame, FitContext(ppg_id="P1", controls=["ctrl"]))
    assert result.sign_ok, f"{key} got {result.own_elasticity}"
    fit = to_elasticity_fit(result)
    assert fit is not None
    row = {"ppg_id": "P1", "winner_model": key, "winner": fit.to_dict()}
    preds = build_predictor(row, frame, ["ctrl"], test_ratio=0.0).predict_log(frame)
    assert len(preds) == len(frame)
    assert np.isfinite(preds).all()


def test_bayesian_ridge_is_linear_coeff() -> None:
    result = registry.get("bayesian_ridge").fit(
        _demand_frame(), FitContext(ppg_id="P1", controls=["ctrl"])
    )
    assert "const" in result.coefficients
    assert "log_price" in result.coefficients
