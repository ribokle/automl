"""Regularized plugins recover elasticity sign + feed the predictor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.library import registry
from core.models.library.base import FitContext
from core.models.predictor import build_predictor
from core.models.result import to_elasticity_fit


def _demand_frame(n: int = 160, beta: float = -1.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_price = rng.normal(0.0, 0.25, n)
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


@pytest.mark.parametrize("key", ["ridge", "lasso", "elasticnet"])
def test_recovers_negative_sign(key: str) -> None:
    frame = _demand_frame()
    plugin = registry.get(key)
    ctx = FitContext(ppg_id="P1", controls=["ctrl"], test=None)
    result = plugin.fit(frame, ctx)
    assert result.sign_ok, f"{key} got positive elasticity {result.own_elasticity}"
    assert "const" in result.coefficients


@pytest.mark.parametrize("key", ["ridge", "lasso", "elasticnet"])
def test_test_wape_reported_when_holdout_supplied(key: str) -> None:
    frame = _demand_frame()
    train, test = frame.iloc[:120], frame.iloc[120:]
    result = registry.get(key).fit(train, FitContext(ppg_id="P1", controls=["ctrl"], test=test))
    assert "test_wape" in result.diagnostics


def test_regularized_winner_drives_predictor() -> None:
    frame = _demand_frame()
    result = registry.get("ridge").fit(frame, FitContext(ppg_id="P1", controls=["ctrl"]))
    fit = to_elasticity_fit(result)
    assert fit is not None
    modeling_row = {"ppg_id": "P1", "winner_model": "ridge", "winner": fit.to_dict()}
    predictor = build_predictor(modeling_row, frame, ["ctrl"], test_ratio=0.0)
    preds = predictor.predict_log(frame)
    assert len(preds) == len(frame)
    assert np.isfinite(preds).all()
