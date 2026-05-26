"""Double-ML recovers a negative, confounding-corrected elasticity."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.models.library import registry
from core.models.library.base import FitContext
from core.models.predictor import LINEAR_COEFF_MODELS, build_predictor
from core.models.result import to_elasticity_fit


def _confounded_frame(n: int = 160, beta: float = -1.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ctrl = rng.normal(0.0, 1.0, n)
    # price is driven partly by the control -> naive OLS is confounded
    log_price = 0.5 * ctrl + rng.normal(0.0, 0.2, n)
    log_units = 5.0 + beta * log_price + 0.8 * ctrl + rng.normal(0.0, 0.05, n)
    return pd.DataFrame(
        {
            "log_price": log_price,
            "log_units": log_units,
            "ctrl": ctrl,
            "week_start": pd.date_range("2021-01-01", periods=n, freq="W"),
        }
    )


def test_double_ml_recovers_negative_elasticity() -> None:
    frame = _confounded_frame()
    result = registry.get("double_ml").fit(frame, FitContext(ppg_id="P1", controls=["ctrl"]))
    assert result.sign_ok, result.own_elasticity
    assert -3.0 < result.own_elasticity < -0.5
    assert "const" in result.coefficients and "log_price" in result.coefficients


def test_double_ml_is_linear_coeff_and_predicts() -> None:
    assert "double_ml" in LINEAR_COEFF_MODELS
    frame = _confounded_frame()
    result = registry.get("double_ml").fit(frame, FitContext(ppg_id="P1", controls=["ctrl"]))
    fit = to_elasticity_fit(result)
    assert fit is not None
    row = {"ppg_id": "P1", "winner_model": "double_ml", "winner": fit.to_dict()}
    preds = build_predictor(row, frame, ["ctrl"], test_ratio=0.0).predict_log(frame)
    assert np.isfinite(preds).all()
