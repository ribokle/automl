"""Tree-ensemble plugins recover elasticity sign + refit through the predictor.

xgboost / catboost are optional; their tests are skipped cleanly when the
dependency isn't installed (verifying the graceful-skip contract).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.library import registry
from core.models.library.base import FitContext
from core.models.predictor import build_predictor
from core.models.result import to_elasticity_fit

_NO_DEP_TREES = ["random_forest", "extra_trees"]
_OPTIONAL_TREES = ["xgboost", "catboost"]


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


@pytest.mark.parametrize("key", _NO_DEP_TREES)
def test_sklearn_trees_recover_sign_and_predict(key: str) -> None:
    frame = _demand_frame()
    result = registry.get(key).fit(frame, FitContext(ppg_id="P1", controls=["ctrl"]))
    assert result.sign_ok, f"{key} got {result.own_elasticity}"
    fit = to_elasticity_fit(result)
    assert fit is not None
    row = {"ppg_id": "P1", "winner_model": key, "winner": fit.to_dict()}
    preds = build_predictor(row, frame, ["ctrl"], test_ratio=0.0).predict_log(frame)
    assert len(preds) == len(frame)
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("key", _OPTIONAL_TREES)
def test_optional_trees_skip_cleanly_when_absent(key: str) -> None:
    plugin = registry.get(key)
    if not plugin.is_available():
        pytest.skip(f"{key} dependency not installed")
    result = plugin.fit(_demand_frame(), FitContext(ppg_id="P1", controls=["ctrl"]))
    assert result.own_elasticity is not None
