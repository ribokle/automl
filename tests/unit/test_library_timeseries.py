"""Time-series plugins produce forecasts (+ elasticity for exog models)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.library import registry
from core.models.library.base import FitContext
from core.models.result import Capability


def _series(n: int = 130, beta: float = -1.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_price = rng.normal(0.0, 0.2, n)
    season = np.sin(np.arange(n) * 2 * np.pi / 52)
    log_units = 5.0 + beta * log_price + 0.4 * season + rng.normal(0.0, 0.05, n)
    return pd.DataFrame(
        {
            "log_price": log_price,
            "log_units": log_units,
            "week_start": pd.date_range("2020-01-01", periods=n, freq="W"),
        }
    )


def _fit(key: str):
    frame = _series()
    train, test = frame.iloc[:104], frame.iloc[104:]
    return registry.get(key).fit(train, FitContext(ppg_id="P1", controls=[], test=test))


@pytest.mark.parametrize("key", ["arimax", "sarimax", "ets", "holt_winters", "state_space"])
def test_produces_forecast_with_wape(key: str) -> None:
    result = _fit(key)
    assert result.forecast is not None
    assert result.forecast.horizon == 26
    assert len(result.forecast.mean) == 26
    assert Capability.FORECAST in result.capabilities
    assert "test_wape" in result.diagnostics


@pytest.mark.parametrize("key", ["arimax", "sarimax", "state_space"])
def test_exog_models_report_negative_elasticity(key: str) -> None:
    result = _fit(key)
    assert result.own_elasticity is not None
    assert result.own_elasticity < 0


@pytest.mark.parametrize("key", ["ets", "holt_winters"])
def test_smoothing_models_have_no_elasticity(key: str) -> None:
    assert _fit(key).own_elasticity is None


@pytest.mark.parametrize("key", ["prophet", "tbats"])
def test_optional_ts_skip_cleanly_when_absent(key: str) -> None:
    plugin = registry.get(key)
    if not plugin.is_available():
        pytest.skip(f"{key} dependency not installed")
    assert _fit(key).forecast is not None
