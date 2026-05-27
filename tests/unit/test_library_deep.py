"""Deep sequence forecasters (optional dep: torch). Forecast-only, opt-in."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.library import registry
from core.models.library.base import FitContext
from core.models.result import Capability

_DEEP_KEYS = ["lstm", "gru"]


def _series(n: int = 140, beta: float = -1.4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    season = np.sin(np.arange(n) * 2 * np.pi / 26)
    lp = rng.normal(0.0, 0.2, n)
    lu = 5.0 + beta * lp + 0.5 * season + rng.normal(0.0, 0.05, n)
    return pd.DataFrame(
        {
            "log_price": lp,
            "log_units": lu,
            "week_start": pd.date_range("2020-01-01", periods=n, freq="W"),
        }
    )


def test_deep_models_registered() -> None:
    assert {"lstm", "gru"} <= set(registry.all_keys())


@pytest.mark.parametrize("key", _DEEP_KEYS)
def test_deep_forecast_when_available(key: str) -> None:
    plugin = registry.get(key)
    if not plugin.is_available():
        pytest.skip("torch not installed")
    frame = _series()
    train, test = frame.iloc[:112], frame.iloc[112:]
    # small epochs to keep the test fast
    ctx = FitContext(ppg_id="P1", controls=[], test=test, hparams={"epochs": 40})
    result = plugin.fit(train, ctx)
    assert result.forecast is not None
    assert result.forecast.horizon == 28
    assert len(result.forecast.mean) == 28
    assert Capability.FORECAST in result.capabilities
    assert result.own_elasticity is None  # forecast-only
    assert "test_wape" in result.diagnostics
