"""Data-diagnostics profile that drives router model selection.

Computed once per cell from the feature slice before routing. Pure function of
the frame + context; no config thresholds baked in (the router compares the
profile against its own configurable thresholds).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

LOG_PRICE = "log_price"
TARGET = "log_units"
TIME_COL = "week_start"
_CROSS_PRICE_HINTS = ("competitor", "cross_", "_other", "rival")


@dataclass
class DataProfile:
    n_obs: int
    n_train: int
    n_test: int
    log_price_std: float
    price_cv: float
    is_panel: bool
    n_entities: int
    has_cross_price_cols: bool
    time_length: int
    has_regular_time_index: bool
    seasonality_detected: bool
    n_seasons: int
    target_type: str
    n_controls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _has_cross_price(columns: list[str]) -> bool:
    lowered = [c.lower() for c in columns]
    if "log_competitor_price" in lowered:
        return True
    return any(hint in col for col in lowered for hint in _CROSS_PRICE_HINTS)


def _detect_seasonality(
    frame: pd.DataFrame, min_length: int
) -> tuple[bool, int]:
    if TARGET not in frame.columns:
        return False, 0
    series = frame[TARGET].astype(float).dropna()
    n = len(series)
    if n < min_length * 2:
        return False, 0
    values = series.to_numpy()
    values = values - values.mean()
    denom = float(np.dot(values, values))
    if denom == 0.0:
        return False, 0
    lag = min_length
    autocorr = float(np.dot(values[:-lag], values[lag:]) / denom)
    return autocorr > 0.3, int(n // min_length)


def profile(
    frame: pd.DataFrame,
    controls: list[str],
    *,
    grain: str = "ppg_week",
    test_ratio: float = 0.2,
    seasonality_min_length: int = 52,
) -> DataProfile:
    n = int(len(frame))
    n_test = max(1, int(round(n * test_ratio))) if n else 0
    n_train = max(1, n - n_test) if n else 0

    if LOG_PRICE in frame.columns and n:
        log_price = frame[LOG_PRICE].astype(float)
        log_price_std = float(log_price.std(ddof=0))
        price = np.exp(log_price.to_numpy())
        mean_price = float(np.mean(price)) if len(price) else 0.0
        price_cv = float(np.std(price) / mean_price) if mean_price else 0.0
    else:
        log_price_std = 0.0
        price_cv = 0.0

    is_panel = "grain_unit" in frame.columns
    n_entities = int(frame["grain_unit"].nunique()) if is_panel else 1
    has_time = TIME_COL in frame.columns
    seasonal, n_seasons = _detect_seasonality(frame, seasonality_min_length)

    return DataProfile(
        n_obs=n,
        n_train=n_train,
        n_test=n_test,
        log_price_std=log_price_std,
        price_cv=price_cv,
        is_panel=is_panel,
        n_entities=n_entities,
        has_cross_price_cols=_has_cross_price(list(frame.columns)),
        time_length=n,
        has_regular_time_index=has_time,
        seasonality_detected=seasonal,
        n_seasons=n_seasons,
        target_type="continuous_log_units",
        n_controls=len([c for c in controls if c in frame.columns]),
    )
