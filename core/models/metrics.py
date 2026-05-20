"""Shared metrics + train/test splitting for the elasticity fitters.

WAPE is reported on raw units (not log units) so the number is directly
comparable across model families and aligned with how stakeholders read
hold-out accuracy. Predictions arrive on the log scale and are
exponentiated before the WAPE computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wape_units(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """WAPE on raw units; both inputs are predictions/observations on log(units)."""
    y_true = np.exp(np.asarray(y_true_log, dtype=float))
    y_pred = np.exp(np.asarray(y_pred_log, dtype=float))
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def chronological_split(
    frame: pd.DataFrame, test_ratio: float = 0.2, time_col: str = "week_start"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `test_ratio` of rows (by `time_col`) form the hold-out.

    Falls back to row-order split when the time column is missing — keeps the
    helper usable from synthetic mini-fixtures that don't carry calendars.
    """
    if time_col in frame.columns:
        ordered = frame.sort_values(time_col).reset_index(drop=True)
    else:
        ordered = frame.reset_index(drop=True)
    n = len(ordered)
    n_test = max(1, int(round(n * test_ratio)))
    n_train = max(1, n - n_test)
    return ordered.iloc[:n_train].copy(), ordered.iloc[n_train:].copy()
