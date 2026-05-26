"""Rolling-origin (expanding-window) cross-validation for elasticity fitters.

The modelling agent reports a single 80/20 chronological hold-out WAPE.
That's a point estimate — it doesn't tell us whether the fit is stable
across time or whether the elasticity wanders fold-to-fold.

Rolling-origin CV partitions the chronologically-sorted frame into
``k`` folds with an expanding training window:

::

    fold 0:  train = [w0..wN]                       test = [wN+1..wN+h]
    fold 1:  train = [w0..wN+h]                     test = [wN+h+1..wN+2h]
    ...
    fold k:  train = [w0..wN+(k-1)h]                test = [wN+(k-1)h+1..wN+kh]

Each fold refits the winning OLS family and records the elasticity +
hold-out WAPE on the fold's test window. The aggregator then reports
mean/std/min/max across folds for both metrics, plus a sign-stability
percentage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import core.models.library  # noqa: F401 — register model plugins
from core.models.library import registry as model_registry
from core.models.library.base import FitContext
from core.models.result import to_elasticity_fit


@dataclass
class Fold:
    index: int
    train: pd.DataFrame
    test: pd.DataFrame


def build_folds(
    frame: pd.DataFrame,
    *,
    n_folds: int = 4,
    min_train_size: int = 20,
    time_col: str = "week_start",
) -> list[Fold]:
    """Expanding-window folds.

    The first fold trains on at least ``min_train_size`` rows; each
    subsequent fold extends the training window by one test-window's
    worth of rows. Returns an empty list when ``len(frame) <
    min_train_size + n_folds`` — there aren't enough rows to make
    ``n_folds`` distinct test windows.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    if time_col in frame.columns:
        ordered = frame.sort_values(time_col).reset_index(drop=True)
    else:
        ordered = frame.reset_index(drop=True)

    n = len(ordered)
    if n < min_train_size + n_folds:
        return []

    available_for_test = n - min_train_size
    test_size = max(1, available_for_test // n_folds)

    folds: list[Fold] = []
    for i in range(n_folds):
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size
        if i == n_folds - 1:
            test_end = n  # final fold absorbs any remainder
        if test_end <= train_end or train_end >= n:
            break
        train = ordered.iloc[:train_end].copy()
        test = ordered.iloc[train_end:test_end].copy()
        folds.append(Fold(index=i, train=train, test=test))
    return folds


def fit_one_fold(
    ppg_id: str,
    fold: Fold,
    controls: list[str],
    model_kind: str,
) -> dict:
    """Refit the winning model family on one fold, return elasticity + WAPE.

    Routes through the model-library registry so any registered, available
    plugin can be cross-validated, not just the original three families.
    """
    if not model_registry.has(model_kind):
        raise ValueError(f"unsupported model_kind={model_kind!r}")
    plugin = model_registry.get(model_kind)
    if not plugin.is_available():
        raise ValueError(f"model_kind={model_kind!r} dependency unavailable")
    ctx = FitContext(ppg_id=ppg_id, controls=controls, test=fold.test)
    fit = to_elasticity_fit(plugin.fit(fold.train, ctx))
    if fit is None:
        raise ValueError(f"model_kind={model_kind!r} produced no scalar elasticity")
    return {
        "fold": fold.index,
        "n_train": int(len(fold.train)),
        "n_test": int(len(fold.test)),
        "own_elasticity": float(fit.own_elasticity),
        "sign_ok": bool(fit.sign_ok),
        "r_squared": float(fit.r_squared),
        "train_wape": float(fit.diagnostics.get("train_wape", float("nan"))),
        "test_wape": float(fit.diagnostics.get("test_wape", float("nan"))),
        "test_residuals_log": _residuals_log(fit, fold.test),
    }


def _residuals_log(fit, test: pd.DataFrame) -> list[float]:
    """``observed_log - predicted_log`` for the fold's hold-out window.

    Returns an empty list when the fit didn't capture a comparable
    prediction (the OLS/LGBM fitters all stash ``diagnostics['test_residuals_log']``
    when a test frame is provided; this helper just normalises the type).
    """
    raw = fit.diagnostics.get("test_residuals_log") if fit.diagnostics else None
    if raw is None:
        return []
    arr = np.asarray(raw, dtype=float)
    return [float(v) for v in arr[np.isfinite(arr)]]
