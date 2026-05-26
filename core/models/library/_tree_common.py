"""Shared tree-ensemble elasticity recovery.

Tree models don't expose an elasticity coefficient; we recover an average
own-price elasticity numerically by bumping ``log_price`` by 1% at every row
and averaging the predicted slope — the same approach as the LightGBM fitter.
NOT a model module (registers nothing), so model modules may import it.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from core.models.metrics import wape_units
from core.models.result import Capability, ModelResult, ProblemType

LOG_PRICE = "log_price"
TARGET = "log_units"
_DELTA = float(np.log(1.01))


def _usable_controls(frame: pd.DataFrame, controls: list[str]) -> list[str]:
    cols = [c for c in controls if c in frame.columns and c not in (LOG_PRICE, TARGET)]
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


def fit_tree_elasticity(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    *,
    model_name: str,
    estimator_factory: Callable[[], Any],
    test: pd.DataFrame | None = None,
) -> ModelResult:
    if LOG_PRICE not in frame.columns or TARGET not in frame.columns:
        raise ValueError(f"frame missing {LOG_PRICE} or {TARGET}")

    usable = _usable_controls(frame, controls)
    cols = [LOG_PRICE] + usable
    sub = frame[[TARGET, *cols]].dropna()
    X = sub[cols].astype(float).copy()
    y = sub[TARGET].astype(float).to_numpy()

    estimator = estimator_factory()
    estimator.fit(X, y)

    base = np.asarray(estimator.predict(X), dtype=float)
    bumped = X.copy()
    bumped[LOG_PRICE] = bumped[LOG_PRICE] + _DELTA
    high = np.asarray(estimator.predict(bumped), dtype=float)
    point = (high - base) / _DELTA
    elasticity = float(np.mean(point))
    elasticity_sd = float(np.std(point))

    ss_res = float(np.sum((y - base) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    diagnostics: dict[str, Any] = {
        "train_wape": wape_units(y, base),
        "delta_log_price": _DELTA,
        "elasticity_sd_across_rows": elasticity_sd,
        "predictor_kind": "tree",
    }
    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None:
        diagnostics["feature_importances"] = {
            c: float(v) for c, v in zip(cols, importances)
        }

    if test is not None and len(test):
        tsub = test[[TARGET, *cols]].dropna()
        if len(tsub):
            X_test = tsub[cols].astype(float)
            y_test = tsub[TARGET].astype(float).to_numpy()
            test_pred = np.asarray(estimator.predict(X_test), dtype=float)
            diagnostics["test_wape"] = wape_units(y_test, test_pred)
            diagnostics["n_test"] = int(len(y_test))
            diagnostics["test_residuals_log"] = (y_test - test_pred).tolist()

    return ModelResult(
        ppg_id=ppg_id,
        model=model_name,
        problem_type=ProblemType.OWN_ELASTICITY,
        own_elasticity=elasticity,
        std_err=elasticity_sd,
        p_value=None,
        r_squared=r_squared,
        n_obs=int(len(sub)),
        controls=usable,
        coefficients={},
        diagnostics=diagnostics,
        capabilities=Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS,
    )
