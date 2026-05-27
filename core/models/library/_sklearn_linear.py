"""Shared log-log fitting helper for sklearn linear estimators.

Used by the regularized family (ridge / lasso / elasticnet). NOT a model
module — it registers nothing — so model modules may import it without
violating the no-cross-import rule. Produces a ``ModelResult`` whose
``coefficients`` dict (``const`` + log_price + controls) is OLS-compatible, so
``core.models.predictor.build_predictor`` can drive downstream stages with no
special-casing.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from core.models.metrics import wape_units
from core.models.result import Capability, ModelResult, ProblemType

LOG_PRICE = "log_price"
TARGET = "log_units"


def _usable_controls(frame: pd.DataFrame, controls: list[str]) -> list[str]:
    cols = [c for c in controls if c in frame.columns and c not in (LOG_PRICE, TARGET)]
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


def fit_sklearn_loglog(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    *,
    model_name: str,
    estimator_factory: Callable[[], Any],
    test: pd.DataFrame | None = None,
    coef_fn: Callable[[Any], Any] | None = None,
    intercept_fn: Callable[[Any], Any] | None = None,
) -> ModelResult:
    if LOG_PRICE not in frame.columns or TARGET not in frame.columns:
        raise ValueError(f"frame missing {LOG_PRICE} or {TARGET}")

    usable = _usable_controls(frame, controls)
    cols = [LOG_PRICE] + usable

    train = frame[[TARGET, *cols]].dropna()
    X = train[cols].astype(float).to_numpy()
    y = train[TARGET].astype(float).to_numpy()

    estimator = estimator_factory()
    estimator.fit(X, y)

    coef = np.asarray(coef_fn(estimator) if coef_fn else estimator.coef_, dtype=float).ravel()
    intercept = float(intercept_fn(estimator) if intercept_fn else estimator.intercept_)

    own_beta = float(coef[0])
    coefs: dict[str, float] = {"const": intercept}
    coefs.update({c: float(b) for c, b in zip(cols, coef, strict=False)})

    train_pred = estimator.predict(X)
    diagnostics: dict[str, Any] = {
        "train_wape": wape_units(y, train_pred),
        "alpha": float(getattr(estimator, "alpha_", getattr(estimator, "alpha", float("nan")))),
        "n_nonzero_coef": int(np.count_nonzero(coef)),
    }
    r_squared = float(estimator.score(X, y))

    if test is not None and len(test):
        sub = test[[TARGET, *cols]].dropna()
        if len(sub):
            X_test = sub[cols].astype(float).to_numpy()
            y_test = sub[TARGET].astype(float).to_numpy()
            test_pred = estimator.predict(X_test)
            diagnostics["test_wape"] = wape_units(y_test, test_pred)
            diagnostics["n_test"] = int(len(y_test))
            diagnostics["test_residuals_log"] = (
                np.asarray(y_test, dtype=float) - np.asarray(test_pred, dtype=float)
            ).tolist()

    return ModelResult(
        ppg_id=ppg_id,
        model=model_name,
        problem_type=ProblemType.OWN_ELASTICITY,
        own_elasticity=own_beta,
        std_err=None,
        p_value=None,
        r_squared=r_squared,
        n_obs=int(len(train)),
        controls=usable,
        coefficients=coefs,
        diagnostics=diagnostics,
        capabilities=Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS,
    )
