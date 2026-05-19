"""LightGBM elasticity fitter.

LightGBM doesn't expose an elasticity coefficient the way OLS does — the
relationship between log_price and log_units is captured by a tree
ensemble. We recover an average own-price elasticity numerically: at each
training row, predict at the observed log_price and at log_price + δ (a 1%
price bump, i.e. δ = log(1.01) ≈ 0.00995), then average the slope
``(ŷ_high - ŷ_base) / δ`` across rows. The signed average is the
local-elasticity estimate the agent reports.

R² and hold-out WAPE are reported on the same scale as the OLS fitters
(R² on log_units, WAPE on raw units) so the agent can rank candidates on
equal footing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from core.models.base import ElasticityFit
from core.models.metrics import wape_units
from core.models.shap_attribution import lightgbm_shap_summary


TARGET = "log_units"
LOG_PRICE = "log_price"
_DELTA = np.log(1.01)


def _design_xy(frame: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    sub = frame[[TARGET, *cols]].dropna()
    X = sub[cols].astype(float).copy()
    y = sub[TARGET].astype(float).to_numpy()
    return X, y


def _average_elasticity(model: LGBMRegressor, X: pd.DataFrame) -> tuple[float, float]:
    """Numerical own-price elasticity averaged over rows.

    Returns ``(mean_elasticity, std_elasticity)`` — the spread is reported as
    a pseudo standard error so the UI can show a band, but it isn't a true
    sampling SE.
    """
    pred_base = model.predict(X)
    bumped = X.copy()
    bumped[LOG_PRICE] = bumped[LOG_PRICE] + _DELTA
    pred_high = model.predict(bumped)
    point = (pred_high - pred_base) / _DELTA
    return float(np.mean(point)), float(np.std(point))


def fit_lightgbm(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    test: pd.DataFrame | None = None,
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 15,
    min_child_samples: int = 5,
    random_state: int = 0,
) -> ElasticityFit:
    """Fit LightGBM and recover the average own-price elasticity numerically."""
    if TARGET not in frame.columns or LOG_PRICE not in frame.columns:
        raise ValueError(f"frame missing {TARGET} or {LOG_PRICE}")

    usable = [c for c in controls if c in frame.columns and c not in (LOG_PRICE, TARGET)]
    usable = [c for c in usable if frame[c].nunique(dropna=True) > 1]
    cols = [LOG_PRICE] + usable

    X_train, y_train = _design_xy(frame, cols)
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    elasticity, elasticity_sd = _average_elasticity(model, X_train)

    train_pred = model.predict(X_train)
    ss_res = float(np.sum((y_train - train_pred) ** 2))
    ss_tot = float(np.sum((y_train - np.mean(y_train)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    diagnostics: dict = {
        "train_wape": wape_units(y_train, train_pred),
        "delta_log_price": _DELTA,
        "elasticity_sd_across_rows": elasticity_sd,
        "feature_importances": {
            c: float(v) for c, v in zip(cols, model.feature_importances_)
        },
        "shap": lightgbm_shap_summary(model, X_train),
    }
    if test is not None and len(test):
        X_test, y_test = _design_xy(test, cols)
        if len(y_test):
            test_pred = model.predict(X_test)
            diagnostics["test_wape"] = wape_units(y_test, test_pred)
            diagnostics["n_test"] = int(len(y_test))

    return ElasticityFit(
        ppg_id=ppg_id,
        model="lightgbm",
        own_elasticity=elasticity,
        std_err=elasticity_sd,
        p_value=float("nan"),
        r_squared=float(r_squared),
        n_obs=int(len(X_train)),
        controls=usable,
        coefficients={},
        diagnostics=diagnostics,
    )
