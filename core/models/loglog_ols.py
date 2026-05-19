"""Log-log OLS elasticity fitter.

Fits ``log_units = α + β·log_price + Σ γᵢ·controlᵢ`` per PPG via statsmodels
OLS. β is the own-price elasticity directly. Returns an ``ElasticityFit``
so the modelling agent can compare across PPGs and against semi-log
alternatives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.models.base import ElasticityFit
from core.models.metrics import wape_units
from core.models.shap_attribution import ols_shap_summary


PRICE_COL = "log_price"
TARGET = "log_units"


def _design(frame: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sub = frame[[TARGET, *cols]].dropna()
    y = sub[TARGET].astype(float).to_numpy()
    X = sm.add_constant(sub[cols].astype(float).to_numpy(), has_constant="add")
    return y, X


def fit_loglog(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    test: pd.DataFrame | None = None,
) -> ElasticityFit:
    """Fit a log-log OLS for one PPG.

    ``frame`` must contain ``log_units`` + ``log_price`` + every column listed
    in ``controls``. Controls are filtered to those that actually vary on the
    slice (statsmodels chokes on perfectly collinear or constant regressors).
    When ``test`` is supplied, hold-out WAPE on raw units is added to the
    diagnostics dict so models can be ranked on equal footing.
    """
    if PRICE_COL not in frame.columns or TARGET not in frame.columns:
        raise ValueError(f"frame missing {PRICE_COL} or {TARGET}")
    usable = [c for c in controls if c in frame.columns and c != PRICE_COL and c != TARGET]
    usable = [c for c in usable if frame[c].nunique(dropna=True) > 1]

    cols = [PRICE_COL] + usable
    y_train, X_train = _design(frame, cols)
    model = sm.OLS(y_train, X_train).fit()

    coefs = dict(zip(["const", *cols], (float(v) for v in model.params)))
    own_idx = 1
    own_beta = float(model.params[own_idx])
    own_se = float(model.bse[own_idx])
    own_p = float(model.pvalues[own_idx])

    diagnostics: dict = {
        "aic": float(model.aic),
        "bic": float(model.bic),
        "adj_r_squared": float(model.rsquared_adj),
        "log_price_mean": float(np.mean(frame[PRICE_COL])),
    }
    train_pred = model.predict(X_train)
    diagnostics["train_wape"] = wape_units(y_train, train_pred)
    diagnostics["shap"] = ols_shap_summary(coefs, frame, cols)
    if test is not None and len(test):
        y_test, X_test = _design(test, cols)
        if len(y_test):
            test_pred = model.predict(X_test)
            diagnostics["test_wape"] = wape_units(y_test, test_pred)
            diagnostics["n_test"] = int(len(y_test))

    return ElasticityFit(
        ppg_id=ppg_id,
        model="loglog_ols",
        own_elasticity=own_beta,
        std_err=own_se,
        p_value=own_p,
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
        controls=usable,
        coefficients=coefs,
        diagnostics=diagnostics,
    )
