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


PRICE_COL = "log_price"
TARGET = "log_units"


def fit_loglog(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
) -> ElasticityFit:
    """Fit a log-log OLS for one PPG.

    ``frame`` must contain ``log_units`` + ``log_price`` + every column listed
    in ``controls``. Controls are filtered to those that actually vary on the
    slice (statsmodels chokes on perfectly collinear or constant regressors).
    """
    if PRICE_COL not in frame.columns or TARGET not in frame.columns:
        raise ValueError(f"frame missing {PRICE_COL} or {TARGET}")
    usable = [c for c in controls if c in frame.columns and c != PRICE_COL and c != TARGET]
    usable = [c for c in usable if frame[c].nunique(dropna=True) > 1]

    cols = [PRICE_COL] + usable
    sub = frame[[TARGET, *cols]].dropna()
    y = sub[TARGET].astype(float).to_numpy()
    X = sm.add_constant(sub[cols].astype(float).to_numpy(), has_constant="add")
    model = sm.OLS(y, X).fit()

    coefs = dict(zip(["const", *cols], (float(v) for v in model.params)))
    own_idx = 1
    own_beta = float(model.params[own_idx])
    own_se = float(model.bse[own_idx])
    own_p = float(model.pvalues[own_idx])

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
        diagnostics={
            "aic": float(model.aic),
            "bic": float(model.bic),
            "adj_r_squared": float(model.rsquared_adj),
            "log_price_mean": float(np.mean(sub[PRICE_COL])),
        },
    )
