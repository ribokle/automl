"""Semi-log elasticity fitter.

Fits ``log_units = α + β·price + Σ γᵢ·controlᵢ`` per PPG. β alone is not an
elasticity — it's a semi-elasticity (%Δunits per absolute Δprice). We
convert to a comparable own-price elasticity by evaluating at the mean
price: ``ε = β · mean(price)``. Standard error and p-value on the elasticity
are scaled the same way (linear function of β).

Used as the sign-retry fallback for log-log: log-log occasionally returns a
positive coefficient on noisy panels (multicollinearity with controls,
limited price variation), and semi-log gives a different functional form
without changing the units of analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.models.base import ElasticityFit


TARGET = "log_units"
LOG_PRICE = "log_price"
PRICE = "price"


def fit_semilog(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
) -> ElasticityFit:
    """Fit semi-log OLS for one PPG.

    Reconstructs raw price from ``log_price`` when ``price`` itself is not in
    the feature frame (engineered features only carry the log-transform).
    """
    if TARGET not in frame.columns or LOG_PRICE not in frame.columns:
        raise ValueError(f"frame missing {TARGET} or {LOG_PRICE}")
    work = frame.copy()
    if PRICE not in work.columns:
        work[PRICE] = np.exp(work[LOG_PRICE].astype(float))

    usable = [c for c in controls if c in work.columns and c not in (PRICE, LOG_PRICE, TARGET)]
    usable = [c for c in usable if work[c].nunique(dropna=True) > 1]

    cols = [PRICE] + usable
    sub = work[[TARGET, *cols]].dropna()
    y = sub[TARGET].astype(float).to_numpy()
    X = sm.add_constant(sub[cols].astype(float).to_numpy(), has_constant="add")
    model = sm.OLS(y, X).fit()

    own_idx = 1
    beta = float(model.params[own_idx])
    beta_se = float(model.bse[own_idx])
    p_mean = float(np.mean(sub[PRICE]))

    elasticity = beta * p_mean
    elasticity_se = beta_se * p_mean

    coefs = dict(zip(["const", *cols], (float(v) for v in model.params)))

    return ElasticityFit(
        ppg_id=ppg_id,
        model="semilog_ols",
        own_elasticity=elasticity,
        std_err=elasticity_se,
        p_value=float(model.pvalues[own_idx]),
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
        controls=usable,
        coefficients=coefs,
        diagnostics={
            "aic": float(model.aic),
            "bic": float(model.bic),
            "adj_r_squared": float(model.rsquared_adj),
            "beta_price": beta,
            "price_mean": p_mean,
        },
    )
