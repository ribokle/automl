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
from core.models.metrics import wape_units


TARGET = "log_units"
LOG_PRICE = "log_price"
PRICE = "price"


def _ensure_price(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if PRICE not in work.columns:
        work[PRICE] = np.exp(work[LOG_PRICE].astype(float))
    return work


def _design(
    frame: pd.DataFrame, cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    sub = frame[[TARGET, *cols]].dropna()
    y = sub[TARGET].astype(float).to_numpy()
    X = sm.add_constant(sub[cols].astype(float).to_numpy(), has_constant="add")
    return y, X


def fit_semilog(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    test: pd.DataFrame | None = None,
) -> ElasticityFit:
    """Fit semi-log OLS for one PPG.

    Reconstructs raw price from ``log_price`` when ``price`` itself is not in
    the feature frame (engineered features only carry the log-transform).
    Hold-out WAPE is added to diagnostics when ``test`` is supplied.
    """
    if TARGET not in frame.columns or LOG_PRICE not in frame.columns:
        raise ValueError(f"frame missing {TARGET} or {LOG_PRICE}")
    train = _ensure_price(frame)

    usable = [
        c for c in controls if c in train.columns and c not in (PRICE, LOG_PRICE, TARGET)
    ]
    usable = [c for c in usable if train[c].nunique(dropna=True) > 1]

    cols = [PRICE] + usable
    y_train, X_train = _design(train, cols)
    model = sm.OLS(y_train, X_train).fit()

    own_idx = 1
    beta = float(model.params[own_idx])
    beta_se = float(model.bse[own_idx])
    p_mean = float(np.mean(train[PRICE]))

    elasticity = beta * p_mean
    elasticity_se = beta_se * p_mean

    coefs = dict(zip(["const", *cols], (float(v) for v in model.params)))

    diagnostics: dict = {
        "aic": float(model.aic),
        "bic": float(model.bic),
        "adj_r_squared": float(model.rsquared_adj),
        "beta_price": beta,
        "price_mean": p_mean,
        "train_wape": wape_units(y_train, model.predict(X_train)),
    }
    if test is not None and len(test):
        test_p = _ensure_price(test)
        y_test, X_test = _design(test_p, cols)
        if len(y_test):
            diagnostics["test_wape"] = wape_units(y_test, model.predict(X_test))
            diagnostics["n_test"] = int(len(y_test))

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
        diagnostics=diagnostics,
    )
