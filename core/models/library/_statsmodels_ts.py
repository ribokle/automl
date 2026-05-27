"""Shared statsmodels time-series fitting helpers.

Two builders share one ``ModelResult`` shape:

- ``fit_exog_ts`` (SARIMAX / UnobservedComponents): regresses ``log_units`` on
  lagged dynamics + an exogenous design that leads with ``log_price``, so the
  exog coefficient on ``log_price`` is reported as the own-price elasticity.
  Also returns a hold-out ``ForecastBlock``.
- ``fit_ets`` (Exponential Smoothing / Holt-Winters): endog-only smoothing, so
  it returns a forecast but no elasticity.

NOT a model module (registers nothing); model modules may import it. Time-series
winners are forecast-capable but NOT price-sweepable, so they are intentionally
left out of ``predictor.PREDICTABLE_MODELS`` — downstream optimisation skips
them, which is correct for the FORECAST problem.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from core.models.metrics import wape_units
from core.models.result import Capability, ForecastBlock, ModelResult, ProblemType

LOG_PRICE = "log_price"
TARGET = "log_units"
TIME_COL = "week_start"


def _usable_controls(frame: pd.DataFrame, controls: list[str]) -> list[str]:
    cols = [c for c in controls if c in frame.columns and c not in (LOG_PRICE, TARGET)]
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


def _index_labels(frame: pd.DataFrame, n: int) -> list[str]:
    if TIME_COL in frame.columns and len(frame):
        return [str(v) for v in frame[TIME_COL].astype(str).tolist()[:n]]
    return [str(i) for i in range(n)]


def _find_coef(series: pd.Series, name: str) -> float | None:
    if name in series.index:
        return float(series[name])
    for idx in series.index:
        if str(idx) == name or str(idx).endswith("." + name) or str(idx).endswith(name):
            return float(series[idx])
    return None


def _forecast_block(
    mean_log: np.ndarray,
    test: pd.DataFrame,
    lower_log: np.ndarray | None,
    upper_log: np.ndarray | None,
) -> ForecastBlock:
    n = len(mean_log)
    return ForecastBlock(
        horizon=n,
        index=_index_labels(test, n),
        mean=[float(v) for v in np.exp(mean_log)],
        lower=None if lower_log is None else [float(v) for v in np.exp(lower_log)],
        upper=None if upper_log is None else [float(v) for v in np.exp(upper_log)],
    )


def fit_exog_ts(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    *,
    model_name: str,
    builder: Callable[[np.ndarray, pd.DataFrame], Any],
    test: pd.DataFrame | None = None,
) -> ModelResult:
    """Fit an exogenous-regressor TS model; elasticity = coef on log_price."""
    if TARGET not in frame.columns or LOG_PRICE not in frame.columns:
        raise ValueError(f"frame missing {TARGET} or {LOG_PRICE}")
    usable = _usable_controls(frame, controls)
    exog_cols = [LOG_PRICE] + usable
    sub = frame[[TARGET, *exog_cols]].dropna()
    if len(sub) < 12:
        raise ValueError(f"too few rows for time-series fit ({len(sub)})")
    endog = sub[TARGET].astype(float).to_numpy()
    exog = sub[exog_cols].astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = builder(endog, exog).fit(disp=False)

    elasticity = _find_coef(res.params, LOG_PRICE)
    std_err = _find_coef(res.bse, LOG_PRICE)

    diagnostics: dict[str, Any] = {"aic": float(getattr(res, "aic", float("nan")))}
    forecast: ForecastBlock | None = None
    if test is not None and len(test):
        tsub = test[[TARGET, *exog_cols]].dropna()
        if len(tsub):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = res.get_forecast(steps=len(tsub), exog=tsub[exog_cols].astype(float))
            mean_log = np.asarray(fc.predicted_mean, dtype=float)
            try:
                ci = np.asarray(fc.conf_int(), dtype=float)
                lower_log, upper_log = ci[:, 0], ci[:, 1]
            except Exception:  # noqa: BLE001
                lower_log = upper_log = None
            y_test = tsub[TARGET].astype(float).to_numpy()
            diagnostics["test_wape"] = wape_units(y_test, mean_log)
            diagnostics["n_test"] = int(len(y_test))
            forecast = _forecast_block(mean_log, tsub, lower_log, upper_log)

    caps = Capability.FORECAST
    if elasticity is not None:
        caps |= Capability.SCALAR_ELASTICITY
    return ModelResult(
        ppg_id=ppg_id,
        model=model_name,
        problem_type=ProblemType.FORECAST,
        own_elasticity=elasticity,
        std_err=std_err,
        p_value=None,
        forecast=forecast,
        r_squared=None,
        n_obs=int(len(sub)),
        controls=usable,
        coefficients={},
        diagnostics=diagnostics,
        capabilities=caps,
    )


def fit_ets(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    *,
    model_name: str,
    builder: Callable[[np.ndarray], Any],
    test: pd.DataFrame | None = None,
) -> ModelResult:
    """Fit an endog-only smoothing model; forecast but no elasticity."""
    if TARGET not in frame.columns:
        raise ValueError(f"frame missing {TARGET}")
    sub = frame[[TARGET]].dropna()
    if len(sub) < 12:
        raise ValueError(f"too few rows for time-series fit ({len(sub)})")
    endog = sub[TARGET].astype(float).to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = builder(endog).fit()

    diagnostics: dict[str, Any] = {"aic": float(getattr(res, "aic", float("nan")))}
    forecast: ForecastBlock | None = None
    if test is not None and len(test):
        tsub = test[[TARGET]].dropna()
        if len(tsub):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean_log = np.asarray(res.forecast(len(tsub)), dtype=float)
            y_test = tsub[TARGET].astype(float).to_numpy()
            diagnostics["test_wape"] = wape_units(y_test, mean_log)
            diagnostics["n_test"] = int(len(y_test))
            forecast = _forecast_block(mean_log, tsub, None, None)

    return ModelResult(
        ppg_id=ppg_id,
        model=model_name,
        problem_type=ProblemType.FORECAST,
        own_elasticity=None,
        forecast=forecast,
        n_obs=int(len(sub)),
        controls=[],
        coefficients={},
        diagnostics=diagnostics,
        capabilities=Capability.FORECAST,
    )
