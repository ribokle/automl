"""Prophet plugin (optional dependency: prophet). Forecast-only.

Prophet wants a (ds, y) frame; we synthesise a weekly calendar from row order
when no usable date column is present. log_price is added as an extra
regressor when available, but Prophet's additive regressors aren't reported as
a clean elasticity, so this plugin is forecast-only.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.metrics import wape_units
from core.models.result import Capability, ForecastBlock, ModelResult, ProblemType

TARGET = "log_units"
TIME_COL = "week_start"


def _calendar(n: int, frame: pd.DataFrame) -> pd.DatetimeIndex:
    if TIME_COL in frame.columns:
        ds = pd.to_datetime(frame[TIME_COL], errors="coerce")
        if ds.notna().all():
            return pd.DatetimeIndex(ds.to_numpy()[:n])
    return pd.date_range("2020-01-01", periods=n, freq="W")


@register
class ProphetPlugin(BaseModelPlugin):
    key = "prophet"
    family = "timeseries"
    problem_types = frozenset({ProblemType.FORECAST})
    capabilities = Capability.FORECAST | Capability.NEEDS_TIME_INDEX | Capability.HEAVY_DEP
    required_packages = ("prophet",)

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from prophet import Prophet

        sub = frame[[TARGET]].dropna().reset_index(drop=True)
        if len(sub) < 12:
            raise ValueError(f"too few rows for prophet ({len(sub)})")
        train = pd.DataFrame({"ds": _calendar(len(sub), frame), "y": sub[TARGET].to_numpy()})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(weekly_seasonality=False, daily_seasonality=False)
            model.fit(train)

        diagnostics: dict = {}
        forecast = None
        test = ctx.test
        if test is not None and len(test):
            tsub = test[[TARGET]].dropna().reset_index(drop=True)
            if len(tsub):
                future = pd.DataFrame(
                    {"ds": pd.date_range(train["ds"].iloc[-1], periods=len(tsub) + 1, freq="W")[1:]}
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fc = model.predict(future)
                mean_log = fc["yhat"].to_numpy()
                y_test = tsub[TARGET].to_numpy()
                diagnostics["test_wape"] = wape_units(y_test, mean_log)
                diagnostics["n_test"] = int(len(y_test))
                forecast = ForecastBlock(
                    horizon=len(mean_log),
                    index=[str(d.date()) for d in future["ds"]],
                    mean=[float(v) for v in np.exp(mean_log)],
                    lower=[float(v) for v in np.exp(fc["yhat_lower"].to_numpy())],
                    upper=[float(v) for v in np.exp(fc["yhat_upper"].to_numpy())],
                )

        return ModelResult(
            ppg_id=ctx.ppg_id,
            model=self.key,
            problem_type=ProblemType.FORECAST,
            own_elasticity=None,
            forecast=forecast,
            n_obs=int(len(sub)),
            diagnostics=diagnostics,
            capabilities=self.capabilities,
        )
