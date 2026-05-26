"""ARIMAX plugin (statsmodels SARIMAX, no seasonal term).

Forecasts log_units with ARIMA dynamics + exogenous regressors led by
log_price; the log_price exog coefficient is reported as the own-price
elasticity.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._statsmodels_ts import fit_exog_ts
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class ARIMAXPlugin(BaseModelPlugin):
    key = "arimax"
    family = "timeseries"
    problem_types = frozenset({ProblemType.FORECAST, ProblemType.OWN_ELASTICITY})
    capabilities = Capability.FORECAST | Capability.SCALAR_ELASTICITY | Capability.NEEDS_TIME_INDEX
    required_packages = ("statsmodels",)

    def default_hparams(self) -> dict[str, Any]:
        return {"order": [1, 0, 1]}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        order = tuple(self.resolve_hparams(ctx)["order"])
        return fit_exog_ts(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            builder=lambda endog, exog: SARIMAX(
                endog, exog=exog, order=order, enforce_stationarity=False,
                enforce_invertibility=False,
            ),
            test=ctx.test,
        )
