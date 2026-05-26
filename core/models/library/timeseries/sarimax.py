"""SARIMAX plugin (statsmodels) — seasonal ARIMA with exogenous regressors.

Falls back to a non-seasonal fit when the training window is shorter than two
seasonal cycles (seasonal terms are unidentifiable otherwise).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._statsmodels_ts import fit_exog_ts
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class SARIMAXPlugin(BaseModelPlugin):
    key = "sarimax"
    family = "timeseries"
    problem_types = frozenset({ProblemType.FORECAST, ProblemType.OWN_ELASTICITY})
    capabilities = Capability.FORECAST | Capability.SCALAR_ELASTICITY | Capability.NEEDS_TIME_INDEX
    required_packages = ("statsmodels",)

    def default_hparams(self) -> dict[str, Any]:
        return {"order": [1, 0, 0], "seasonal_order": [1, 0, 0], "m": 52}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        hp = self.resolve_hparams(ctx)
        order = tuple(hp["order"])
        m = int(hp["m"])
        s = list(hp["seasonal_order"])
        # Seasonal terms need >= 2 full cycles in the training window.
        seasonal_order = tuple(s + [m]) if len(frame) >= 2 * m else (0, 0, 0, 0)

        return fit_exog_ts(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            builder=lambda endog, exog: SARIMAX(
                endog, exog=exog, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ),
            test=ctx.test,
        )
