"""Holt-Winters plugin (statsmodels) — trend + seasonal smoothing.

Adds a seasonal component when the training window covers >= 2 cycles; falls
back to trend-only otherwise. Forecast-only (no elasticity).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._statsmodels_ts import fit_ets
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class HoltWintersPlugin(BaseModelPlugin):
    key = "holt_winters"
    family = "timeseries"
    problem_types = frozenset({ProblemType.FORECAST})
    capabilities = Capability.FORECAST | Capability.NEEDS_TIME_INDEX
    required_packages = ("statsmodels",)

    def default_hparams(self) -> dict[str, Any]:
        return {"trend": "add", "seasonal": "add", "m": 52}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        hp = self.resolve_hparams(ctx)
        m = int(hp["m"])
        seasonal = hp["seasonal"] if len(frame) >= 2 * m else None
        seasonal_periods = m if seasonal else None
        return fit_ets(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            builder=lambda endog: ExponentialSmoothing(
                endog, trend=hp["trend"], seasonal=seasonal, seasonal_periods=seasonal_periods
            ),
            test=ctx.test,
        )
