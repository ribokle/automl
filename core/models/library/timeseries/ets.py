"""ETS / exponential-smoothing plugin (statsmodels). Forecast-only (no elasticity)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._statsmodels_ts import fit_ets
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class ETSPlugin(BaseModelPlugin):
    key = "ets"
    family = "timeseries"
    problem_types = frozenset({ProblemType.FORECAST})
    capabilities = Capability.FORECAST | Capability.NEEDS_TIME_INDEX
    required_packages = ("statsmodels",)

    def default_hparams(self) -> dict[str, Any]:
        return {"trend": "add"}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        trend = self.resolve_hparams(ctx)["trend"]
        return fit_ets(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            builder=lambda endog: ExponentialSmoothing(endog, trend=trend),
            test=ctx.test,
        )
