"""State-space plugin (statsmodels UnobservedComponents) with exog regressors.

Local-level + trend structural model; the log_price exog coefficient is the
own-price elasticity.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._statsmodels_ts import fit_exog_ts
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class StateSpacePlugin(BaseModelPlugin):
    key = "state_space"
    family = "timeseries"
    problem_types = frozenset({ProblemType.FORECAST, ProblemType.OWN_ELASTICITY})
    capabilities = Capability.FORECAST | Capability.SCALAR_ELASTICITY | Capability.NEEDS_TIME_INDEX
    required_packages = ("statsmodels",)

    def default_hparams(self) -> dict[str, Any]:
        return {"level": "local linear trend"}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        level = str(self.resolve_hparams(ctx)["level"])
        return fit_exog_ts(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            builder=lambda endog, exog: UnobservedComponents(
                endog, level=level, exog=exog
            ),
            test=ctx.test,
        )
