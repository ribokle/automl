"""Theil-Sen robust log-log plugin — median-of-slopes estimator."""
from __future__ import annotations

import pandas as pd

from core.models.library._sklearn_linear import fit_sklearn_loglog
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class TheilSenPlugin(BaseModelPlugin):
    key = "theil_sen"
    family = "robust_quantile"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.linear_model import TheilSenRegressor

        return fit_sklearn_loglog(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: TheilSenRegressor(random_state=int(ctx.rng_seed)),
            test=ctx.test,
        )
