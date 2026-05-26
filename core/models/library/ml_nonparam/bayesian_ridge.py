"""Bayesian ridge log-log plugin — Gaussian prior on coefficients (scikit-learn)."""
from __future__ import annotations

import pandas as pd

from core.models.library._sklearn_linear import fit_sklearn_loglog
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class BayesianRidgePlugin(BaseModelPlugin):
    key = "bayesian_ridge"
    family = "ml_nonparam"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.linear_model import BayesianRidge

        return fit_sklearn_loglog(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=BayesianRidge,
            test=ctx.test,
        )
