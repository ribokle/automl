"""Gaussian-process regression elasticity plugin (scikit-learn)."""
from __future__ import annotations

import pandas as pd

from core.models.library._tree_common import fit_tree_elasticity
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class GaussianProcessPlugin(BaseModelPlugin):
    key = "gaussian_process"
    family = "ml_nonparam"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        return fit_tree_elasticity(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: GaussianProcessRegressor(
                kernel=RBF() + WhiteKernel(),
                normalize_y=True,
                random_state=int(ctx.rng_seed),
            ),
            test=ctx.test,
        )
