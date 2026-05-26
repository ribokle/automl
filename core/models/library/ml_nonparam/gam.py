"""Generalized Additive Model elasticity plugin (optional dependency: pygam)."""
from __future__ import annotations

import pandas as pd

from core.models.library._tree_common import fit_tree_elasticity
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class GAMPlugin(BaseModelPlugin):
    key = "gam"
    family = "ml_nonparam"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = (
        Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS | Capability.HEAVY_DEP
    )
    required_packages = ("pygam",)

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from pygam import LinearGAM

        return fit_tree_elasticity(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=LinearGAM,
            test=ctx.test,
        )
