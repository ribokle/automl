"""Lasso log-log plugin — CV-selected L1 shrinkage + feature selection."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._sklearn_linear import fit_sklearn_loglog
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class LassoPlugin(BaseModelPlugin):
    key = "lasso"
    family = "regularized"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def default_hparams(self) -> dict[str, Any]:
        return {"cv": 3, "max_iter": 5000}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.linear_model import LassoCV

        hp = self.resolve_hparams(ctx)
        return fit_sklearn_loglog(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: LassoCV(
                cv=int(hp["cv"]),
                max_iter=int(hp["max_iter"]),
                random_state=int(ctx.rng_seed),
            ),
            test=ctx.test,
        )
