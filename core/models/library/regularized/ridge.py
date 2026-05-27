"""Ridge log-log plugin — CV-selected L2 shrinkage via scikit-learn."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.models.library._sklearn_linear import fit_sklearn_loglog
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class RidgePlugin(BaseModelPlugin):
    key = "ridge"
    family = "regularized"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def default_hparams(self) -> dict[str, Any]:
        return {"alphas": [0.01, 0.1, 1.0, 10.0]}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.linear_model import RidgeCV

        hp = self.resolve_hparams(ctx)
        alphas = np.asarray(hp["alphas"], dtype=float)
        return fit_sklearn_loglog(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: RidgeCV(alphas=alphas),
            test=ctx.test,
        )
