"""Huber robust log-log plugin — down-weights leverage points (promo weeks)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._sklearn_linear import fit_sklearn_loglog
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class HuberPlugin(BaseModelPlugin):
    key = "huber"
    family = "robust_quantile"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def default_hparams(self) -> dict[str, Any]:
        return {"epsilon": 1.35, "alpha": 0.0001, "max_iter": 200}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.linear_model import HuberRegressor

        hp = self.resolve_hparams(ctx)
        return fit_sklearn_loglog(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: HuberRegressor(
                epsilon=float(hp["epsilon"]),
                alpha=float(hp["alpha"]),
                max_iter=int(hp["max_iter"]),
            ),
            test=ctx.test,
        )
