"""Quantile-regression log-log plugin — median (or other quantile) demand."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._sklearn_linear import fit_sklearn_loglog
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class QuantilePlugin(BaseModelPlugin):
    key = "quantile"
    family = "robust_quantile"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def default_hparams(self) -> dict[str, Any]:
        return {"quantile": 0.5, "alpha": 0.0, "solver": "highs"}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.linear_model import QuantileRegressor

        hp = self.resolve_hparams(ctx)
        return fit_sklearn_loglog(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: QuantileRegressor(
                quantile=float(hp["quantile"]),
                alpha=float(hp["alpha"]),
                solver=str(hp["solver"]),
            ),
            test=ctx.test,
        )
