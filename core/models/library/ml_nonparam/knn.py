"""k-nearest-neighbours regression elasticity plugin (scikit-learn)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._tree_common import fit_tree_elasticity
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class KNNPlugin(BaseModelPlugin):
    key = "knn"
    family = "ml_nonparam"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def default_hparams(self) -> dict[str, Any]:
        return {"n_neighbors": 10}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.neighbors import KNeighborsRegressor

        hp = self.resolve_hparams(ctx)
        n = len(frame)
        k = min(int(hp["n_neighbors"]), max(2, n // 5))
        return fit_tree_elasticity(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: KNeighborsRegressor(n_neighbors=k),
            test=ctx.test,
        )
