"""Support-vector regression elasticity plugin (scikit-learn)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._tree_common import fit_tree_elasticity
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class SVRPlugin(BaseModelPlugin):
    key = "svr"
    family = "ml_nonparam"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def default_hparams(self) -> dict[str, Any]:
        return {"kernel": "rbf", "C": 10.0}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.svm import SVR

        hp = self.resolve_hparams(ctx)
        return fit_tree_elasticity(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: SVR(kernel=str(hp["kernel"]), C=float(hp["C"])),
            test=ctx.test,
        )
