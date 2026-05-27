"""Extremely Randomized Trees elasticity plugin (scikit-learn)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._tree_common import fit_tree_elasticity
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class ExtraTreesPlugin(BaseModelPlugin):
    key = "extra_trees"
    family = "trees"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def default_hparams(self) -> dict[str, Any]:
        return {"n_estimators": 300, "min_samples_leaf": 3, "max_depth": None}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.ensemble import ExtraTreesRegressor

        hp = self.resolve_hparams(ctx)
        return fit_tree_elasticity(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: ExtraTreesRegressor(
                n_estimators=int(hp["n_estimators"]),
                min_samples_leaf=int(hp["min_samples_leaf"]),
                max_depth=hp["max_depth"],
                random_state=int(ctx.rng_seed),
                n_jobs=1,
            ),
            test=ctx.test,
        )
