"""CatBoost elasticity plugin (optional dependency: catboost)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._tree_common import fit_tree_elasticity
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class CatBoostPlugin(BaseModelPlugin):
    key = "catboost"
    family = "trees"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = (
        Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS | Capability.HEAVY_DEP
    )
    required_packages = ("catboost",)

    def default_hparams(self) -> dict[str, Any]:
        return {"iterations": 300, "learning_rate": 0.05, "depth": 4}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from catboost import CatBoostRegressor

        hp = self.resolve_hparams(ctx)
        return fit_tree_elasticity(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: CatBoostRegressor(
                iterations=int(hp["iterations"]),
                learning_rate=float(hp["learning_rate"]),
                depth=int(hp["depth"]),
                random_seed=int(ctx.rng_seed),
                verbose=False,
                allow_writing_files=False,
            ),
            test=ctx.test,
        )
