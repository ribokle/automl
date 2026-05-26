"""XGBoost elasticity plugin (optional dependency: xgboost)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._tree_common import fit_tree_elasticity
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class XGBoostPlugin(BaseModelPlugin):
    key = "xgboost"
    family = "trees"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = (
        Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS | Capability.HEAVY_DEP
    )
    required_packages = ("xgboost",)

    def default_hparams(self) -> dict[str, Any]:
        return {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
        }

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from xgboost import XGBRegressor

        hp = self.resolve_hparams(ctx)
        return fit_tree_elasticity(
            ctx.ppg_id,
            frame,
            ctx.controls,
            model_name=self.key,
            estimator_factory=lambda: XGBRegressor(
                n_estimators=int(hp["n_estimators"]),
                learning_rate=float(hp["learning_rate"]),
                max_depth=int(hp["max_depth"]),
                subsample=float(hp["subsample"]),
                random_state=int(ctx.rng_seed),
                verbosity=0,
            ),
            test=ctx.test,
        )
