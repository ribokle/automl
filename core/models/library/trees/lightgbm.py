"""LightGBM plugin — wraps the existing ``fit_lightgbm`` fitter.

Hyperparameters are read from ``ctx.hparams`` (resolved from
``settings.model_hparams.lightgbm``) and fall back to ``default_hparams``;
nothing is hardcoded in the call.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.lightgbm_model import fit_lightgbm
from core.models.result import Capability, ModelResult, ProblemType, from_elasticity_fit


@register
class LightGBMPlugin(BaseModelPlugin):
    key = "lightgbm"
    family = "trees"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY, ProblemType.FORECAST})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("lightgbm",)

    def default_hparams(self) -> dict[str, Any]:
        return {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_child_samples": 5,
        }

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        hp = self.resolve_hparams(ctx)
        fit = fit_lightgbm(
            ctx.ppg_id,
            frame,
            ctx.controls,
            test=ctx.test,
            n_estimators=int(hp["n_estimators"]),
            learning_rate=float(hp["learning_rate"]),
            num_leaves=int(hp["num_leaves"]),
            min_child_samples=int(hp["min_child_samples"]),
            random_state=int(ctx.rng_seed),
        )
        return from_elasticity_fit(fit, capabilities=self.capabilities)
