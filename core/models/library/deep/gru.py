"""GRU sequence-forecaster plugin (optional dependency: torch). Forecast-only."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.models.library._torch_seq import fit_torch_seq
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class GRUPlugin(BaseModelPlugin):
    key = "gru"
    family = "deep"
    problem_types = frozenset({ProblemType.FORECAST})
    capabilities = Capability.FORECAST | Capability.NEEDS_TIME_INDEX | Capability.HEAVY_DEP
    required_packages = ("torch",)

    def default_hparams(self) -> dict[str, Any]:
        return {"lookback": 8, "hidden": 16, "epochs": 200, "lr": 0.01}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        hp = {**self.resolve_hparams(ctx), "rng_seed": int(ctx.rng_seed)}
        return fit_torch_seq(
            ctx.ppg_id, frame, ctx.controls,
            model_name=self.key, cell="gru", test=ctx.test, hparams=hp,
        )
