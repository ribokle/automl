"""Random-effects panel plugin (optional dependency: linearmodels).

Random store effects on a per-PPG store×week panel; the log_price coefficient
is the own-price elasticity. A constant is added so the GLS intercept is
identified.
"""
from __future__ import annotations

import pandas as pd

from core.models.library._panel_common import fit_panel
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class RandomEffectsPlugin(BaseModelPlugin):
    key = "random_effects"
    family = "panel"
    problem_types = frozenset({ProblemType.PANEL, ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.NEEDS_PANEL | Capability.HEAVY_DEP
    required_packages = ("linearmodels",)

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from linearmodels.panel import RandomEffects
        from linearmodels.panel.utility import AbsorbingEffectError  # noqa: F401

        def _fit(y, X):
            X = X.copy()
            X.insert(0, "const", 1.0)
            return RandomEffects(y, X).fit()

        return fit_panel(ctx.ppg_id, frame, ctx.controls, model_name=self.key, fitter=_fit)
