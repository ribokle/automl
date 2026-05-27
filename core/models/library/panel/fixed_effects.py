"""Fixed-effects panel plugin (optional dependency: linearmodels).

One-way entity (store) fixed effects on a per-PPG store×week panel; the
log_price coefficient is the within-store own-price elasticity.
"""
from __future__ import annotations

import pandas as pd

from core.models.library._panel_common import fit_panel
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.result import Capability, ModelResult, ProblemType


@register
class FixedEffectsPlugin(BaseModelPlugin):
    key = "fixed_effects"
    family = "panel"
    problem_types = frozenset({ProblemType.PANEL, ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.NEEDS_PANEL | Capability.HEAVY_DEP
    required_packages = ("linearmodels",)

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from linearmodels.panel import PanelOLS

        def _fit(y, X):
            return PanelOLS(y, X, entity_effects=True, drop_absorbed=True).fit(
                cov_type="clustered", cluster_entity=True
            )

        return fit_panel(ctx.ppg_id, frame, ctx.controls, model_name=self.key, fitter=_fit)
