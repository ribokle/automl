"""Log-log OLS plugin — wraps the existing ``fit_loglog`` fitter."""
from __future__ import annotations

import pandas as pd

from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.loglog_ols import fit_loglog
from core.models.result import Capability, ModelResult, ProblemType, from_elasticity_fit


@register
class LogLogOLSPlugin(BaseModelPlugin):
    key = "loglog_ols"
    family = "classical"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        fit = fit_loglog(ctx.ppg_id, frame, ctx.controls, test=ctx.test)
        return from_elasticity_fit(fit, capabilities=self.capabilities)
