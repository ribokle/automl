"""TBATS plugin (optional dependency: tbats). Forecast-only."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.models.library._statsmodels_ts import _index_labels
from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.metrics import wape_units
from core.models.result import Capability, ForecastBlock, ModelResult, ProblemType

TARGET = "log_units"


@register
class TBATSPlugin(BaseModelPlugin):
    key = "tbats"
    family = "timeseries"
    problem_types = frozenset({ProblemType.FORECAST})
    capabilities = Capability.FORECAST | Capability.NEEDS_TIME_INDEX | Capability.HEAVY_DEP
    required_packages = ("tbats",)

    def default_hparams(self) -> dict[str, Any]:
        return {"seasonal_periods": [52]}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from tbats import TBATS

        sub = frame[[TARGET]].dropna()
        if len(sub) < 12:
            raise ValueError(f"too few rows for tbats ({len(sub)})")
        endog = sub[TARGET].astype(float).to_numpy()
        periods = self.resolve_hparams(ctx)["seasonal_periods"]
        seasonal = periods if len(sub) >= 2 * max(periods) else None
        estimator = TBATS(seasonal_periods=seasonal, use_arma_errors=False, show_warnings=False)
        model = estimator.fit(endog)

        diagnostics: dict = {}
        forecast = None
        test = ctx.test
        if test is not None and len(test):
            tsub = test[[TARGET]].dropna()
            if len(tsub):
                mean_log = np.asarray(model.forecast(steps=len(tsub)), dtype=float)
                y_test = tsub[TARGET].astype(float).to_numpy()
                diagnostics["test_wape"] = wape_units(y_test, mean_log)
                diagnostics["n_test"] = int(len(y_test))
                forecast = ForecastBlock(
                    horizon=len(mean_log),
                    index=_index_labels(tsub, len(mean_log)),
                    mean=[float(v) for v in np.exp(mean_log)],
                )

        return ModelResult(
            ppg_id=ctx.ppg_id,
            model=self.key,
            problem_type=ProblemType.FORECAST,
            own_elasticity=None,
            forecast=forecast,
            n_obs=int(len(sub)),
            diagnostics=diagnostics,
            capabilities=self.capabilities,
        )
