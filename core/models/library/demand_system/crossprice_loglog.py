"""Cross-price log-log demand system (multi-entity, no extra deps).

Unlike the per-cell elasticity models, this fits ONE equation for a target PPG
regressed on EVERY PPG's log price (plus the target's own controls), so the
coefficient on the target's own log price is the own elasticity and the
coefficients on the other PPGs' log prices are the cross-price (cannibalisation)
elasticities. The plugin therefore expects the FULL multi-PPG feature frame, not
a single PPG slice — the modelling agent routes it on the DEMAND_SYSTEM problem
path. It is forecast-/elasticity-reporting but NOT price-sweepable (its
regressors reference other PPGs' prices), so it stays out of
``predictor.PREDICTABLE_MODELS`` and downstream optimisation skips it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.metrics import wape_units
from core.models.result import Capability, ModelResult, ProblemType

LOG_PRICE = "log_price"
TARGET = "log_units"
TIME_COL = "week_start"
_PRICE_PREFIX = "price::"


def _usable_controls(frame: pd.DataFrame, controls: list[str]) -> list[str]:
    cols = [c for c in controls if c in frame.columns and c not in (LOG_PRICE, TARGET)]
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


@register
class CrossPriceLogLogPlugin(BaseModelPlugin):
    key = "crossprice_loglog"
    family = "demand_system"
    problem_types = frozenset({ProblemType.DEMAND_SYSTEM, ProblemType.CROSS_PRICE})
    capabilities = (
        Capability.SCALAR_ELASTICITY | Capability.CROSS_PRICE_MATRIX | Capability.NEEDS_PANEL
    )
    required_packages = ("statsmodels",)

    def default_hparams(self) -> dict[str, Any]:
        return {"test_ratio": 0.2}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        if "grain_unit" in frame.columns:
            raise ValueError("cross-price demand system runs at a chain grain (no grain_unit)")
        for col in (TIME_COL, "ppg_id", LOG_PRICE, TARGET):
            if col not in frame.columns:
                raise ValueError(f"frame missing {col}")

        target = ctx.ppg_id
        prices = frame.pivot_table(index=TIME_COL, columns="ppg_id", values=LOG_PRICE)
        units = frame.pivot_table(index=TIME_COL, columns="ppg_id", values=TARGET)
        ppgs = [str(c) for c in prices.columns]
        if target not in units.columns or len(ppgs) < 2:
            raise ValueError("need >= 2 PPGs aligned by week for a demand system")

        usable = _usable_controls(frame[frame["ppg_id"] == target], ctx.controls)
        ctrl = (
            frame[frame["ppg_id"] == target].set_index(TIME_COL)[usable] if usable else None
        )

        design = prices.add_prefix(_PRICE_PREFIX)
        design = design.join(units[target].rename("y"))
        if ctrl is not None:
            design = design.join(ctrl)
        design = design.dropna().sort_index()
        if len(design) < 12:
            raise ValueError(f"too few aligned weeks for demand system ({len(design)})")

        x_cols = [f"{_PRICE_PREFIX}{p}" for p in ppgs] + usable
        ratio = float(self.resolve_hparams(ctx)["test_ratio"])
        n = len(design)
        n_test = max(1, int(round(n * ratio)))
        n_train = max(2, n - n_test)
        train, test = design.iloc[:n_train], design.iloc[n_train:]

        y = train["y"].to_numpy(dtype=float)
        X = sm.add_constant(train[x_cols].to_numpy(dtype=float), has_constant="add")
        model = sm.OLS(y, X).fit()
        params = dict(zip(["const", *x_cols], (float(v) for v in model.params), strict=True))

        cross_row = {p: params[f"{_PRICE_PREFIX}{p}"] for p in ppgs}
        own = float(cross_row[target])

        diagnostics: dict[str, Any] = {
            "r_squared": float(model.rsquared),
            "n_ppgs": len(ppgs),
        }
        if len(test):
            y_test = test["y"].to_numpy(dtype=float)
            X_test = sm.add_constant(test[x_cols].to_numpy(dtype=float), has_constant="add")
            diagnostics["test_wape"] = wape_units(y_test, np.asarray(model.predict(X_test)))
            diagnostics["n_test"] = int(len(test))

        return ModelResult(
            ppg_id=target,
            model=self.key,
            problem_type=ProblemType.DEMAND_SYSTEM,
            own_elasticity=own,
            std_err=None,
            p_value=None,
            cross_price={target: cross_row},
            r_squared=float(model.rsquared),
            n_obs=int(n_train),
            controls=usable,
            coefficients={},
            diagnostics=diagnostics,
            capabilities=self.capabilities,
        )
