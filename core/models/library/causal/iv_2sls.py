"""Instrumental-variables (2SLS) elasticity plugin (optional dep: linearmodels).

Corrects price endogeneity by instrumenting ``log_price`` with cost-shifter /
Hausman-style columns the operator nominates via ``hparams['instruments']``.
Because the default feature set carries no instrument, this plugin raises (and
the escalation loop skips it) unless an instrument column is both configured
AND present — so it is opt-in by construction. It emits a full log-space
coefficient vector, so it feeds the downstream predictor like any linear model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.models.library.base import BaseModelPlugin, FitContext
from core.models.library.registry import register
from core.models.metrics import wape_units
from core.models.result import Capability, ModelResult, ProblemType

LOG_PRICE = "log_price"
TARGET = "log_units"


def _usable_controls(frame: pd.DataFrame, controls: list[str], instruments: list[str]) -> list[str]:
    excluded = {LOG_PRICE, TARGET, *instruments}
    cols = [c for c in controls if c in frame.columns and c not in excluded]
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


def _predict_log(coefs: dict[str, float], frame: pd.DataFrame) -> np.ndarray:
    out = np.full(len(frame), float(coefs.get("const", 0.0)), dtype=float)
    for col, beta in coefs.items():
        if col != "const" and col in frame.columns:
            out += float(beta) * frame[col].astype(float).to_numpy()
    return out


@register
class IV2SLSPlugin(BaseModelPlugin):
    key = "iv_2sls"
    family = "causal"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = (
        Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS | Capability.HEAVY_DEP
    )
    required_packages = ("linearmodels",)

    def default_hparams(self) -> dict[str, Any]:
        return {"instruments": []}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from linearmodels.iv import IV2SLS

        instruments = [
            c for c in (self.resolve_hparams(ctx).get("instruments") or []) if c in frame.columns
        ]
        if not instruments:
            raise ValueError(
                "iv_2sls requires instrument columns via hparams['instruments'] "
                "(none configured or present in the frame)"
            )

        usable = _usable_controls(frame, ctx.controls, instruments)
        data = frame[[TARGET, LOG_PRICE, *usable, *instruments]].dropna()
        if len(data) < 20:
            raise ValueError(f"too few rows for 2SLS ({len(data)})")

        dep = data[TARGET].astype(float)
        exog = data[usable].astype(float).copy()
        exog.insert(0, "const", 1.0)
        endog = data[[LOG_PRICE]].astype(float)
        instr = data[instruments].astype(float)
        res = IV2SLS(dep, exog, endog, instr).fit()

        own = float(res.params[LOG_PRICE])
        std_err = float(res.std_errors[LOG_PRICE])
        coefs = {str(name): float(res.params[name]) for name in res.params.index}

        diagnostics: dict[str, Any] = {
            "train_wape": wape_units(dep.to_numpy(), _predict_log(coefs, data)),
            "instruments": instruments,
            "rsquared": float(getattr(res, "rsquared", float("nan"))),
        }
        test = ctx.test
        if test is not None and len(test):
            tsub = test[[TARGET, LOG_PRICE, *usable]].dropna()
            if len(tsub):
                diagnostics["test_wape"] = wape_units(
                    tsub[TARGET].astype(float).to_numpy(), _predict_log(coefs, tsub)
                )
                diagnostics["n_test"] = int(len(tsub))

        return ModelResult(
            ppg_id=ctx.ppg_id,
            model=self.key,
            problem_type=ProblemType.OWN_ELASTICITY,
            own_elasticity=own,
            std_err=std_err,
            p_value=None,
            r_squared=diagnostics["rsquared"],
            n_obs=int(len(data)),
            controls=usable,
            coefficients=coefs,
            diagnostics=diagnostics,
            capabilities=self.capabilities,
        )
