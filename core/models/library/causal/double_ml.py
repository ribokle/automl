"""Double / debiased machine-learning elasticity plugin (scikit-learn).

Partially-linear model ``log_units = θ·log_price + g(controls) + ε`` estimated
by Frisch-Waugh residualisation with cross-fitting: flexible ML nuisance models
predict log_units and log_price from the controls, and θ (the own-price
elasticity) is the OLS slope of the residuals. θ is the endogeneity-corrected
price coefficient; the control coefficients are recovered by a final OLS on the
θ-adjusted target so the result is a full log-space coefficient vector and feeds
the downstream predictor like any linear model.
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


def _usable_controls(frame: pd.DataFrame, controls: list[str]) -> list[str]:
    cols = [c for c in controls if c in frame.columns and c not in (LOG_PRICE, TARGET)]
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


@register
class DoubleMLPlugin(BaseModelPlugin):
    key = "double_ml"
    family = "causal"
    problem_types = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS
    required_packages = ("sklearn",)

    def default_hparams(self) -> dict[str, Any]:
        return {"n_splits": 3, "n_estimators": 200, "min_samples_leaf": 5}

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold, cross_val_predict

        if LOG_PRICE not in frame.columns or TARGET not in frame.columns:
            raise ValueError(f"frame missing {LOG_PRICE} or {TARGET}")
        hp = self.resolve_hparams(ctx)
        usable = _usable_controls(frame, controls=ctx.controls)
        sub = frame[[TARGET, LOG_PRICE, *usable]].dropna()
        n = len(sub)
        y = sub[TARGET].astype(float).to_numpy()
        t = sub[LOG_PRICE].astype(float).to_numpy()

        if usable:
            X = sub[usable].astype(float).to_numpy()
            n_splits = max(2, min(int(hp["n_splits"]), n // 5))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(ctx.rng_seed))

            def _nuisance() -> RandomForestRegressor:
                return RandomForestRegressor(
                    n_estimators=int(hp["n_estimators"]),
                    min_samples_leaf=int(hp["min_samples_leaf"]),
                    random_state=int(ctx.rng_seed),
                    n_jobs=1,
                )

            y_hat = cross_val_predict(_nuisance(), X, y, cv=kf)
            t_hat = cross_val_predict(_nuisance(), X, t, cv=kf)
            y_res = y - y_hat
            t_res = t - t_hat
        else:
            X = np.empty((n, 0))
            y_res = y - y.mean()
            t_res = t - t.mean()

        denom = float(np.dot(t_res, t_res))
        if denom <= 0:
            raise ValueError("log_price has no residual variation after controls")
        theta = float(np.dot(t_res, y_res) / denom)

        resid = y_res - theta * t_res
        var_theta = float(np.mean((t_res**2) * (resid**2)) / (n * (denom / n) ** 2))
        std_err = float(np.sqrt(max(var_theta, 0.0)))

        # Recover control coefficients on the theta-adjusted target so the
        # winner carries a full, predictor-compatible log-space coefficient set.
        adj = y - theta * t
        coefs: dict[str, float] = {}
        if usable:
            ols = LinearRegression().fit(X, adj)
            coefs["const"] = float(ols.intercept_)
            coefs.update({c: float(b) for c, b in zip(usable, ols.coef_)})
        else:
            coefs["const"] = float(adj.mean())
        coefs[LOG_PRICE] = theta

        def _predict_log(d: pd.DataFrame) -> np.ndarray:
            out = np.full(len(d), coefs["const"], dtype=float)
            for col, beta in coefs.items():
                if col != "const" and col in d.columns:
                    out += beta * d[col].astype(float).to_numpy()
            return out

        diagnostics: dict[str, Any] = {
            "train_wape": wape_units(y, _predict_log(sub)),
            "n_splits": int(hp["n_splits"]),
            "nuisance": "random_forest",
        }
        test = ctx.test
        if test is not None and len(test):
            tsub = test[[TARGET, LOG_PRICE, *usable]].dropna()
            if len(tsub):
                y_test = tsub[TARGET].astype(float).to_numpy()
                diagnostics["test_wape"] = wape_units(y_test, _predict_log(tsub))
                diagnostics["n_test"] = int(len(tsub))

        return ModelResult(
            ppg_id=ctx.ppg_id,
            model=self.key,
            problem_type=ProblemType.OWN_ELASTICITY,
            own_elasticity=theta,
            std_err=std_err,
            p_value=None,
            r_squared=None,
            n_obs=n,
            controls=usable,
            coefficients=coefs,
            diagnostics=diagnostics,
            capabilities=self.capabilities,
        )
