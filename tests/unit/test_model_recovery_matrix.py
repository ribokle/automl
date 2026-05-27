"""Cross-dataset recovery matrix.

Runs each model family against the synthetic regime built for its failure mode
(see ``tests/datasets.py``) and asserts it recovers the known, benchmark-anchored
ground truth — so models are exercised on diverse data, not one generator.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.models.library import registry
from core.models.library.base import FitContext
from core.models.result import ProblemType
from tests import datasets

# Linear-coefficient elasticity models: expect sign + a magnitude band.
_LINEAR = ["loglog_ols", "ridge", "lasso", "elasticnet", "bayesian_ridge"]
# Nonparametric / tree elasticity models: expect the correct sign (bump
# recovery understates magnitude, so we don't band it tightly).
_NONPARAM = ["lightgbm", "random_forest", "extra_trees", "gaussian_process", "svr", "knn"]


def _fit_per_ppg(key: str, frame, truth):
    out = {}
    for ppg in truth.elasticity:
        slice_ = frame[frame["ppg_id"] == ppg]
        ctx = FitContext(ppg_id=ppg, controls=truth.controls)
        out[ppg] = registry.get(key).fit(slice_, ctx)
    return out


@pytest.mark.parametrize("key", _LINEAR)
def test_linear_models_recover_benchmark_elasticity(key: str) -> None:
    frame, truth = datasets.clean_panel()
    fits = _fit_per_ppg(key, frame, truth)
    for ppg, result in fits.items():
        assert result.sign_ok, f"{key}/{ppg}: {result.own_elasticity}"
        assert abs(result.own_elasticity - truth.elasticity[ppg]) < 0.9, (
            f"{key}/{ppg}: got {result.own_elasticity}, truth {truth.elasticity[ppg]}"
        )


@pytest.mark.parametrize("key", _NONPARAM)
def test_nonparametric_models_recover_sign(key: str) -> None:
    frame, truth = datasets.clean_panel()
    for ppg, result in _fit_per_ppg(key, frame, truth).items():
        assert result.sign_ok, f"{key}/{ppg}: {result.own_elasticity}"


def test_double_ml_beats_naive_ols_under_confounding() -> None:
    frame, truth = datasets.confounded_with_instrument()
    ppg = next(iter(truth.elasticity))
    e_true = truth.elasticity[ppg]
    naive = float(np.polyfit(frame["log_price"], frame["log_units"], 1)[0])
    dml = registry.get("double_ml").fit(frame, FitContext(ppg_id=ppg, controls=[]))
    assert dml.sign_ok
    assert abs(dml.own_elasticity - e_true) < abs(naive - e_true)


def test_iv_recovers_under_endogeneity_when_available() -> None:
    plugin = registry.get("iv_2sls")
    if not plugin.is_available():
        pytest.skip("linearmodels not installed")
    frame, truth = datasets.confounded_with_instrument()
    ppg = next(iter(truth.elasticity))
    naive = float(np.polyfit(frame["log_price"], frame["log_units"], 1)[0])
    result = plugin.fit(
        frame, FitContext(ppg_id=ppg, controls=[], hparams={"instruments": [truth.instrument]})
    )
    assert abs(result.own_elasticity - truth.elasticity[ppg]) < abs(naive - truth.elasticity[ppg])


@pytest.mark.parametrize("key", ["arimax", "sarimax", "state_space"])
def test_exog_timeseries_recover_sign_and_forecast(key: str) -> None:
    frame, truth = datasets.seasonal_series()
    ppg = next(iter(truth.elasticity))
    train, test = frame.iloc[:130], frame.iloc[130:]
    result = registry.get(key).fit(
        train, FitContext(ppg_id=ppg, controls=[], test=test, problem_type=ProblemType.FORECAST)
    )
    assert result.forecast is not None
    assert result.own_elasticity is not None and result.own_elasticity < 0


@pytest.mark.parametrize("key", ["ets", "holt_winters"])
def test_smoothing_models_forecast_seasonal(key: str) -> None:
    frame, truth = datasets.seasonal_series()
    ppg = next(iter(truth.elasticity))
    train, test = frame.iloc[:130], frame.iloc[130:]
    result = registry.get(key).fit(train, FitContext(ppg_id=ppg, controls=[], test=test))
    assert result.forecast is not None and result.forecast.horizon == len(test)


@pytest.mark.parametrize("key", ["fixed_effects", "random_effects"])
def test_panel_models_recover_within_entity_elasticity(key: str) -> None:
    plugin = registry.get(key)
    if not plugin.is_available():
        pytest.skip("linearmodels not installed")
    frame, truth = datasets.store_panel()
    ppg = next(iter(truth.elasticity))
    result = plugin.fit(frame, FitContext(ppg_id=ppg, controls=[], problem_type=ProblemType.PANEL))
    assert result.sign_ok
    assert abs(result.own_elasticity - truth.elasticity[ppg]) < 0.9


def test_demand_system_recovers_cross_substitution() -> None:
    frame, truth = datasets.cross_price_pair()
    a, b = (k for k in truth.elasticity)
    result = registry.get("crossprice_loglog").fit(
        frame, FitContext(ppg_id=a, controls=[], problem_type=ProblemType.DEMAND_SYSTEM)
    )
    assert result.own_elasticity < 0
    assert result.cross_price[a][b] > 0.2  # substitute -> positive cross elasticity


def test_robust_tracks_bulk_elasticity_under_outliers() -> None:
    frame, truth = datasets.outlier_promo()
    ppg = next(iter(truth.elasticity))
    e_true = truth.elasticity[ppg]
    ctx = FitContext(ppg_id=ppg, controls=[])
    ols = registry.get("loglog_ols").fit(frame, ctx).own_elasticity
    theil = registry.get("theil_sen").fit(frame, ctx).own_elasticity
    # The robust estimator should sit at least as close to the bulk truth as OLS.
    assert abs(theil - e_true) <= abs(ols - e_true) + 1e-9
