"""Shared ``Predictor`` abstraction.

The OLS path reproduces the closed-form prediction exactly; the LightGBM
path refits the booster on the PPG slice deterministically (random_state
fixed) and predicts via the trained estimator. Both expose the same
``predict_log`` / ``predict_units`` interface so downstream agents can
ignore the underlying family.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.models.predictor import Predictor, build_predictor

COEFS = {
    "const": 6.5,
    "log_price": -2.0,
    "tpr_share": 0.6,
    "log_distribution_acv": 0.4,
}


def _toy_frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    log_base_price = np.log(3.0) * np.ones(n)
    log_price = log_base_price + np.log(0.8 + 0.4 * rng.random(n))
    tpr = rng.binomial(1, 0.3, size=n).astype(float)
    log_acv = np.log(70 + 20 * rng.random(n))
    log_units = (
        COEFS["const"]
        + COEFS["log_price"] * log_price
        + COEFS["tpr_share"] * tpr
        + COEFS["log_distribution_acv"] * (log_acv - log_acv.mean())
        + rng.normal(0, 0.05, size=n)
    )
    return pd.DataFrame(
        {
            "ppg_id": "PPG",
            "week_start": pd.date_range("2024-01-01", periods=n, freq="W").astype(str),
            "log_units": log_units,
            "log_price": log_price,
            "log_base_price": log_base_price,
            "tpr_share": tpr,
            "log_distribution_acv": log_acv,
        }
    )


def test_ols_predictor_matches_closed_form() -> None:
    frame = _toy_frame()
    pred = Predictor(
        ppg_id="PPG",
        model_kind="loglog_ols",
        feature_cols=[c for c in COEFS if c != "const"],
        coefficients=COEFS,
    )
    log_units = pred.predict_log(frame)
    expected = (
        COEFS["const"]
        + COEFS["log_price"] * frame["log_price"].to_numpy()
        + COEFS["tpr_share"] * frame["tpr_share"].to_numpy()
        + COEFS["log_distribution_acv"] * frame["log_distribution_acv"].to_numpy()
    )
    assert np.allclose(log_units, expected, atol=1e-12)


def test_build_predictor_for_lightgbm_recovers_negative_elasticity() -> None:
    """LightGBM refitted on a clean DGP should bump-elasticity negatively."""
    frame = _toy_frame(n=120)
    modeling_row = {
        "ppg_id": "PPG",
        "winner_model": "lightgbm",
        "winner": {"model": "lightgbm"},
    }
    predictor = build_predictor(
        modeling_row,
        frame,
        controls=["tpr_share", "log_distribution_acv"],
    )
    assert predictor.model_kind == "lightgbm"
    assert "log_price" in predictor.feature_cols
    # Numerical bump on log_price: prediction should drop when price rises.
    base = predictor.predict_log(frame).mean()
    bumped = frame.copy()
    bumped["log_price"] = bumped["log_price"] + math.log(1.05)
    higher = predictor.predict_log(bumped).mean()
    assert higher < base


def test_lightgbm_monotone_constraint_pins_log_price_decreasing() -> None:
    """Even on noisy data with a few positive-slope rows, the monotone
    constraint guarantees the booster's predictions are non-increasing in
    log_price across the *training* domain."""
    rng = np.random.default_rng(101)
    n = 200
    log_base_price = np.log(3.0) * np.ones(n)
    log_price = log_base_price + np.log(0.8 + 0.4 * rng.random(n))
    # Inject ~5% positive-slope outliers to make sure the constraint
    # actually binds.
    noise = rng.normal(0, 0.1, n)
    log_units = 6.5 - 2.0 * log_price + noise
    outliers = rng.choice(n, size=10, replace=False)
    log_units[outliers] += 3.0 * (log_price[outliers] - log_base_price[outliers])
    frame = pd.DataFrame(
        {
            "ppg_id": "PPG",
            "week_start": pd.date_range("2024-01-01", periods=n, freq="W").astype(str),
            "log_units": log_units,
            "log_price": log_price,
            "log_base_price": log_base_price,
            "tpr_share": rng.binomial(1, 0.3, size=n).astype(float),
            "log_distribution_acv": np.log(70 + 20 * rng.random(n)),
        }
    )
    predictor = build_predictor(
        {"ppg_id": "PPG", "winner_model": "lightgbm", "winner": {}},
        frame,
        controls=["tpr_share", "log_distribution_acv"],
        test_ratio=0.0,
    )
    # Sweep log_price within the training range; predictions must be
    # non-increasing.
    sweep = pd.DataFrame(
        {
            "log_price": np.linspace(log_price.min(), log_price.max(), 50),
            "tpr_share": 0.0,
            "log_distribution_acv": float(np.log(85)),
        }
    )
    preds = predictor.predict_log(sweep[predictor.feature_cols])
    assert np.all(np.diff(preds) <= 1e-9), "monotone constraint violated"


def test_build_predictor_for_lightgbm_full_frame_when_test_ratio_zero() -> None:
    frame = _toy_frame(n=80)
    modeling_row = {"ppg_id": "PPG", "winner_model": "lightgbm", "winner": {}}
    predictor = build_predictor(
        modeling_row, frame, controls=["tpr_share"], test_ratio=0.0
    )
    # Booster predicts log_units; on the training frame the predictions
    # should correlate strongly with the observed log_units.
    pred = predictor.predict_log(frame)
    corr = np.corrcoef(pred, frame["log_units"].to_numpy())[0, 1]
    assert corr > 0.8


def test_build_predictor_for_ols_uses_saved_coefficients() -> None:
    frame = _toy_frame()
    modeling_row = {
        "ppg_id": "PPG",
        "winner_model": "loglog_ols",
        "winner": {"model": "loglog_ols", "coefficients": COEFS},
    }
    predictor = build_predictor(modeling_row, frame, controls=[])
    assert predictor.coefficients == COEFS
    assert predictor.model_kind == "loglog_ols"
