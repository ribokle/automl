"""Ablation decomposition: reconciliation + group attribution for the
LightGBM path. Mirrors ``test_decomposition`` but using the
``Predictor`` + ablation pipeline so we cover the non-OLS branch the
Phase-4b refactor unlocked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.decomp.ablation import (
    decompose_via_ablation,
    reference_frame,
    summarise_groups,
)
from core.models.predictor import Predictor


def _toy_frame(seed: int = 11, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_base_price = np.log(3.0) * np.ones(n)
    log_price = log_base_price + np.log(0.8 + 0.4 * rng.random(n))
    tpr = rng.binomial(1, 0.3, size=n).astype(float)
    log_acv = np.log(70 + 20 * rng.random(n))
    log_units = (
        6.5 - 2.0 * log_price + 0.6 * tpr + 0.4 * (log_acv - log_acv.mean())
        + rng.normal(0, 0.05, size=n)
    )
    return pd.DataFrame(
        {
            "ppg_id": "PPG_A",
            "week_start": pd.date_range("2024-01-01", periods=n, freq="W").astype(str),
            "log_units": log_units,
            "log_price": log_price,
            "log_base_price": log_base_price,
            "tpr_share": tpr,
            "log_distribution_acv": log_acv,
        }
    )


def _ols_predictor(coefs: dict[str, float]) -> Predictor:
    return Predictor(
        ppg_id="PPG_A",
        model_kind="loglog_ols",
        feature_cols=[c for c in coefs if c != "const"],
        coefficients=coefs,
    )


def test_reference_frame_zeros_dummies_and_means_continuous() -> None:
    frame = _toy_frame()
    ref = reference_frame(
        frame, ["log_price", "tpr_share", "log_distribution_acv"]
    )
    assert (ref["tpr_share"] == 0.0).all()
    # log_price baselines to per-row log_base_price (here uniform 3.0).
    assert np.allclose(ref["log_price"].to_numpy(), np.log(3.0))
    assert np.isclose(
        ref["log_distribution_acv"].iloc[0], frame["log_distribution_acv"].mean()
    )


def test_ablation_reconciles_to_predicted_in_aggregate() -> None:
    """Per-row reconciliation isn't exact for ablation (allocation
    re-normalises the share), but the aggregate ``base + Σ due_group``
    should equal ``predicted`` to within floating-point noise."""
    frame = _toy_frame()
    coefs = {
        "const": 6.5,
        "log_price": -2.0,
        "tpr_share": 0.6,
        "log_distribution_acv": 0.4,
    }
    weekly = decompose_via_ablation(_ols_predictor(coefs), frame)
    summary = summarise_groups(weekly)
    assert abs(summary["reconciliation_pct_error"]) < 1e-6


def test_ablation_residual_identity() -> None:
    frame = _toy_frame()
    coefs = {"const": 6.5, "log_price": -2.0, "tpr_share": 0.6}
    weekly = decompose_via_ablation(_ols_predictor(coefs), frame)
    diff = (weekly["observed"] - (weekly["predicted"] + weekly["residual"])).abs()
    assert diff.max() < 1e-9


def test_ablation_zero_lift_when_at_reference() -> None:
    """If every feature equals its reference, lift and per-group due-tos vanish."""
    n = 30
    frame = pd.DataFrame(
        {
            "log_units": np.zeros(n),
            "log_price": np.log(3.0) * np.ones(n),
            "log_base_price": np.log(3.0) * np.ones(n),
            "tpr_share": np.zeros(n),
        }
    )
    coefs = {"const": 6.0, "log_price": -1.5, "tpr_share": 0.5}
    weekly = decompose_via_ablation(_ols_predictor(coefs), frame)
    assert np.allclose(weekly["lift"], 0.0)
    for col in (c for c in weekly.columns if c.startswith("due_group_")):
        assert np.allclose(weekly[col], 0.0)


def test_ablation_price_drives_negative_lift_when_price_above_base() -> None:
    frame = _toy_frame()
    coefs = {"const": 6.5, "log_price": -2.0, "tpr_share": 0.6}
    weekly = decompose_via_ablation(_ols_predictor(coefs), frame)
    # Where log_price > log_base_price, the price group's contribution
    # should be non-positive (price increase pulls units down) on average.
    high_price = frame["log_price"] > frame["log_base_price"]
    assert weekly.loc[high_price.values, "due_group_price"].mean() <= 0
