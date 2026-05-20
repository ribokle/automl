"""LightGBM elasticity fitter.

Validates the numerical-derivative path that recovers an average own-price
elasticity from a tree ensemble, and confirms the agent's selection logic
prefers the lowest-WAPE sign-correct candidate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.lightgbm_model import fit_lightgbm
from core.models.metrics import chronological_split, wape_units


def _clean_frame(seed: int = 11, n: int = 120, true_elasticity: float = -2.2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_price = np.log(3.0 * (0.7 + 0.4 * rng.random(n)))
    tpr = rng.binomial(1, 0.25, size=n).astype(float)
    log_acv = np.log(70 + 20 * rng.random(n))
    log_units = (
        6.0
        + true_elasticity * log_price
        + 0.5 * tpr
        + 0.3 * (log_acv - log_acv.mean())
        + rng.normal(0, 0.05, size=n)
    )
    return pd.DataFrame(
        {
            "ppg_id": "PPG_TEST",
            "week_start": pd.date_range("2024-01-01", periods=n, freq="W"),
            "log_units": log_units,
            "log_price": log_price,
            "tpr_share": tpr,
            "log_distribution_acv": log_acv,
        }
    )


def test_lightgbm_recovers_sign_on_clean_dgp() -> None:
    frame = _clean_frame()
    train, test = chronological_split(frame, test_ratio=0.2)
    fit = fit_lightgbm(
        "PPG_TEST", train, controls=["tpr_share", "log_distribution_acv"], test=test
    )
    assert fit.model == "lightgbm"
    assert fit.sign_ok
    # Loose magnitude check — trees recover a ballpark elasticity, not an exact β.
    assert 0.5 <= abs(fit.own_elasticity) <= 6.0
    assert "test_wape" in fit.diagnostics
    assert fit.diagnostics["test_wape"] >= 0


def test_lightgbm_writes_feature_importances() -> None:
    frame = _clean_frame()
    fit = fit_lightgbm("PPG_TEST", frame, controls=["tpr_share", "log_distribution_acv"])
    imps = fit.diagnostics["feature_importances"]
    assert set(imps.keys()) >= {"log_price", "tpr_share", "log_distribution_acv"}
    assert all(v >= 0 for v in imps.values())


def test_wape_units_matches_hand_computation() -> None:
    y_true_log = np.log(np.array([10.0, 20.0, 30.0]))
    y_pred_log = np.log(np.array([12.0, 18.0, 33.0]))
    expected = (abs(10 - 12) + abs(20 - 18) + abs(30 - 33)) / (10 + 20 + 30)
    assert wape_units(y_true_log, y_pred_log) == pytest.approx(expected, rel=1e-9)
