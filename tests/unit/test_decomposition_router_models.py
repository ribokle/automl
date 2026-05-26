"""Decomposition handles the new linear + tree winners (Phase 8 families).

Linear-coefficient winners take the closed-form path; tree winners take the
ablation path via the predictor. Both must produce an attributed weekly frame
rather than being skipped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.agents.decomposition import _decompose_one_ppg


def _frame(n: int = 160, beta: float = -1.5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_price = rng.normal(0.0, 0.25, n)
    acv = rng.normal(0.0, 1.0, n)
    log_units = 5.0 + beta * log_price + 0.3 * acv + rng.normal(0.0, 0.05, n)
    return pd.DataFrame(
        {
            "ppg_id": "P1",
            "log_price": log_price,
            "log_units": log_units,
            "log_distribution_acv": acv,
            "week_start": pd.date_range("2021-01-01", periods=n, freq="W"),
        }
    )


def test_ridge_winner_uses_closed_form() -> None:
    frame = _frame()
    row = {"ppg_id": "P1", "winner_model": "ridge", "winner": {"coefficients": {}}}
    weekly, summary = _decompose_one_ppg(
        "P1", frame, ["log_distribution_acv"], "ridge", row
    )
    assert summary["model_kind"] == "ridge"
    assert summary["attribution_method"] == "closed_form"
    assert len(weekly) == len(frame)


def test_random_forest_winner_uses_ablation() -> None:
    frame = _frame()
    row = {"ppg_id": "P1", "winner_model": "random_forest", "winner": {"coefficients": {}}}
    weekly, summary = _decompose_one_ppg(
        "P1", frame, ["log_distribution_acv"], "random_forest", row
    )
    assert summary["model_kind"] == "random_forest"
    assert summary["attribution_method"] == "ablation"
    assert len(weekly) == len(frame)
