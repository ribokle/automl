"""SHAP feature attribution.

Phase 3b' verification anchor: the per-row identity
``ŷ = base_value + Σ shapᵢ`` must hold exactly for OLS (linear identity)
and within tree-prediction tolerance for LightGBM. The agent's
candidates table relies on mean |SHAP| being sorted by importance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from core.models.shap_attribution import (
    MAX_BEESWARM_ROWS,
    lightgbm_shap_summary,
    ols_shap_summary,
)


def _frame(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_price = np.log(3.0 + 0.5 * rng.standard_normal(n))
    tpr = rng.binomial(1, 0.3, size=n).astype(float)
    return pd.DataFrame({"log_price": log_price, "tpr_share": tpr})


def test_ols_shap_reconstructs_prediction() -> None:
    """OLS attribution is the linear identity centred on x̄. Per-row
    sum(shap_i) + base_value must equal the model's ŷ within fp tolerance."""
    frame = _frame()
    coefs = {"const": 6.0, "log_price": -2.0, "tpr_share": 0.5}
    feats = ["log_price", "tpr_share"]
    summary = ols_shap_summary(coefs, frame, feats)

    # Hand-compute the per-row prediction and compare to base + Σ shap.
    pred = (
        coefs["const"]
        + coefs["log_price"] * frame["log_price"]
        + coefs["tpr_share"] * frame["tpr_share"]
    ).to_numpy()
    base = summary["base_value"]
    beeswarm_rows = {r["row"]: r for r in summary["beeswarm"]}
    for row, hand_pred in enumerate(pred):
        if row not in beeswarm_rows:
            continue
        contribs = sum(v["shap"] for v in beeswarm_rows[row]["values"])
        assert base + contribs == pytest.approx(hand_pred, abs=1e-9)


def test_ols_shap_mean_abs_sorted_descending() -> None:
    frame = _frame()
    coefs = {"const": 6.0, "log_price": -2.0, "tpr_share": 0.5}
    summary = ols_shap_summary(coefs, frame, ["log_price", "tpr_share"])
    values = [r["value"] for r in summary["mean_abs_shap"]]
    assert values == sorted(values, reverse=True)
    assert {r["feature"] for r in summary["mean_abs_shap"]} == {"log_price", "tpr_share"}


def test_ols_shap_beeswarm_capped() -> None:
    frame = _frame(n=MAX_BEESWARM_ROWS * 3)
    coefs = {"const": 6.0, "log_price": -2.0}
    summary = ols_shap_summary(coefs, frame, ["log_price"])
    assert len(summary["beeswarm"]) == MAX_BEESWARM_ROWS
    assert summary["n_rows"] == MAX_BEESWARM_ROWS * 3


def test_ols_shap_handles_missing_coefficient_gracefully() -> None:
    """If a control is in X but missing from coefs, it gets β=0 — the
    contribution column is identically zero, but the row count stands."""
    frame = _frame()
    coefs = {"const": 5.0, "log_price": -1.5}  # missing tpr_share
    summary = ols_shap_summary(coefs, frame, ["log_price", "tpr_share"])
    tpr_mean_abs = next(r for r in summary["mean_abs_shap"] if r["feature"] == "tpr_share")
    assert tpr_mean_abs["value"] == pytest.approx(0.0)


def test_lightgbm_shap_reconstructs_prediction() -> None:
    """LightGBM's pred_contrib path is exact tree-SHAP. Per-row sum + bias
    must equal the model's own predict() output."""
    frame = _frame(n=60, seed=2)
    y = (6.5 - 2.0 * frame["log_price"] + 0.4 * frame["tpr_share"]).to_numpy()
    X = frame[["log_price", "tpr_share"]]
    model = LGBMRegressor(n_estimators=30, num_leaves=8, verbosity=-1, random_state=0)
    model.fit(X, y)

    summary = lightgbm_shap_summary(model, X)
    preds = model.predict(X)
    beeswarm_rows = {r["row"]: r for r in summary["beeswarm"]}
    # Bias is the per-row last column; we averaged it. Reconstruct against
    # the raw contribution matrix because the mean drops row-specific bias
    # in tree models (in practice LGBM's bias is constant across rows).
    contribs = model.predict(X, pred_contrib=True)
    for row in beeswarm_rows:
        sum_shap = sum(v["shap"] for v in beeswarm_rows[row]["values"])
        assert sum_shap + contribs[row, -1] == pytest.approx(preds[row], abs=1e-6)
    assert summary["method"] == "tree_shap"
    assert summary["n_features"] == 2


def test_lightgbm_shap_ranks_dominant_feature_first() -> None:
    """In a DGP where log_price drives y, mean |SHAP| must rank it first."""
    frame = _frame(n=120, seed=4)
    y = (6.5 - 2.5 * frame["log_price"] + 0.05 * frame["tpr_share"]).to_numpy()
    X = frame[["log_price", "tpr_share"]]
    model = LGBMRegressor(
        n_estimators=80, num_leaves=12, verbosity=-1, random_state=0
    )
    model.fit(X, y)
    summary = lightgbm_shap_summary(model, X)
    assert summary["mean_abs_shap"][0]["feature"] == "log_price"
