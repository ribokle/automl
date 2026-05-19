"""Decomposition core + agent.

Phase 4 verification anchor: per-row decomposition reconciles to the
model's prediction exactly (base + Σ due-to-feature == predicted), and
the per-PPG aggregate reports a reconciliation error < 1e-6 on the
synthetic panel.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.agents.decomposition import DecompositionAgent
from core.decomp.due_to import (
    aggregate_to_groups,
    decompose_ols_frame,
    summarise_ppg,
)
from core.decomp.groups import FEATURE_TO_GROUP, GROUP_ORDER
from core.orchestrator.state import AgentResult, AgentStatus, RunState


def _toy_frame(seed: int = 3, n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_base_price = np.log(3.0) * np.ones(n)
    price_mult = 0.8 + 0.4 * rng.random(n)
    log_price = log_base_price + np.log(price_mult)
    tpr_share = rng.binomial(1, 0.3, size=n).astype(float)
    log_acv = np.log(70 + 20 * rng.random(n))
    log_units = (
        6.5 - 2.0 * log_price + 0.6 * tpr_share + 0.4 * (log_acv - log_acv.mean())
        + rng.normal(0, 0.05, size=n)
    )
    return pd.DataFrame(
        {
            "ppg_id": "PPG_T",
            "week_start": pd.date_range("2024-01-01", periods=n, freq="W").astype(str),
            "log_units": log_units,
            "log_price": log_price,
            "log_base_price": log_base_price,
            "tpr_share": tpr_share,
            "log_distribution_acv": log_acv,
        }
    )


def test_decomposition_reconciles_to_predicted_per_row() -> None:
    frame = _toy_frame()
    coefs = {
        "const": 6.5,
        "log_price": -2.0,
        "tpr_share": 0.6,
        "log_distribution_acv": 0.4,
    }
    weekly = decompose_ols_frame(frame, coefs)
    due_cols = [c for c in weekly.columns if c.startswith("due_")]
    reconstructed = weekly["base"] + weekly[due_cols].sum(axis=1)
    diff = (reconstructed - weekly["predicted"]).abs()
    assert diff.max() < 1e-9, (
        f"reconstructed predicted != predicted; max diff = {diff.max()}"
    )


def test_residual_equals_observed_minus_predicted() -> None:
    frame = _toy_frame()
    coefs = {
        "const": 6.5,
        "log_price": -2.0,
        "tpr_share": 0.6,
        "log_distribution_acv": 0.4,
    }
    weekly = decompose_ols_frame(frame, coefs)
    diff = (weekly["observed"] - (weekly["predicted"] + weekly["residual"])).abs()
    assert diff.max() < 1e-9


def test_group_aggregation_matches_per_feature_sum() -> None:
    frame = _toy_frame()
    coefs = {
        "const": 6.5,
        "log_price": -2.0,
        "tpr_share": 0.6,
        "log_distribution_acv": 0.4,
    }
    weekly = decompose_ols_frame(frame, coefs)
    features = ["log_price", "tpr_share", "log_distribution_acv"]
    weekly = aggregate_to_groups(weekly, features, FEATURE_TO_GROUP)
    # log_price is in 'price', tpr_share in 'promo', log_distribution_acv in 'distribution'.
    assert np.allclose(weekly["due_group_price"], weekly["due_log_price"])
    assert np.allclose(weekly["due_group_promo"], weekly["due_tpr_share"])
    assert np.allclose(weekly["due_group_distribution"], weekly["due_log_distribution_acv"])


def test_zero_lift_when_all_features_at_reference() -> None:
    """If every feature equals its reference value, lift should be zero and
    due-tos should all be zero (no spurious attribution)."""
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
    weekly = decompose_ols_frame(frame, coefs)
    assert np.allclose(weekly["lift"], 0.0)
    for col in (c for c in weekly.columns if c.startswith("due_")):
        assert np.allclose(weekly[col], 0.0)


def test_summary_reconciliation_is_negligible() -> None:
    frame = _toy_frame()
    coefs = {
        "const": 6.5,
        "log_price": -2.0,
        "tpr_share": 0.6,
        "log_distribution_acv": 0.4,
    }
    weekly = decompose_ols_frame(frame, coefs)
    features = ["log_price", "tpr_share", "log_distribution_acv"]
    summary = summarise_ppg(weekly, features, FEATURE_TO_GROUP)
    assert abs(summary["reconciliation_pct_error"]) < 1e-9


def _seed_run(tmp_path: Path, frame: pd.DataFrame, modeling: dict) -> RunState:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    state.run_dir = str(run_dir.resolve())
    state.agents["decomposition"] = AgentResult(
        agent="decomposition", status=AgentStatus.pending
    )
    frame.to_csv(run_dir / "features.csv", index=False)
    (run_dir / "modeling_results.json").write_text(json.dumps(modeling))
    return state


def test_decomposition_agent_writes_three_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frame = _toy_frame()
    coefs = {
        "const": 6.5,
        "log_price": -2.0,
        "tpr_share": 0.6,
        "log_distribution_acv": 0.4,
    }
    modeling = {
        "controls_used": ["tpr_share", "log_distribution_acv"],
        "per_ppg": [
            {
                "ppg_id": "PPG_T",
                "winner_model": "loglog_ols",
                "winner": {
                    "ppg_id": "PPG_T",
                    "model": "loglog_ols",
                    "coefficients": coefs,
                    "controls": ["tpr_share", "log_distribution_acv"],
                    "own_elasticity": -2.0,
                },
                "attempts": [],
                "sign_retry_fired": False,
                "n_train": 48,
                "n_test": 12,
            }
        ],
    }
    state = _seed_run(tmp_path, frame, modeling)
    asyncio.run(DecompositionAgent().run(state))

    run_dir = Path(state.run_dir)
    for name in (
        "decomposition_per_ppg_week.json",
        "decomposition_summary.json",
        "decomposition_table.json",
    ):
        assert (run_dir / name).exists(), f"{name} should be on disk"

    summary = json.loads((run_dir / "decomposition_summary.json").read_text())
    assert len(summary) == 1
    assert abs(summary[0]["reconciliation_pct_error"]) < 1e-6

    table = json.loads((run_dir / "decomposition_table.json").read_text())
    groups_in_table = {row["group"] for row in table}
    assert "price" in groups_in_table
    assert "promo" in groups_in_table


def test_decomposition_agent_skips_lightgbm_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LightGBM-winning PPGs should be recorded as skipped, not crash."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frame = _toy_frame()
    modeling = {
        "controls_used": ["tpr_share"],
        "per_ppg": [
            {
                "ppg_id": "PPG_T",
                "winner_model": "lightgbm",
                "winner": {"model": "lightgbm", "coefficients": {}},
                "attempts": [],
                "sign_retry_fired": False,
            }
        ],
    }
    state = _seed_run(tmp_path, frame, modeling)
    asyncio.run(DecompositionAgent().run(state))
    result = state.agents["decomposition"]
    assert result.outputs["n_decomposed"] == 0
    assert result.outputs["n_skipped"] == 1
