"""Scenario-grid simulator + simulation agent.

The closed-form sweep should monotone the response: for a price-sensitive
PPG, raising price should monotone decrease units. Revenue follows an
inverted-U with a maximum near where |ε| ≈ 1 (the elasticity sweet
spot). Tests pin both behaviours.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.agents.simulation import SimulationAgent
from core.orchestrator.state import AgentResult, AgentStatus, RunState
from core.simulation.grid import (
    DEFAULT_PRICE_MULTIPLIERS,
    DEFAULT_PROMO_STATES,
    ScenarioGridSpec,
    grid_summary,
    simulate_ols_grid,
)


def test_loglog_units_monotone_decreasing_in_price() -> None:
    """For ε = -2, holding promo and ACV constant, doubling price should
    cut units by a factor of 4. We check monotonicity across the default grid."""
    coefs = {
        "const": 6.5,
        "log_price": -2.0,
        "tpr_share": 0.6,
        "log_distribution_acv": 0.4,
    }
    spec = ScenarioGridSpec(
        promo_states=(0,),
        context={"log_distribution_acv": np.log(85)},
    )
    grid = simulate_ols_grid(coefs, base_price=3.0, spec=spec, model_kind="loglog_ols")
    by_price = grid.sort_values("price")
    units = by_price["units"].to_numpy()
    assert np.all(np.diff(units) < 0), "units must strictly decrease as price rises"


def test_loglog_revenue_peaks_for_elastic_demand() -> None:
    """ε = -2 (elastic). Revenue = p · q ∝ p · p^(-2) = p^(-1), so the
    revenue-maximising price in an unconstrained sweep is the *lowest*
    multiplier on the grid (not a vague interior optimum)."""
    coefs = {"const": 6.5, "log_price": -2.0}
    spec = ScenarioGridSpec(promo_states=(0,))
    grid = simulate_ols_grid(coefs, base_price=3.0, spec=spec, model_kind="loglog_ols")
    summary = grid_summary(grid)
    assert summary["best_revenue"]["price_multiplier"] == min(DEFAULT_PRICE_MULTIPLIERS)


def test_promo_lifts_units() -> None:
    coefs = {"const": 6.5, "log_price": -2.0, "tpr_share": 0.6}
    spec = ScenarioGridSpec(price_multipliers=(1.0,), promo_states=(0, 1))
    grid = simulate_ols_grid(coefs, base_price=3.0, spec=spec, model_kind="loglog_ols")
    promo_off = grid.loc[grid["promo"] == 0, "units"].iloc[0]
    promo_on = grid.loc[grid["promo"] == 1, "units"].iloc[0]
    assert promo_on > promo_off, "TPR coefficient is positive; promo must lift units"


def test_semilog_grid_shape() -> None:
    coefs = {"const": 8.0, "price": -0.5, "tpr_share": 0.5}
    spec = ScenarioGridSpec()
    grid = simulate_ols_grid(coefs, base_price=4.0, spec=spec, model_kind="semilog_ols")
    assert len(grid) == len(DEFAULT_PRICE_MULTIPLIERS) * len(DEFAULT_PROMO_STATES)
    assert set(grid.columns) >= {"price", "promo", "units", "revenue", "margin"}


def _seed_run(tmp_path: Path, frame: pd.DataFrame, modeling: dict) -> RunState:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    state.run_dir = str(run_dir.resolve())
    state.agents["simulation"] = AgentResult(
        agent="simulation", status=AgentStatus.pending
    )
    frame.to_csv(run_dir / "features.csv", index=False)
    (run_dir / "modeling_results.json").write_text(json.dumps(modeling))
    return state


def _toy_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    log_base_price = np.log(3.0) * np.ones(n)
    log_price = log_base_price + np.log(0.8 + 0.4 * rng.random(n))
    return pd.DataFrame(
        {
            "ppg_id": "PPG_S",
            "week_start": pd.date_range("2024-01-01", periods=n, freq="W").astype(str),
            "log_units": 6.5 - 2.0 * log_price + rng.normal(0, 0.05, size=n),
            "log_price": log_price,
            "log_base_price": log_base_price,
            "tpr_share": rng.binomial(1, 0.3, size=n).astype(float),
            "log_distribution_acv": np.log(70 + 20 * rng.random(n)),
        }
    )


def test_simulation_agent_writes_three_artifacts(
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
                "ppg_id": "PPG_S",
                "winner_model": "loglog_ols",
                "winner": {
                    "model": "loglog_ols",
                    "coefficients": coefs,
                },
                "attempts": [],
                "sign_retry_fired": False,
            }
        ],
    }
    state = _seed_run(tmp_path, frame, modeling)
    asyncio.run(SimulationAgent().run(state))

    run_dir = Path(state.run_dir)
    for name in (
        "simulation_grid.json",
        "simulation_summary.json",
        "simulation_table.json",
    ):
        assert (run_dir / name).exists(), f"{name} should be on disk"

    table = json.loads((run_dir / "simulation_table.json").read_text())
    objectives = {row["objective"] for row in table}
    assert objectives == {"revenue", "margin"}


def test_simulation_agent_skips_lightgbm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frame = _toy_frame()
    modeling = {
        "controls_used": [],
        "per_ppg": [
            {
                "ppg_id": "PPG_S",
                "winner_model": "lightgbm",
                "winner": {"model": "lightgbm", "coefficients": {}},
                "attempts": [],
                "sign_retry_fired": False,
            }
        ],
    }
    state = _seed_run(tmp_path, frame, modeling)
    asyncio.run(SimulationAgent().run(state))
    out = state.agents["simulation"].outputs
    assert out["n_simulated"] == 0
    assert out["n_skipped"] == 1
