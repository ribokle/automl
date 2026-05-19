"""Optimization core: predict, continuous, MILP, and the agent.

The strict MILP must pick a ladder rung that respects margin floor +
competitive gap; the soft-constraint fallback must fire when those
constraints can't all be satisfied and report which one was relaxed.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.agents.optimization import OptimizationAgent
from core.optimization.constraints import OptimizationConstraints, PPGOptInputs
from core.optimization.continuous import solve_continuous
from core.optimization.milp import solve_milp
from core.optimization.predict import cell_metrics, predict_units
from core.orchestrator.state import AgentResult, AgentStatus, RunState
from core.simulation.grid import ScenarioGridSpec, simulate_ols_grid


COEFS_ELASTIC = {
    "const": 6.5,
    "log_price": -2.0,
    "tpr_share": 0.6,
    "log_distribution_acv": 0.4,
}


def test_predict_matches_simulator_grid_cell() -> None:
    spec = ScenarioGridSpec(
        price_multipliers=(1.0, 1.05),
        promo_states=(0, 1),
        context={"log_distribution_acv": math.log(85)},
    )
    grid = simulate_ols_grid(COEFS_ELASTIC, base_price=3.0, spec=spec, model_kind="loglog_ols")
    for _, row in grid.iterrows():
        u = predict_units(
            COEFS_ELASTIC,
            base_price=3.0,
            price=float(row["price"]),
            promo=int(row["promo"]),
            model_kind="loglog_ols",
            context={"log_distribution_acv": math.log(85)},
        )
        assert math.isclose(u, float(row["units"]), rel_tol=1e-12)


def test_continuous_picks_lower_bound_for_elastic_demand() -> None:
    """ε = -2: revenue is monotone decreasing in price, so the
    continuous solver must hit the lower bound of the feasible
    multiplier interval."""
    inp = PPGOptInputs(
        ppg_id="P",
        model_kind="loglog_ols",
        coefficients=COEFS_ELASTIC,
        base_price=3.0,
        context={"log_distribution_acv": math.log(85)},
    )
    c = OptimizationConstraints(
        margin_floor_pct=0.0,
        comp_gap_pct=1.0,
        max_decrease=0.20,
        max_increase=0.20,
        objective="revenue",
    )
    res = solve_continuous(inp, c)
    assert res.feasible
    assert math.isclose(res.price_multiplier, 1.0 - c.max_decrease, abs_tol=1e-3)


def test_continuous_infeasible_returns_flag() -> None:
    """Margin floor pushes lower bound above the move guardrail upper bound."""
    inp = PPGOptInputs(
        ppg_id="P",
        model_kind="loglog_ols",
        coefficients=COEFS_ELASTIC,
        base_price=3.0,
        context={},
        competitor_price=2.0,
    )
    c = OptimizationConstraints(
        margin_floor_pct=0.0,
        comp_gap_pct=0.01,
        max_decrease=0.20,
        max_increase=0.20,
    )
    res = solve_continuous(inp, c)
    assert res.feasible is False


def test_milp_respects_ladder_and_margin_floor() -> None:
    inp = PPGOptInputs(
        ppg_id="P",
        model_kind="loglog_ols",
        coefficients=COEFS_ELASTIC,
        base_price=3.0,
        context={"log_distribution_acv": math.log(85)},
    )
    c = OptimizationConstraints(
        price_ladder=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10),
        margin_floor_pct=0.0,
        comp_gap_pct=1.0,
        objective="revenue",
    )
    res = solve_milp(inp, c)
    assert res.feasible_strict is True
    assert res.price_multiplier in c.price_ladder
    floor_price = c.cog_pct * inp.base_price + c.margin_floor_pct * inp.base_price
    assert res.price >= floor_price - 1e-9


def test_milp_relaxes_when_no_cell_feasible() -> None:
    """Force infeasibility: a tight comp gap that doesn't intersect the ladder."""
    inp = PPGOptInputs(
        ppg_id="P",
        model_kind="loglog_ols",
        coefficients=COEFS_ELASTIC,
        base_price=3.0,
        context={"log_distribution_acv": math.log(85)},
        competitor_price=10.0,  # nowhere near base 3.0
    )
    c = OptimizationConstraints(
        price_ladder=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10),
        margin_floor_pct=0.0,
        comp_gap_pct=0.01,
    )
    res = solve_milp(inp, c)
    assert res.feasible_strict is False
    assert res.relaxed is True
    assert any(v["constraint"].startswith("comp_gap") for v in res.binding_violations)


def test_milp_picks_higher_price_for_inelastic_margin() -> None:
    """ε = -0.5 (inelastic), margin objective: optimal multiplier should
    sit at the upper bound of the ladder within the move guardrail."""
    coefs = {"const": 5.0, "log_price": -0.5, "tpr_share": 0.0}
    inp = PPGOptInputs(
        ppg_id="P",
        model_kind="loglog_ols",
        coefficients=coefs,
        base_price=3.0,
        context={},
    )
    c = OptimizationConstraints(
        price_ladder=(0.95, 1.00, 1.05, 1.10, 1.15),
        margin_floor_pct=0.0,
        comp_gap_pct=1.0,
        objective="margin",
    )
    res = solve_milp(inp, c)
    assert res.feasible_strict
    assert res.price_multiplier == max(c.price_ladder)


def test_cell_metrics_margin_zero_when_price_equals_cost() -> None:
    coefs = {"const": 4.0, "log_price": -1.0}
    m = cell_metrics(
        coefs,
        base_price=2.0,
        price=2.0 * 0.55,
        promo=0,
        model_kind="loglog_ols",
        context={},
        cog_pct=0.55,
    )
    assert math.isclose(m["margin"], 0.0, abs_tol=1e-9)


def _toy_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(2)
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
            "competitor_price": 3.0 + rng.normal(0, 0.05, size=n),
        }
    )


def _seed_run(tmp_path: Path, frame: pd.DataFrame, modeling: dict, options: dict | None = None) -> RunState:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir, options=options or {})
    state.run_dir = str(run_dir.resolve())
    state.agents["optimization"] = AgentResult(agent="optimization", status=AgentStatus.pending)
    frame.to_csv(run_dir / "features.csv", index=False)
    (run_dir / "modeling_results.json").write_text(json.dumps(modeling))
    return state


def test_optimization_agent_writes_three_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frame = _toy_frame()
    modeling = {
        "controls_used": ["tpr_share", "log_distribution_acv"],
        "per_ppg": [
            {
                "ppg_id": "PPG_S",
                "winner_model": "loglog_ols",
                "winner": {
                    "model": "loglog_ols",
                    "coefficients": COEFS_ELASTIC,
                },
                "attempts": [],
                "sign_retry_fired": False,
            }
        ],
    }
    state = _seed_run(tmp_path, frame, modeling)
    asyncio.run(OptimizationAgent().run(state))

    run_dir = Path(state.run_dir)
    for name in (
        "optimization_results.json",
        "optimization_table.json",
        "optimization_constraints.json",
    ):
        assert (run_dir / name).exists(), f"{name} should be on disk"

    table = json.loads((run_dir / "optimization_table.json").read_text())
    assert len(table) == 1
    row = table[0]
    assert row["ppg_id"] == "PPG_S"
    assert "price_multiplier" in row
    assert row["base_price"] > 0

    results = json.loads((run_dir / "optimization_results.json").read_text())
    assert results[0]["milp"]["feasible_strict"] is True
    assert results[0]["continuous"]["feasible"] is True


def test_optimization_agent_skips_lightgbm(tmp_path: Path, monkeypatch) -> None:
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
    asyncio.run(OptimizationAgent().run(state))
    out = state.agents["optimization"].outputs
    assert out["n_optimised"] == 0
    assert out["n_skipped"] == 1


def test_optimization_agent_honours_options_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frame = _toy_frame()
    modeling = {
        "controls_used": ["tpr_share", "log_distribution_acv"],
        "per_ppg": [
            {
                "ppg_id": "PPG_S",
                "winner_model": "loglog_ols",
                "winner": {
                    "model": "loglog_ols",
                    "coefficients": COEFS_ELASTIC,
                },
                "attempts": [],
                "sign_retry_fired": False,
            }
        ],
    }
    state = _seed_run(
        tmp_path,
        frame,
        modeling,
        options={
            "optimization": {
                "objective": "margin",
                "price_ladder": [0.95, 1.00, 1.05],
                "margin_floor_pct": 0.02,
            }
        },
    )
    asyncio.run(OptimizationAgent().run(state))
    constraints = json.loads((Path(state.run_dir) / "optimization_constraints.json").read_text())
    assert constraints["objective"] == "margin"
    assert constraints["price_ladder"] == [0.95, 1.00, 1.05]
    assert constraints["margin_floor_pct"] == 0.02

    results = json.loads((Path(state.run_dir) / "optimization_results.json").read_text())
    chosen = results[0]["milp"]["price_multiplier"]
    assert chosen in (0.95, 1.00, 1.05)
