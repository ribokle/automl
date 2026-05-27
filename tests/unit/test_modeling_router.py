"""Modeling agent with the router enabled (Phase 8c wiring)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core.agents.modeling as modeling_mod
from core.agents.modeling import ModelingAgent, _resolve_config
from core.config import ModelLibrarySettings, RouterSettings, Settings
from core.orchestrator.state import RunState


def _features(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def mk(ppg: str, beta: float, n: int = 140) -> pd.DataFrame:
        lp = rng.normal(0.0, 0.25, n)
        ctrl = rng.normal(0.0, 1.0, n)
        lu = 5.0 + beta * lp + 0.3 * ctrl + rng.normal(0.0, 0.05, n)
        return pd.DataFrame(
            {
                "ppg_id": ppg,
                "log_price": lp,
                "log_units": lu,
                "log_distribution_acv": ctrl,
                "week_start": pd.date_range("2021-01-01", periods=n, freq="W"),
            }
        )

    return pd.concat([mk("P1", -1.4), mk("P2", -2.1)], ignore_index=True)


def _seed(tmp_path: Path) -> RunState:
    run_dir = tmp_path / "run"
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    _features().to_csv(run_dir / "features.csv", index=False)
    (run_dir / "ppg_selection.json").write_text(
        json.dumps([{"ppg_id": "P1", "eligible": True}, {"ppg_id": "P2", "eligible": True}])
    )
    (run_dir / "feature_refine.json").write_text(
        json.dumps({"kept": ["log_price", "log_distribution_acv"]})
    )
    return state


def test_routed_run_writes_router_decision_and_recovers_sign(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        modeling_mod,
        "get_settings",
        lambda: Settings(model_library=ModelLibrarySettings(router_enabled=True)),
    )
    state = _seed(tmp_path)
    asyncio.run(ModelingAgent().run(state))

    run_dir = Path(state.run_dir)
    results = json.loads((run_dir / "modeling_results.json").read_text())
    assert results["router_enabled"] is True
    assert set(results["model_pool"]) >= {"loglog_ols", "ridge", "lasso", "elasticnet", "lightgbm"}
    for row in results["per_ppg"]:
        assert row["winner"] is not None
        assert row["winner"]["sign_ok"]
        assert row["router"]["candidates"]

    decision = json.loads((run_dir / "router_decision.json").read_text())
    assert decision["router_mode"] == "auto"
    assert len(decision["decisions"]) == 2


def test_legacy_path_writes_no_router_decision(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        modeling_mod,
        "get_settings",
        lambda: Settings(model_library=ModelLibrarySettings(router_enabled=False)),
    )
    state = _seed(tmp_path)
    asyncio.run(ModelingAgent().run(state))
    run_dir = Path(state.run_dir)
    assert not (run_dir / "router_decision.json").exists()
    results = json.loads((run_dir / "modeling_results.json").read_text())
    assert results["router_enabled"] is False
    assert results["model_pool"] == ["loglog_ols", "semilog_ols", "lightgbm"]


def test_resolve_config_applies_per_run_overrides(tmp_path) -> None:
    settings = Settings(
        model_library=ModelLibrarySettings(router_enabled=False),
        router=RouterSettings(mode="auto"),
    )
    run = RunState.new(
        data_path="x.csv",
        run_dir=tmp_path / "run",
        options={
            "modeling": {
                "router_enabled": True,
                "enabled_models": ["loglog_ols", "ridge"],
                "mode": "rules",
                "default_problem_type": "forecast",
            }
        },
    )
    lib, rtr = _resolve_config(run, settings)
    assert lib.router_enabled is True
    assert lib.enabled_models == ["loglog_ols", "ridge"]
    assert rtr.mode == "rules"
    assert rtr.default_problem_type == "forecast"
    # global settings object is untouched
    assert settings.model_library.router_enabled is False
