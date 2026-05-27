"""Hardening guarantees for the model library + router.

These assert the two contracts that keep the platform robust:
1. The library imports and registers with only base dependencies installed
   (heavy/optional families degrade to "registered but unavailable").
2. The full pipeline runs end-to-end with the router enabled, in dry-run, with
   no heavy optional deps present.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core.agents.modeling as modeling_mod
from core.agents.modeling import ModelingAgent
from core.config import ModelLibrarySettings, RouterSettings, Settings
from core.models.library import registry
from core.orchestrator.state import RunState


def test_registry_nonempty_with_base_deps_only() -> None:
    # Always-available (base-dep) models must register and be available.
    avail = registry.available_keys()
    for key in {"loglog_ols", "semilog_ols", "lightgbm", "ridge", "double_ml", "arimax"}:
        assert key in avail


def test_catalog_shape() -> None:
    rows = registry.catalog()
    assert rows, "catalog is empty"
    keys = {r["key"] for r in rows}
    assert {"xgboost", "catboost", "prophet", "tbats", "gam"} <= keys  # optional ones listed
    for r in rows:
        assert set(r) == {"key", "family", "problem_types", "available", "required_packages"}


def test_optional_models_registered_but_unavailable_without_extras() -> None:
    # Proves the graceful-skip contract: optional deps are catalogued but drop
    # out of the available set when not installed.
    all_keys = set(registry.all_keys())
    for key in {"xgboost", "catboost", "prophet", "tbats", "gam"}:
        assert key in all_keys
        if not registry.get(key).is_available():
            assert key not in registry.available_keys()


def _seed(tmp_path: Path) -> RunState:
    rng = np.random.default_rng(0)

    def mk(ppg: str, beta: float, n: int = 120) -> pd.DataFrame:
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

    feats = pd.concat([mk("P1", -1.4), mk("P2", -2.1)], ignore_index=True)
    run_dir = tmp_path / "run"
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    feats.to_csv(run_dir / "features.csv", index=False)
    (run_dir / "ppg_selection.json").write_text(
        json.dumps([{"ppg_id": "P1", "eligible": True}, {"ppg_id": "P2", "eligible": True}])
    )
    (run_dir / "feature_refine.json").write_text(
        json.dumps({"kept": ["log_price", "log_distribution_acv"]})
    )
    return state


def test_router_modeling_dry_run_completes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        modeling_mod,
        "get_settings",
        lambda: Settings(
            model_library=ModelLibrarySettings(router_enabled=True),
            router=RouterSettings(mode="auto"),  # dry-run -> rules fallback
        ),
    )
    state = _seed(tmp_path)
    asyncio.run(ModelingAgent().run(state))

    results = json.loads((Path(state.run_dir) / "modeling_results.json").read_text())
    assert results["router_enabled"] is True
    assert all(r["winner"] is not None and r["winner"]["sign_ok"] for r in results["per_ppg"])
    assert (Path(state.run_dir) / "router_decision.json").exists()
