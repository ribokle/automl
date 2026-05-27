"""Modeling agent in FORECAST mode writes a forecasts.json artifact."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core.agents.modeling as modeling_mod
from core.agents.modeling import ModelingAgent
from core.config import ModelLibrarySettings, RouterSettings, Settings
from core.orchestrator.state import RunState


def _seed(tmp_path: Path) -> RunState:
    rng = np.random.default_rng(0)

    def mk(ppg: str, beta: float, n: int = 130) -> pd.DataFrame:
        lp = rng.normal(0.0, 0.2, n)
        season = np.sin(np.arange(n) * 2 * np.pi / 52)
        lu = 5.0 + beta * lp + 0.4 * season + rng.normal(0.0, 0.05, n)
        return pd.DataFrame(
            {
                "ppg_id": ppg,
                "log_price": lp,
                "log_units": lu,
                "week_start": pd.date_range("2020-01-01", periods=n, freq="W"),
            }
        )

    feats = pd.concat([mk("P1", -1.5), mk("P2", -2.0)], ignore_index=True)
    run_dir = tmp_path / "run"
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    feats.to_csv(run_dir / "features.csv", index=False)
    (run_dir / "ppg_selection.json").write_text(
        json.dumps([{"ppg_id": "P1", "eligible": True}, {"ppg_id": "P2", "eligible": True}])
    )
    (run_dir / "feature_refine.json").write_text(json.dumps({"kept": ["log_price"]}))
    return state


def test_forecast_mode_writes_forecasts(tmp_path, monkeypatch) -> None:
    settings = Settings(
        model_library=ModelLibrarySettings(router_enabled=True),
        router=RouterSettings(default_problem_type="forecast"),
    )
    monkeypatch.setattr(modeling_mod, "get_settings", lambda: settings)
    state = _seed(tmp_path)
    asyncio.run(ModelingAgent().run(state))

    fc = json.loads((Path(state.run_dir) / "forecasts.json").read_text())
    assert fc["problem_type"] == "forecast"
    assert fc["n_ppg"] == 2
    for entry in fc["per_ppg"]:
        assert entry["forecast"]["horizon"] > 0
        assert len(entry["forecast"]["mean"]) == entry["forecast"]["horizon"]
        assert entry["test_wape"] is not None
