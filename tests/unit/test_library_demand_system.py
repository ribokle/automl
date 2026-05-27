"""Cross-price log-log demand system recovers own + cross elasticities."""
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
from core.models.library.base import FitContext
from core.models.result import Capability, ProblemType


def _system_frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wk = pd.date_range("2020-01-01", periods=n, freq="W")
    lp1, lp2, lp3 = (rng.normal(0, 0.2, n) for _ in range(3))
    lu1 = 5 - 1.5 * lp1 + 0.6 * lp2 + rng.normal(0, 0.05, n)  # P2 substitutes for P1
    lu2 = 5 - 2.0 * lp2 + 0.4 * lp1 + rng.normal(0, 0.05, n)
    lu3 = 5 - 1.0 * lp3 + rng.normal(0, 0.05, n)  # independent

    def blk(p: str, lp, lu) -> pd.DataFrame:
        return pd.DataFrame({"ppg_id": p, "week_start": wk, "log_price": lp, "log_units": lu})

    blocks = [blk("P1", lp1, lu1), blk("P2", lp2, lu2), blk("P3", lp3, lu3)]
    return pd.concat(blocks, ignore_index=True)


def test_recovers_own_and_cross_signs() -> None:
    ctx = FitContext(ppg_id="P1", controls=[], problem_type=ProblemType.DEMAND_SYSTEM)
    result = registry.get("crossprice_loglog").fit(_system_frame(), ctx)
    assert result.own_elasticity < 0
    row = result.cross_price["P1"]
    assert row["P2"] > 0.2  # substitute -> positive cross-price elasticity
    assert abs(row["P3"]) < 0.2  # independent -> ~zero
    assert Capability.CROSS_PRICE_MATRIX in result.capabilities


def test_needs_multiple_ppgs() -> None:
    single = _system_frame()
    single = single[single["ppg_id"] == "P1"]
    try:
        registry.get("crossprice_loglog").fit(single, FitContext(ppg_id="P1", controls=[]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a single-PPG demand system")


def test_demand_system_run_writes_cross_price_matrix(tmp_path, monkeypatch) -> None:
    feats = _system_frame()
    state = _seed_demand_run(tmp_path, feats)
    monkeypatch.setattr(
        modeling_mod,
        "get_settings",
        lambda: Settings(
            model_library=ModelLibrarySettings(router_enabled=True),
            router=RouterSettings(default_problem_type="demand_system"),
        ),
    )
    asyncio.run(ModelingAgent().run(state))
    matrix = json.loads((Path(state.run_dir) / "cross_price_matrix.json").read_text())
    assert set(matrix["ppgs"]) == {"P1", "P2", "P3"}
    assert all(matrix["own_elasticity"][p] < 0 for p in matrix["ppgs"])
    assert matrix["matrix"]["P1"]["P2"] > 0.2


def _seed_demand_run(tmp_path: Path, feats: pd.DataFrame):
    from core.orchestrator.state import RunState

    run_dir = tmp_path / "run"
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    feats.to_csv(run_dir / "features.csv", index=False)
    (run_dir / "ppg_selection.json").write_text(
        json.dumps([{"ppg_id": p, "eligible": True} for p in ["P1", "P2", "P3"]])
    )
    (run_dir / "feature_refine.json").write_text(json.dumps({"kept": ["log_price"]}))
    return state
