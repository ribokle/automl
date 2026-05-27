"""Panel FE/RE recover the within-entity elasticity (optional: linearmodels)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.agents.modeling as modeling_mod
from core.agents.modeling import ModelingAgent
from core.config import ModelLibrarySettings, RouterSettings, Settings
from core.models.library import registry
from core.models.library.base import FitContext
from core.models.result import ProblemType
from core.orchestrator.state import RunState

_PANEL_KEYS = ["fixed_effects", "random_effects"]


def _panel_frame(
    ppgs=(("P1", -1.7), ("P2", -2.3)), n_stores: int = 6, n_weeks: int = 60, seed: int = 0
):
    rng = np.random.default_rng(seed)
    rows = []
    for ppg, beta in ppgs:
        for s in range(n_stores):
            a = rng.normal(5.0, 0.5)  # store fixed effect
            for w in range(n_weeks):
                lp = rng.normal(0.0, 0.2)
                lu = a + beta * lp + rng.normal(0.0, 0.05)
                rows.append(
                    {
                        "ppg_id": ppg,
                        "grain_unit": f"store_{s}",
                        "week_start": pd.Timestamp("2021-01-01") + pd.Timedelta(weeks=w),
                        "log_price": lp,
                        "log_units": lu,
                    }
                )
    return pd.DataFrame(rows)


def test_panel_models_registered_unavailable_without_dep() -> None:
    keys = set(registry.all_keys())
    assert {"fixed_effects", "random_effects"} <= keys


@pytest.mark.parametrize("key", _PANEL_KEYS)
def test_recovers_within_entity_elasticity(key: str) -> None:
    plugin = registry.get(key)
    if not plugin.is_available():
        pytest.skip("linearmodels not installed")
    frame = _panel_frame(ppgs=(("P1", -1.7),))
    result = plugin.fit(frame, FitContext(ppg_id="P1", controls=[], problem_type=ProblemType.PANEL))
    assert result.own_elasticity < 0
    assert -2.2 < result.own_elasticity < -1.2  # ~ -1.7
    assert result.diagnostics["n_entities"] == 6


def test_panel_needs_multiple_entities() -> None:
    plugin = registry.get("fixed_effects")
    if not plugin.is_available():
        pytest.skip("linearmodels not installed")
    frame = _panel_frame(ppgs=(("P1", -1.7),), n_stores=1)
    with pytest.raises(ValueError):
        plugin.fit(frame, FitContext(ppg_id="P1", controls=[]))


def test_panel_run_writes_panel_elasticity(tmp_path, monkeypatch) -> None:
    if not registry.get("fixed_effects").is_available():
        pytest.skip("linearmodels not installed")
    feats = _panel_frame()
    run_dir = tmp_path / "run"
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    feats.to_csv(run_dir / "features.csv", index=False)
    (run_dir / "ppg_selection.json").write_text(
        json.dumps([{"ppg_id": "P1", "eligible": True}, {"ppg_id": "P2", "eligible": True}])
    )
    (run_dir / "feature_refine.json").write_text(json.dumps({"kept": ["log_price"]}))
    monkeypatch.setattr(
        modeling_mod,
        "get_settings",
        lambda: Settings(
            modelling_grain="store_ppg_week",
            model_library=ModelLibrarySettings(router_enabled=True),
            router=RouterSettings(default_problem_type="panel"),
        ),
    )
    asyncio.run(ModelingAgent().run(state))
    panel = json.loads((Path(state.run_dir) / "panel_elasticity.json").read_text())
    assert panel["n_ppg"] == 2
    assert all(e["own_elasticity"] < 0 for e in panel["per_ppg"])
    assert all(e["n_entities"] == 6 for e in panel["per_ppg"])
