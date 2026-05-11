from __future__ import annotations

from pathlib import Path

from core.orchestrator.state import AGENT_ORDER, RunState, RunStatus


def test_run_state_roundtrip(tmp_path: Path):
    state = RunState.new(data_path="data/synthetic.csv", run_dir=tmp_path)
    assert set(state.agents.keys()) == set(AGENT_ORDER)
    assert state.status == RunStatus.pending
    path = state.save()
    loaded = RunState.load(path.parent)
    assert loaded.id == state.id
    assert loaded.data_path == state.data_path
