"""On-startup persistence + orphan recovery for the runs registry."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.routes.runs import _RUNS, rehydrate_runs
from core.orchestrator.state import AgentResult, AgentStatus, RunState, RunStatus


def _write_run(base: Path, run_id: str, status: RunStatus, agent_status: AgentStatus) -> Path:
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(
        id=run_id,
        data_path=str(base / "input.csv"),
        duckdb_path=str(run_dir / "warehouse.duckdb"),
        run_dir=str(run_dir),
        status=status,
        agents={"ingestion": AgentResult(agent="ingestion", status=agent_status)},
    )
    state.save()
    return run_dir


def test_rehydrate_loads_persisted_runs(tmp_path: Path) -> None:
    _RUNS.clear()
    _write_run(tmp_path, "completed_run", RunStatus.completed, AgentStatus.done)
    _write_run(tmp_path, "failed_run", RunStatus.failed, AgentStatus.failed)

    counters = rehydrate_runs(tmp_path)

    assert counters == {"loaded": 2, "orphaned": 0}
    assert {r.id for r in _RUNS.values()} == {"completed_run", "failed_run"}
    assert _RUNS["completed_run"].status is RunStatus.completed


def test_orphans_are_marked_failed(tmp_path: Path) -> None:
    _RUNS.clear()
    run_dir = _write_run(tmp_path, "orphan", RunStatus.running, AgentStatus.running)
    _write_run(tmp_path, "pending_gate", RunStatus.awaiting_approval, AgentStatus.awaiting_approval)

    counters = rehydrate_runs(tmp_path)

    assert counters["orphaned"] == 2
    orphan = _RUNS["orphan"]
    assert orphan.status is RunStatus.failed
    assert orphan.agents["ingestion"].status is AgentStatus.failed
    assert orphan.agents["ingestion"].error == "process_restarted"
    assert orphan.options["recovery_reason"] == "process_restarted"

    persisted = RunState.load(run_dir)
    assert persisted.status is RunStatus.failed


def test_lifespan_rehydrates_on_app_startup(tmp_path: Path) -> None:
    _RUNS.clear()
    _write_run(tmp_path, "from_disk", RunStatus.completed, AgentStatus.done)
    os.environ["RUN_DIR"] = str(tmp_path)
    os.environ.pop("API_AUTH_TOKEN", None)

    from api.main import app

    with TestClient(app) as client:
        res = client.get("/runs")
        assert res.status_code == 200
        ids = [r["id"] for r in res.json()]
        assert "from_disk" in ids


def test_corrupt_state_json_is_skipped(tmp_path: Path) -> None:
    _RUNS.clear()
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "state.json").write_text("{not json}")
    _write_run(tmp_path, "good", RunStatus.completed, AgentStatus.done)

    counters = rehydrate_runs(tmp_path)

    assert counters["loaded"] == 1
    assert "good" in _RUNS
    assert "bad" not in _RUNS
