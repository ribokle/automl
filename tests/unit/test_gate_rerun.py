"""Gate rerun loop: edit constraints, re-execute agent, pause again.

Backs the optimization edit-and-re-solve flow. Covers gate-registry state
machine, the `/rerun` endpoint, and the runner's `_wait_for_gate` loop
that re-executes the agent with merged options.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.agents.base import Agent
from core.orchestrator import runner
from core.orchestrator.gates import RERUNNABLE_AGENTS, gate_registry
from core.orchestrator.state import AGENT_ORDER, AgentResult, AgentStatus, RunState


def test_rerunnable_whitelist_includes_optimization() -> None:
    assert "optimization" in RERUNNABLE_AGENTS


def test_request_rerun_rejects_non_whitelisted_agent() -> None:
    ok = gate_registry.request_rerun("R1", "modeling", {"foo": 1})
    assert ok is False


def test_request_rerun_sets_payload_and_event() -> None:
    gate_registry.drop("R2")
    state = gate_registry.get("R2", "optimization")
    assert state.event.is_set() is False
    ok = gate_registry.request_rerun("R2", "optimization", {"margin_floor_pct": 0.1})
    assert ok is True
    assert state.rerun_payload == {"margin_floor_pct": 0.1}
    assert state.event.is_set() is True


def test_request_rerun_rejected_after_resolve() -> None:
    gate_registry.drop("R3")
    gate_registry.approve("R3", "optimization")
    ok = gate_registry.request_rerun("R3", "optimization", {})
    assert ok is False


def test_reset_arms_fresh_event_and_clears_payload() -> None:
    gate_registry.drop("R4")
    gate_registry.request_rerun("R4", "optimization", {"objective": "margin"})
    gate_registry.reset("R4", "optimization")
    state = gate_registry.get("R4", "optimization")
    assert state.event.is_set() is False
    assert state.rerun_payload is None
    assert state.approved is None


def test_rerun_endpoint_returns_400_for_non_whitelisted_agent(tmp_path: Path, monkeypatch) -> None:
    os.environ["RUN_DIR"] = str(tmp_path)
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    from api.main import app

    client = TestClient(app)
    res = client.post("/runs/abc/rerun?agent=modeling", json={})
    assert res.status_code == 400


def test_rerun_endpoint_queues_payload(tmp_path: Path, monkeypatch) -> None:
    os.environ["RUN_DIR"] = str(tmp_path)
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    from api.main import app

    gate_registry.drop("R5")
    client = TestClient(app)
    res = client.post(
        "/runs/R5/rerun?agent=optimization",
        json={"margin_floor_pct": 0.07},
    )
    assert res.status_code == 200
    state = gate_registry.get("R5", "optimization")
    assert state.rerun_payload == {"margin_floor_pct": 0.07}


def test_rerun_endpoint_rejects_already_resolved_gate(tmp_path: Path, monkeypatch) -> None:
    os.environ["RUN_DIR"] = str(tmp_path)
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    from api.main import app

    gate_registry.drop("R6")
    gate_registry.approve("R6", "optimization")
    client = TestClient(app)
    res = client.post("/runs/R6/rerun?agent=optimization", json={})
    assert res.status_code == 409


class _CountingAgent(Agent):
    """Records how many times it's been re-executed; mutates options in place."""

    name = "optimization"
    runs_observed: list[dict] = []  # noqa: RUF012

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        type(self).runs_observed.append(dict(run.options.get("optimization", {})))
        result.outputs = {"call": len(type(self).runs_observed)}
        result.reasoning = "counting"
        result.confidence = 0.5


def test_runner_loops_through_rerun_then_approves(tmp_path: Path, monkeypatch) -> None:
    """`_wait_for_gate` consumes a rerun, re-executes the agent, then
    pauses again. A second resolution (approve) exits the loop."""
    _CountingAgent.runs_observed = []
    monkeypatch.setitem(runner.REAL_AGENTS, "optimization", _CountingAgent)

    gate_registry.drop("RX")
    run_dir = tmp_path / "RX"
    run_dir.mkdir()
    run = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    run.id = "RX"
    run.gates = {"optimization": True}
    run.agents["optimization"] = AgentResult(agent="optimization", status=AgentStatus.done)
    run.save()

    async def driver() -> bool:
        gate_task = asyncio.create_task(runner._wait_for_gate(run, "optimization"))
        await asyncio.sleep(0.05)
        gate_registry.request_rerun("RX", "optimization", {"margin_floor_pct": 0.09})
        await asyncio.sleep(0.05)
        gate_registry.approve("RX", "optimization")
        return await gate_task

    approved = asyncio.run(driver())
    assert approved is True
    assert len(_CountingAgent.runs_observed) == 1
    assert _CountingAgent.runs_observed[0] == {"margin_floor_pct": 0.09}
    assert run.options["optimization"] == {"margin_floor_pct": 0.09}


def test_runner_rerun_merges_with_existing_options(tmp_path: Path, monkeypatch) -> None:
    """Each rerun layers onto whatever's already in `run.options[agent]`."""
    _CountingAgent.runs_observed = []
    monkeypatch.setitem(runner.REAL_AGENTS, "optimization", _CountingAgent)

    gate_registry.drop("RY")
    run_dir = tmp_path / "RY"
    run_dir.mkdir()
    run = RunState.new(
        data_path=str(tmp_path / "x.csv"),
        run_dir=run_dir,
        options={"optimization": {"objective": "revenue"}},
    )
    run.id = "RY"
    run.gates = {"optimization": True}
    run.agents["optimization"] = AgentResult(agent="optimization", status=AgentStatus.done)
    run.save()

    async def driver() -> None:
        gate_task = asyncio.create_task(runner._wait_for_gate(run, "optimization"))
        await asyncio.sleep(0.05)
        gate_registry.request_rerun("RY", "optimization", {"margin_floor_pct": 0.07})
        await asyncio.sleep(0.05)
        gate_registry.approve("RY", "optimization")
        await gate_task

    asyncio.run(driver())
    assert run.options["optimization"] == {
        "objective": "revenue",
        "margin_floor_pct": 0.07,
    }


def test_rejecting_the_gate_returns_false_without_rerun(tmp_path: Path, monkeypatch) -> None:
    _CountingAgent.runs_observed = []
    monkeypatch.setitem(runner.REAL_AGENTS, "optimization", _CountingAgent)

    gate_registry.drop("RZ")
    run_dir = tmp_path / "RZ"
    run_dir.mkdir()
    run = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    run.id = "RZ"
    run.gates = {"optimization": True}
    run.agents["optimization"] = AgentResult(agent="optimization", status=AgentStatus.done)
    run.save()

    async def driver() -> bool:
        gate_task = asyncio.create_task(runner._wait_for_gate(run, "optimization"))
        await asyncio.sleep(0.05)
        gate_registry.reject("RZ", "optimization")
        return await gate_task

    approved = asyncio.run(driver())
    assert approved is False
    assert _CountingAgent.runs_observed == []
