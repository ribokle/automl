"""LLM-trace capture in the base Agent.

The trace is the audit record for the agentic layer: system prompt + user
prompt + raw response per LLM call, written to
``<agent>_llm_trace.json`` at the end of the agent's run. Disabled with
``LLM_TRACE=false`` for sensitive runs. Dry-run calls (no API key) still
write a trace so the UI can show the deterministic fallback.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.agents.base import Agent
from core.orchestrator.state import AgentResult, AgentStatus, RunState


class _CallingAgent(Agent):
    name = "trace_test_agent"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        await asyncio.to_thread(
            self.call_llm,
            result,
            system="be helpful",
            user="say hi",
            max_tokens=8,
            label="hello",
        )
        result.outputs = {"ok": True}
        result.reasoning = "did a thing"


def _new_state(tmp_path: Path, agent_name: str = "trace_test_agent") -> RunState:
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=tmp_path)
    run_dir = tmp_path / state.id
    run_dir.mkdir()
    state.run_dir = str(run_dir.resolve())
    state.agents[agent_name] = AgentResult(agent=agent_name, status=AgentStatus.pending)
    return state


def test_dry_run_trace_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = _new_state(tmp_path)
    asyncio.run(_CallingAgent().run(state))
    trace_path = Path(state.run_dir) / "trace_test_agent_llm_trace.json"
    assert trace_path.exists(), "trace file should be written when LLM_TRACE is unset"
    blob = json.loads(trace_path.read_text())
    assert blob["agent"] == "trace_test_agent"
    assert len(blob["calls"]) == 1
    call = blob["calls"][0]
    assert call["label"] == "hello"
    assert call["dry_run"] is True
    assert call["system"] == "be helpful"
    assert call["user"] == "say hi"
    assert call["tokens_in"] == 0


def test_llm_trace_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_TRACE", "false")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = _new_state(tmp_path)
    asyncio.run(_CallingAgent().run(state))
    trace_path = Path(state.run_dir) / "trace_test_agent_llm_trace.json"
    assert not trace_path.exists(), "trace file must not be written when LLM_TRACE=false"


def test_failed_agent_still_writes_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the agent calls the LLM then raises, the trace should still be on disk."""
    monkeypatch.delenv("LLM_TRACE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    class _Boom(_CallingAgent):
        name = "trace_test_boom"

        async def _execute(self, run: RunState, result: AgentResult) -> None:
            await super()._execute(run, result)
            raise RuntimeError("intentional")

    state = _new_state(tmp_path, agent_name="trace_test_boom")
    with pytest.raises(RuntimeError):
        asyncio.run(_Boom().run(state))
    trace_path = Path(state.run_dir) / "trace_test_boom_llm_trace.json"
    assert trace_path.exists()
    blob = json.loads(trace_path.read_text())
    assert len(blob["calls"]) == 1
