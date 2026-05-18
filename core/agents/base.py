"""Agent base class.

Each agent owns one node of the pipeline DAG. The base class handles the
standard lifecycle (status transitions, event emission, error trapping)
and delegates the real work to `_execute`.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from core.llm.client import AnthropicClient, LLMResponse
from core.llm.routing import model_for
from core.orchestrator.events import bus
from core.orchestrator.state import AgentResult, AgentStatus, ArtifactRef, RunState


def _trace_enabled() -> bool:
    return os.environ.get("LLM_TRACE", "true").lower() not in ("0", "false", "no", "off")


class Agent:
    name: str = ""

    def __init__(self, llm: AnthropicClient | None = None) -> None:
        if not self.name:
            raise ValueError(f"{type(self).__name__}.name must be set")
        self.llm = llm or AnthropicClient()
        self._llm_calls: list[dict[str, Any]] = []

    async def emit(self, run: RunState, event_type: str, payload: dict[str, Any] | None = None) -> None:
        await bus.publish(run.id, run.run_dir, {"type": event_type, "agent": self.name, **(payload or {})})

    def call_llm(
        self,
        result: AgentResult,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        label: str = "call",
    ) -> LLMResponse:
        """One-shot LLM call with per-agent model routing and usage accounting.

        Captures system / user / response / model / dry-run state into the
        per-agent trace buffer. Written to ``<agent>_llm_trace.json`` at the
        end of the run unless ``LLM_TRACE=false``.
        """
        resp = self.llm.message(
            model=model_for(self.name),
            system=system,
            user=user,
            max_tokens=max_tokens,
        )
        result.tokens_in += resp.tokens_in
        result.tokens_out += resp.tokens_out
        result.cost_usd += resp.cost_usd
        self._llm_calls.append(
            {
                "label": label,
                "model": resp.model,
                "system": system,
                "user": user,
                "response": resp.text,
                "tokens_in": resp.tokens_in,
                "tokens_out": resp.tokens_out,
                "cost_usd": resp.cost_usd,
                "dry_run": bool(resp.raw.get("dry_run", False)),
                "ts": datetime.utcnow().isoformat(),
            }
        )
        return resp

    def _write_llm_trace(self, run: RunState, result: AgentResult) -> None:
        if not self._llm_calls or not _trace_enabled():
            return
        trace_path = Path(run.run_dir) / f"{self.name}_llm_trace.json"
        trace_blob = {"agent": self.name, "calls": list(self._llm_calls)}
        trace_path.write_text(json.dumps(trace_blob, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(trace_path),
                mime="application/json",
                agent=self.name,
                name=trace_path.name,
            )
        )

    async def run(self, run: RunState) -> AgentResult:
        self._llm_calls = []
        result = run.agents[self.name]
        result.status = AgentStatus.running
        result.started_at = datetime.utcnow()
        await self.emit(run, "agent_started")
        try:
            await self._execute(run, result)
        except Exception as exc:  # noqa: BLE001
            result.status = AgentStatus.failed
            result.error = str(exc)
            result.finished_at = datetime.utcnow()
            self._write_llm_trace(run, result)
            await self.emit(run, "agent_failed", {"error": str(exc)})
            raise
        result.finished_at = datetime.utcnow()
        if result.status == AgentStatus.running:
            result.status = AgentStatus.done
        self._write_llm_trace(run, result)
        await self.emit(
            run,
            "agent_finished",
            {"status": result.status.value, "outputs": result.outputs},
        )
        return result

    async def _execute(self, run: RunState, result: AgentResult) -> None:  # pragma: no cover
        raise NotImplementedError


class StubAgent(Agent):
    """Phase 0 placeholder used for stages that haven't been implemented yet."""

    def __init__(self, name: str, llm: AnthropicClient | None = None) -> None:
        self.name = name
        super().__init__(llm=llm)

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        import asyncio

        await asyncio.sleep(0.02)
        result.outputs = {"mocked": True}
        result.reasoning = f"[stub] {self.name} will be implemented in a later phase."
        result.confidence = 0.0
