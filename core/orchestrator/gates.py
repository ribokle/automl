"""Approval-gate definitions + runtime registry.

A gate is a stage name where the runner must pause and wait for human approval
before continuing. Gates default on `ppg_mapping`, `modeling`, and `optimization`.
The CLI / API can disable all gates per-run, in which case the runner skips the
wait entirely.

The runtime side keeps one `asyncio.Event` per (run_id, agent_name) so the
FastAPI `/approve` and `/reject` endpoints can release the orchestrator from
another task. State is in-memory and per-process - good enough for the dev
server; a persistent setup belongs in Phase 6.

Some gates also support a **rerun** resolution: the user can submit new
``run.options`` overrides via `/rerun`, which causes the runner to re-execute
that agent with the new options and then pause again at the same gate. This
backs the optimization constraint-editor loop (Phase 5a/5b): solve with
defaults, edit the ladder / margin floor / comp gap in the UI, re-solve, and
finally approve.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

DEFAULT_GATES: dict[str, bool] = {
    "ppg_mapping": True,
    "modeling": True,
    "optimization": True,
}


# Agents whose `/rerun` endpoint is allowed to mutate `run.options` and trigger
# a re-execution of the agent. Keep this whitelist tight — most agents are not
# safe to re-run in isolation (their outputs feed every downstream stage).
RERUNNABLE_AGENTS: frozenset[str] = frozenset({"optimization"})


@dataclass
class GateState:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    approved: bool | None = None  # None=pending, True=approved, False=rejected
    rerun_payload: dict[str, Any] | None = None


class GateRegistry:
    def __init__(self) -> None:
        self._gates: dict[tuple[str, str], GateState] = {}

    def get(self, run_id: str, agent: str) -> GateState:
        key = (run_id, agent)
        if key not in self._gates:
            self._gates[key] = GateState()
        return self._gates[key]

    def approve(self, run_id: str, agent: str) -> bool:
        state = self.get(run_id, agent)
        state.approved = True
        state.event.set()
        return True

    def reject(self, run_id: str, agent: str) -> bool:
        state = self.get(run_id, agent)
        state.approved = False
        state.event.set()
        return True

    def request_rerun(self, run_id: str, agent: str, payload: dict[str, Any]) -> bool:
        """Signal the runner to re-execute the agent with new options.

        The runner consumes ``rerun_payload`` once, merges it into
        ``run.options``, re-executes the agent, then re-arms the gate for
        another approval cycle. Returns False if the agent isn't on the
        rerunnable whitelist or the gate is already resolved.
        """
        if agent not in RERUNNABLE_AGENTS:
            return False
        state = self.get(run_id, agent)
        if state.approved is not None:
            return False
        state.rerun_payload = dict(payload)
        state.event.set()
        return True

    def reset(self, run_id: str, agent: str) -> None:
        """Re-arm the gate after a rerun: fresh event, clear payload."""
        state = self.get(run_id, agent)
        state.event = asyncio.Event()
        state.rerun_payload = None
        state.approved = None

    def drop(self, run_id: str) -> None:
        for key in [k for k in self._gates if k[0] == run_id]:
            self._gates.pop(key, None)


gate_registry = GateRegistry()
