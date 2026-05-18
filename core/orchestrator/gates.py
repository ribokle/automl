"""Approval-gate definitions + runtime registry.

A gate is a stage name where the runner must pause and wait for human approval
before continuing. Gates default on `ppg_mapping`, `modeling`, and `optimization`.
The CLI / API can disable all gates per-run, in which case the runner skips the
wait entirely.

The runtime side keeps one `asyncio.Event` per (run_id, agent_name) so the
FastAPI `/approve` and `/reject` endpoints can release the orchestrator from
another task. State is in-memory and per-process - good enough for the dev
server; a persistent setup belongs in Phase 6.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

DEFAULT_GATES: dict[str, bool] = {
    "ppg_mapping": True,
    "modeling": True,
    "optimization": True,
}


@dataclass
class GateState:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    approved: bool | None = None  # None=pending, True=approved, False=rejected


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

    def drop(self, run_id: str) -> None:
        for key in [k for k in self._gates if k[0] == run_id]:
            self._gates.pop(key, None)


gate_registry = GateRegistry()
