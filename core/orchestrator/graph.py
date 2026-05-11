"""DAG declaration: which agent runs after which.

In Phase 0 every node is a stub that emits a mocked result. Subsequent phases
replace each stub with the real agent implementation.
"""
from __future__ import annotations

from core.orchestrator.state import AGENT_ORDER

# Sequential pipeline; future phases may parallelise (e.g., EDA fan-out).
EDGES: list[tuple[str, str]] = list(zip(AGENT_ORDER, AGENT_ORDER[1:]))


def topo_order() -> list[str]:
    return list(AGENT_ORDER)
