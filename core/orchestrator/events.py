"""In-memory event bus with per-run asyncio queues for SSE fan-out."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any

from core.orchestrator.state import append_event


class EventBus:
    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)

    def subscribe(self, run_id: str) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
        self._queues[run_id].append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        if q in self._queues.get(run_id, []):
            self._queues[run_id].remove(q)

    async def publish(self, run_id: str, run_dir: str | None, event: dict[str, Any]) -> None:
        event = {"ts": datetime.utcnow().isoformat(), "run_id": run_id, **event}
        if run_dir:
            from pathlib import Path

            append_event(Path(run_dir), event)
        for q in list(self._queues.get(run_id, [])):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass


bus = EventBus()
