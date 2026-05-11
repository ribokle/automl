"""SSE event stream per run."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from api.routes.runs import get_runs_registry
from core.orchestrator.events import bus

router = APIRouter(prefix="/runs", tags=["events"])


@router.get("/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> EventSourceResponse:
    runs = get_runs_registry()
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="run not found")

    queue = bus.subscribe(run_id)

    async def event_gen():
        try:
            # Initial replay: replay events.jsonl so reconnects see history.
            from pathlib import Path

            events_path = Path(runs[run_id].run_dir) / "events.jsonl"
            if events_path.exists():
                for line in events_path.read_text().splitlines():
                    yield {"event": "history", "data": line}

            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield {"event": "message", "data": json.dumps(msg, default=str)}
                    if msg.get("type") == "run_finished":
                        break
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "keepalive"}
        finally:
            bus.unsubscribe(run_id, queue)

    return EventSourceResponse(event_gen())
