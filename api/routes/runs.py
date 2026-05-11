"""Run CRUD endpoints."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.deps import get_run_dir
from api.schemas import CreateRunRequest, RunSummary
from core.orchestrator.runner import execute
from core.orchestrator.state import RunState

router = APIRouter(prefix="/runs", tags=["runs"])

# In-memory registry for Phase 0; Phase 1 swaps in SQLModel persistence.
_RUNS: dict[str, RunState] = {}


@router.post("", response_model=RunSummary)
async def create_run(
    req: CreateRunRequest,
    background: BackgroundTasks,
    run_dir_base: Path = Depends(get_run_dir),
) -> RunSummary:
    data_path = Path(req.data_path)
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"data_path not found: {data_path}")

    state = RunState.new(
        data_path=str(data_path.resolve()),
        run_dir=run_dir_base / "auto",
        options={"label": req.label} if req.label else {},
    )
    # Place under runs/<id>/ for a real per-run directory.
    state_run_dir = run_dir_base / state.id
    state_run_dir.mkdir(parents=True, exist_ok=True)
    state.run_dir = str(state_run_dir.resolve())
    state.duckdb_path = str((state_run_dir / "warehouse.duckdb").resolve())
    state.save()
    _RUNS[state.id] = state

    background.add_task(_run_in_background, state, req.gates_enabled)

    return RunSummary(
        id=state.id,
        status=state.status.value,
        data_path=state.data_path,
        run_dir=state.run_dir,
        created_at=state.created_at.isoformat(),
    )


async def _run_in_background(state: RunState, gates_enabled: bool) -> None:
    await execute(state, gates_enabled=gates_enabled)


@router.get("", response_model=list[RunSummary])
async def list_runs() -> list[RunSummary]:
    return [
        RunSummary(
            id=r.id,
            status=r.status.value,
            data_path=r.data_path,
            run_dir=r.run_dir,
            created_at=r.created_at.isoformat(),
        )
        for r in _RUNS.values()
    ]


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail="run not found")
    return _RUNS[run_id].model_dump(mode="json")


def get_runs_registry() -> dict[str, RunState]:
    return _RUNS
