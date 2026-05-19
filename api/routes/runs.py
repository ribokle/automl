"""Run CRUD endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.auth import require_auth
from api.deps import get_run_dir
from api.schemas import CreateRunRequest, RunSummary
from core.orchestrator.runner import execute
from core.orchestrator.state import AgentStatus, RunState, RunStatus

router = APIRouter(prefix="/runs", tags=["runs"], dependencies=[Depends(require_auth)])

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
    runs = sorted(_RUNS.values(), key=lambda r: r.created_at, reverse=True)
    return [
        RunSummary(
            id=r.id,
            status=r.status.value,
            data_path=r.data_path,
            run_dir=r.run_dir,
            created_at=r.created_at.isoformat(),
        )
        for r in runs
    ]


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail="run not found")
    return _RUNS[run_id].model_dump(mode="json")


def get_runs_registry() -> dict[str, RunState]:
    return _RUNS


def rehydrate_runs(run_dir_base: Path) -> dict[str, int]:
    """Reload `state.json` for every run on disk, marking orphans as failed.

    Returns a small counters dict so the caller can log the recovery. Any run
    whose persisted status is `running` or `awaiting_approval` is downgraded
    to `failed` with reason `process_restarted` — the orchestrator task is
    gone with the previous process and cannot be resumed without a job queue.
    """
    loaded = 0
    orphaned = 0
    if not run_dir_base.exists():
        return {"loaded": 0, "orphaned": 0}

    for state_file in run_dir_base.glob("*/state.json"):
        try:
            state = RunState.load(state_file.parent)
        except Exception:
            continue
        if state.status in (RunStatus.running, RunStatus.awaiting_approval):
            state.status = RunStatus.failed
            for agent_result in state.agents.values():
                if agent_result.status in (AgentStatus.running, AgentStatus.awaiting_approval):
                    agent_result.status = AgentStatus.failed
                    agent_result.error = "process_restarted"
            state.options = {**state.options, "recovery_reason": "process_restarted"}
            state.save()
            orphaned += 1
        _RUNS[state.id] = state
        loaded += 1
    return {"loaded": loaded, "orphaned": orphaned}
