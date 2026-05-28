"""Run CRUD endpoints."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.auth import require_auth
from api.deps import get_run_dir
from api.schemas import CreateRunRequest, RunSummary
from core.orchestrator.gates import gate_registry
from core.orchestrator.runner import execute
from core.orchestrator.state import AgentStatus, RunState, RunStatus

router = APIRouter(prefix="/runs", tags=["runs"], dependencies=[Depends(require_auth)])

_RUNS: dict[str, RunState] = {}


def _is_archived(state: RunState) -> bool:
    return bool(state.options.get("archived", False))


def _summary(state: RunState) -> RunSummary:
    return RunSummary(
        id=state.id,
        status=state.status.value,
        data_path=state.data_path,
        run_dir=state.run_dir,
        created_at=state.created_at.isoformat(),
        archived=_is_archived(state),
    )


@router.post("", response_model=RunSummary)
async def create_run(
    req: CreateRunRequest,
    background: BackgroundTasks,
    run_dir_base: Path = Depends(get_run_dir),
) -> RunSummary:
    data_path = Path(req.data_path)
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"data_path not found: {data_path}")

    options: dict[str, Any] = {"agent_mode": req.agent_mode}
    if req.label:
        options["label"] = req.label
    if req.grain_gate_required:
        options["grain_gate_required"] = True
    state = RunState.new(
        data_path=str(data_path.resolve()),
        run_dir=run_dir_base / "auto",
        options=options,
    )
    state_run_dir = run_dir_base / state.id
    state_run_dir.mkdir(parents=True, exist_ok=True)
    state.run_dir = str(state_run_dir.resolve())
    state.duckdb_path = str((state_run_dir / "warehouse.duckdb").resolve())
    state.save()
    _RUNS[state.id] = state

    background.add_task(
        _run_in_background,
        state,
        req.gates_enabled,
        req.agent_mode,
        req.grain_gate_required,
    )

    return _summary(state)


async def _run_in_background(
    state: RunState,
    gates_enabled: bool,
    agent_mode: bool,
    grain_gate_required: bool,
) -> None:
    await execute(
        state,
        gates_enabled=gates_enabled,
        agent_mode=agent_mode,
        grain_gate_required=grain_gate_required,
    )


@router.get("", response_model=list[RunSummary])
async def list_runs(archived: bool = False) -> list[RunSummary]:
    runs = sorted(_RUNS.values(), key=lambda r: r.created_at, reverse=True)
    return [_summary(r) for r in runs if _is_archived(r) == archived]


@router.get("/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail="run not found")
    return _RUNS[run_id].model_dump(mode="json")


@router.post("/{run_id}/archive", response_model=RunSummary)
async def archive_run(run_id: str) -> RunSummary:
    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail="run not found")
    state = _RUNS[run_id]
    if state.status in (RunStatus.running, RunStatus.awaiting_approval):
        raise HTTPException(status_code=409, detail="cannot archive an active run")
    state.options = {**state.options, "archived": True}
    state.save()
    return _summary(state)


@router.post("/{run_id}/unarchive", response_model=RunSummary)
async def unarchive_run(run_id: str) -> RunSummary:
    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail="run not found")
    state = _RUNS[run_id]
    state.options = {**state.options, "archived": False}
    state.save()
    return _summary(state)


@router.delete("/{run_id}")
async def delete_run(
    run_id: str,
    run_dir_base: Path = Depends(get_run_dir),
) -> dict[str, Any]:
    if run_id not in _RUNS:
        raise HTTPException(status_code=404, detail="run not found")
    state = _RUNS[run_id]
    if not _is_archived(state):
        raise HTTPException(status_code=409, detail="archive the run before deleting")
    target = Path(state.run_dir).resolve()
    try:
        target.relative_to(run_dir_base.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="run_dir outside base") from None
    if target.exists():
        shutil.rmtree(target)
    _RUNS.pop(run_id, None)
    gate_registry.drop(run_id)
    return {"ok": True, "deleted": run_id}


def get_runs_registry() -> dict[str, RunState]:
    return _RUNS


def rehydrate_runs(run_dir_base: Path) -> dict[str, int]:
    """Reload `state.json` for every run on disk, marking orphans as failed.

    Returns a small counters dict so the caller can log the recovery. Any run
    whose persisted status is `running` or `awaiting_approval` is downgraded
    to `failed` with reason `process_restarted` — the orchestrator task is
    gone with the previous process and cannot be resumed without a job queue.

    Recurses into the run base so historical runs nested one level deeper
    (e.g. ``runs/phaseN/<id>/state.json``) are picked up too.
    """
    loaded = 0
    orphaned = 0
    if not run_dir_base.exists():
        return {"loaded": 0, "orphaned": 0}

    for state_file in run_dir_base.rglob("state.json"):
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
