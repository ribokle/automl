"""Approval endpoints.

POST /runs/{id}/approve releases the orchestrator from the gate after `agent`.
POST /runs/{id}/reject  marks the gate as rejected; the runner then fails the
                        run with reason `gate_rejected`.
POST /runs/{id}/rerun   submits new `run.options[agent]` overrides; the runner
                        re-executes the agent with the merged options and
                        re-arms the gate for another review cycle. Restricted
                        to agents in `RERUNNABLE_AGENTS` (currently only
                        `optimization`).
"""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException

from api.auth import require_auth
from core.orchestrator.gates import RERUNNABLE_AGENTS, gate_registry

router = APIRouter(prefix="/runs", tags=["approvals"], dependencies=[Depends(require_auth)])


@router.post("/{run_id}/approve")
async def approve(run_id: str, agent: str) -> dict[str, str]:
    state = gate_registry.get(run_id, agent)
    if state.approved is not None:
        raise HTTPException(409, detail=f"gate {agent} already resolved")
    gate_registry.approve(run_id, agent)
    return {"run_id": run_id, "agent": agent, "status": "approved"}


@router.post("/{run_id}/reject")
async def reject(run_id: str, agent: str) -> dict[str, str]:
    state = gate_registry.get(run_id, agent)
    if state.approved is not None:
        raise HTTPException(409, detail=f"gate {agent} already resolved")
    gate_registry.reject(run_id, agent)
    return {"run_id": run_id, "agent": agent, "status": "rejected"}


@router.post("/{run_id}/rerun")
async def rerun(
    run_id: str,
    agent: str,
    options: dict = Body(default_factory=dict),
) -> dict[str, str]:
    if agent not in RERUNNABLE_AGENTS:
        raise HTTPException(400, detail=f"agent {agent!r} is not rerunnable")
    ok = gate_registry.request_rerun(run_id, agent, options)
    if not ok:
        raise HTTPException(409, detail=f"gate {agent} already resolved")
    return {"run_id": run_id, "agent": agent, "status": "rerun_queued"}
