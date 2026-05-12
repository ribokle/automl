"""Approval endpoints.

POST /runs/{id}/approve releases the orchestrator from the gate after `agent`.
POST /runs/{id}/reject  marks the gate as rejected; the runner then fails the
                        run with reason `gate_rejected`.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from core.orchestrator.gates import gate_registry

router = APIRouter(prefix="/runs", tags=["approvals"])


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
