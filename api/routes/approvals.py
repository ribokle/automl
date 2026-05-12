"""Approval endpoints. Phase 0: no-op (returns OK) so the UI can wire it up."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/runs", tags=["approvals"])


@router.post("/{run_id}/approve")
async def approve(run_id: str, agent: str) -> dict[str, str]:
    return {"run_id": run_id, "agent": agent, "status": "approved"}


@router.post("/{run_id}/reject")
async def reject(run_id: str, agent: str) -> dict[str, str]:
    return {"run_id": run_id, "agent": agent, "status": "rejected"}
