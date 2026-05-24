"""Approval endpoints.

POST /runs/{id}/approve releases the orchestrator from the gate after `agent`.
                        Accepts an optional JSON body carrying top-level
                        run-option overrides (e.g. the grain selector at the
                        ppg_mapping gate). Validated against ``ApprovePayload``.
POST /runs/{id}/reject  marks the gate as rejected; the runner then fails the
                        run with reason `gate_rejected`.
POST /runs/{id}/rerun   submits new `run.options[agent]` overrides; the runner
                        re-executes the agent with the merged options and
                        re-arms the gate for another review cycle. Restricted
                        to agents in `RERUNNABLE_AGENTS` (currently only
                        `optimization`).
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from api.auth import require_auth
from core.config import ModellingGrain
from core.orchestrator.gates import RERUNNABLE_AGENTS, gate_registry

router = APIRouter(prefix="/runs", tags=["approvals"], dependencies=[Depends(require_auth)])


# Operator-facing fan-out depth stages. Must match the frontend's
# GrainSelector checkbox list. Whitelisted here so a typo in the
# payload returns 422 rather than silently no-op'ing the choice.
_COMPARISON_AGENT_CHOICES: frozenset[str] = frozenset(
    {
        "modeling",
        "decomposition",
        "simulation",
        "optimization",
        "validation",
        "insights",
    }
)


class ApprovePayload(BaseModel):
    """Optional payload accepted by ``POST /runs/{id}/approve``.

    Only the fields the runner knows how to merge into ``run.options``
    are accepted; unknown keys are rejected so a typo doesn't silently
    no-op the operator's choice. Extend this model when new gate UIs
    grow new configurable fields.
    """

    modelling_grain: ModellingGrain | None = None
    comparison_grains: list[ModellingGrain] | None = None
    comparison_agents: list[str] | None = None

    model_config = {"extra": "forbid"}

    @field_validator("comparison_grains")
    @classmethod
    def _dedupe_comparison(cls, value: list[ModellingGrain] | None) -> list[ModellingGrain] | None:
        if value is None:
            return None
        seen: list[ModellingGrain] = []
        for g in value:
            if g not in seen:
                seen.append(g)
        return seen

    @field_validator("comparison_agents")
    @classmethod
    def _validate_comparison_agents(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        seen: list[str] = []
        for name in value:
            if name not in _COMPARISON_AGENT_CHOICES:
                raise ValueError(
                    f"comparison_agents: {name!r} is not a valid fan-out stage; "
                    f"choose from {sorted(_COMPARISON_AGENT_CHOICES)}"
                )
            if name not in seen:
                seen.append(name)
        return seen

    def as_options(self) -> dict[str, Any]:
        """Drop None fields; enum values become their string value."""
        out: dict[str, Any] = {}
        if self.modelling_grain is not None:
            out["modelling_grain"] = self.modelling_grain.value
        if self.comparison_grains is not None:
            out["comparison_grains"] = [g.value for g in self.comparison_grains]
        if self.comparison_agents is not None:
            out["comparison_agents"] = list(self.comparison_agents)
        return out


@router.post("/{run_id}/approve")
async def approve(
    run_id: str,
    agent: str,
    payload: ApprovePayload | None = Body(default=None),
) -> dict[str, Any]:
    state = gate_registry.get(run_id, agent)
    if state.approved is not None:
        raise HTTPException(409, detail=f"gate {agent} already resolved")
    options = payload.as_options() if payload else None
    gate_registry.approve(run_id, agent, payload=options)
    out: dict[str, Any] = {"run_id": run_id, "agent": agent, "status": "approved"}
    if options:
        out["applied_options"] = options
    return out


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
