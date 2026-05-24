"""API request/response models."""
from __future__ import annotations

from pydantic import BaseModel


class CreateRunRequest(BaseModel):
    data_path: str
    gates_enabled: bool = False
    agent_mode: bool = True
    # When true, the orchestrator always enables the ``ppg_mapping``
    # gate so the UI can render the grain selector after ingestion + PPG
    # mapping complete. Independent of ``gates_enabled`` (which controls
    # the full DEFAULT_GATES set). UI-triggered runs default to True so
    # the operator always sees the selector; CLI runs default to False
    # so headless runs blow through with whatever grain was passed.
    grain_gate_required: bool = False
    label: str | None = None


class RunSummary(BaseModel):
    id: str
    status: str
    data_path: str
    run_dir: str
    created_at: str
    archived: bool = False
