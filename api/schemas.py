"""API request/response models."""
from __future__ import annotations

from pydantic import BaseModel


class CreateRunRequest(BaseModel):
    data_path: str
    gates_enabled: bool = False
    label: str | None = None


class RunSummary(BaseModel):
    id: str
    status: str
    data_path: str
    run_dir: str
    created_at: str
