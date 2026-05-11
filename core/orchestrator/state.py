"""RunState + AgentResult + ArtifactRef.

The shared contract that every agent reads and writes. Persisted both as
event-sourced JSONL (`runs/<id>/events.jsonl`) and a materialized snapshot
(`runs/<id>/state.json`).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


AGENT_ORDER: list[str] = [
    "ingestion",
    "ppg_mapping",
    "ppg_selection",
    "feature_selection",
    "eda",
    "feature_engineering",
    "feature_refine",
    "modeling",
    "results_reasoning",
    "decomposition",
    "simulation",
    "optimization",
    "validation",
    "insights",
]


class AgentStatus(str, Enum):
    pending = "pending"
    running = "running"
    awaiting_approval = "awaiting_approval"
    done = "done"
    failed = "failed"
    skipped = "skipped"


class ArtifactRef(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    path: str
    mime: str = "application/octet-stream"
    agent: str
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ToolCall(BaseModel):
    name: str
    duration_ms: int = 0
    args_hash: str = ""


class AgentResult(BaseModel):
    agent: str
    status: AgentStatus = AgentStatus.pending
    inputs_used: list[str] = Field(default_factory=list)
    actions_taken: list[ToolCall] = Field(default_factory=list)
    outputs: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    awaiting_approval = "awaiting_approval"
    completed = "completed"
    failed = "failed"


class RunState(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: RunStatus = RunStatus.pending
    data_path: str
    duckdb_path: str
    run_dir: str
    agents: dict[str, AgentResult] = Field(default_factory=dict)
    gates: dict[str, bool] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def new(cls, data_path: str, run_dir: Path, options: dict[str, Any] | None = None) -> "RunState":
        run_dir.mkdir(parents=True, exist_ok=True)
        rs = cls(
            data_path=data_path,
            duckdb_path=str((run_dir / "warehouse.duckdb").resolve()),
            run_dir=str(run_dir.resolve()),
            agents={name: AgentResult(agent=name) for name in AGENT_ORDER},
            options=options or {},
        )
        return rs

    def save(self) -> Path:
        path = Path(self.run_dir) / "state.json"
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        return cls.model_validate_json((run_dir / "state.json").read_text())


def append_event(run_dir: Path, event: dict[str, Any]) -> None:
    """Append a JSON event to the per-run JSONL log (event-sourced trail)."""
    path = run_dir / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")
