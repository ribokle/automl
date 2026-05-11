"""Unified report produced by the Ingestion stage.

Combines dbt (structural) and Great Expectations (statistical) results into
a single severity-tagged list so the Ingestion Agent can narrate both
sources coherently.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(str, Enum):
    error = "error"
    warn = "warn"
    info = "info"


class CheckResult(BaseModel):
    source: Literal["dbt", "ge"]
    name: str
    status: Literal["pass", "fail"]
    severity: Severity
    message: str = ""
    failing_rows: int | None = None
    details: dict = Field(default_factory=dict)


class IngestionReport(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duckdb_path: str
    table: str
    row_count: int
    dbt: list[CheckResult] = Field(default_factory=list)
    ge: list[CheckResult] = Field(default_factory=list)
    data_docs_path: str | None = None

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in [*self.dbt, *self.ge] if c.severity == Severity.error and c.status == "fail"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in [*self.dbt, *self.ge] if c.severity == Severity.warn and c.status == "fail"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0
