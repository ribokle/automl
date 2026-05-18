"""Programmatic dbt invocation + result parsing.

Runs `dbt build` against the project at dbt/automl_dbt and returns a list of
`CheckResult` entries. Each model/test contributes one entry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from core.data.ingestion_report import CheckResult, Severity

DBT_PROJECT_DIR = Path(__file__).resolve().parents[2] / "dbt" / "automl_dbt"


def _severity_from_status(status: str, configured_severity: str | None) -> Severity:
    # dbt encodes failures as "error" or "warn" in `status`.
    if status == "warn":
        return Severity.warn
    if status == "error":
        return Severity.error
    if status in ("pass", "success"):
        # passes still need a severity; mirror what the test was configured as so the agent can
        # group them sensibly.
        if configured_severity == "warn":
            return Severity.warn
        return Severity.info
    return Severity.info


def _ensure_dbt_packages(runner: "object", project_dir: Path) -> None:
    """Install dbt package deps the first time we see this project_dir.

    Lets a fresh checkout run end-to-end without a separate `dbt deps` step —
    the build call below would otherwise fail compiling models that reference
    dbt_utils / dbt_expectations macros.
    """
    if (project_dir / "dbt_packages").exists():
        return
    runner.invoke([  # type: ignore[attr-defined]
        "deps",
        "--project-dir", str(project_dir),
        "--profiles-dir", str(project_dir),
        "--no-version-check",
    ])


def run_dbt_build(duckdb_path: Path, project_dir: Path = DBT_PROJECT_DIR) -> list[CheckResult]:
    """Execute `dbt build` against `duckdb_path` and return per-node results."""
    from dbt.cli.main import dbtRunner, dbtRunnerResult

    os.environ["DUCKDB_PATH"] = str(duckdb_path.resolve())
    runner = dbtRunner()
    _ensure_dbt_packages(runner, project_dir)
    args = [
        "build",
        "--project-dir", str(project_dir),
        "--profiles-dir", str(project_dir),
        "--no-version-check",
    ]
    result: dbtRunnerResult = runner.invoke(args)

    checks: list[CheckResult] = []
    if result.result is None:
        # Couldn't even parse — report a single error.
        checks.append(
            CheckResult(
                source="dbt",
                name="dbt_invocation",
                status="fail",
                severity=Severity.error,
                message=str(result.exception) if result.exception else "dbt failed to start",
            )
        )
        return checks

    for r in result.result.results:  # type: ignore[attr-defined]
        node = r.node
        unique_id = getattr(node, "unique_id", "unknown")
        configured = getattr(node, "config", None)
        configured_severity = getattr(configured, "severity", None) if configured else None
        status = getattr(r, "status", "")
        passed = status in ("pass", "success")
        failing = getattr(r, "failures", None)
        checks.append(
            CheckResult(
                source="dbt",
                name=unique_id,
                status="pass" if passed else "fail",
                severity=_severity_from_status(str(status), configured_severity),
                message=getattr(r, "message", "") or "",
                failing_rows=int(failing) if failing is not None else None,
            )
        )
    return checks
