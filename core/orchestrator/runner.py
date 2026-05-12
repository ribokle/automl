"""Async DAG executor.

Phase 0 deliverable: runs the ingestion stage for real (dbt + GE) and emits
mocked events for the remaining 13 stages so the full timeline is observable
end-to-end through the API and Next.js UI. Subsequent phases replace each
mocked stage with its real agent.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from core.data.dbt_runner import run_dbt_build
from core.data.ge_runner import run_ge_checks
from core.data.io import load_csv_to_duckdb
from core.data.ingestion_report import IngestionReport
from core.orchestrator.events import bus
from core.orchestrator.gates import DEFAULT_GATES
from core.orchestrator.state import (
    AGENT_ORDER,
    AgentResult,
    AgentStatus,
    RunState,
    RunStatus,
)


async def _emit(run: RunState, event_type: str, payload: dict | None = None) -> None:
    await bus.publish(
        run.id,
        run.run_dir,
        {"type": event_type, **(payload or {})},
    )


async def _run_ingestion(run: RunState) -> AgentResult:
    """Real ingestion: CSV -> DuckDB -> dbt build -> GE checks -> IngestionReport."""
    result = run.agents["ingestion"]
    result.status = AgentStatus.running
    result.started_at = datetime.utcnow()
    await _emit(run, "agent_started", {"agent": "ingestion"})

    csv_path = Path(run.data_path)
    duckdb_path = Path(run.duckdb_path)
    row_count = await asyncio.to_thread(load_csv_to_duckdb, csv_path, duckdb_path)
    await _emit(run, "tool_called", {"agent": "ingestion", "tool": "load_csv_to_duckdb", "rows": row_count})

    dbt_results = await asyncio.to_thread(run_dbt_build, duckdb_path)
    await _emit(run, "tool_called", {"agent": "ingestion", "tool": "dbt_build", "checks": len(dbt_results)})

    try:
        ge_results = await asyncio.to_thread(run_ge_checks, duckdb_path, "panel")
        await _emit(run, "tool_called", {"agent": "ingestion", "tool": "ge_checks", "checks": len(ge_results)})
    except Exception as exc:
        ge_results = []
        await _emit(run, "tool_error", {"agent": "ingestion", "tool": "ge_checks", "error": str(exc)})

    report = IngestionReport(
        duckdb_path=str(duckdb_path),
        table="panel",
        row_count=row_count,
        dbt=dbt_results,
        ge=ge_results,
    )
    (Path(run.run_dir) / "ingestion_report.json").write_text(report.model_dump_json(indent=2))

    result.outputs = {
        "row_count": row_count,
        "dbt_failures": len([c for c in dbt_results if c.status == "fail"]),
        "ge_failures": len([c for c in ge_results if c.status == "fail"]),
        "ok": report.ok,
    }
    result.reasoning = (
        f"Loaded {row_count} rows. "
        f"dbt: {len(dbt_results)} checks, {result.outputs['dbt_failures']} failed. "
        f"GE: {len(ge_results)} checks, {result.outputs['ge_failures']} failed."
    )
    result.confidence = 0.95 if report.ok else 0.5
    result.status = AgentStatus.failed if not report.ok else AgentStatus.done
    result.finished_at = datetime.utcnow()
    await _emit(run, "agent_finished", {"agent": "ingestion", "status": result.status.value, "outputs": result.outputs})
    return result


async def _run_stub(run: RunState, agent_name: str) -> AgentResult:
    """Mocked agent execution for Phase 0 (everything past ingestion)."""
    result = run.agents[agent_name]
    result.status = AgentStatus.running
    result.started_at = datetime.utcnow()
    await _emit(run, "agent_started", {"agent": agent_name})

    # Simulated work.
    await asyncio.sleep(0.05)

    result.outputs = {"mocked": True}
    result.reasoning = f"[stub] {agent_name} will be implemented in a later phase."
    result.confidence = 0.0
    result.status = AgentStatus.done
    result.finished_at = datetime.utcnow()
    await _emit(run, "agent_finished", {"agent": agent_name, "status": result.status.value})
    return result


async def execute(run: RunState, gates_enabled: bool = True) -> RunState:
    run.status = RunStatus.running
    run.save()
    if gates_enabled:
        run.gates = dict(DEFAULT_GATES)

    await _emit(run, "run_started", {"agents": AGENT_ORDER})

    for agent_name in AGENT_ORDER:
        try:
            if agent_name == "ingestion":
                await _run_ingestion(run)
                if run.agents["ingestion"].status == AgentStatus.failed:
                    run.status = RunStatus.failed
                    break
            else:
                await _run_stub(run, agent_name)
        except Exception as exc:  # noqa: BLE001
            run.agents[agent_name].status = AgentStatus.failed
            run.agents[agent_name].error = str(exc)
            run.status = RunStatus.failed
            await _emit(run, "agent_failed", {"agent": agent_name, "error": str(exc)})
            break
        run.save()

    if run.status != RunStatus.failed:
        run.status = RunStatus.completed
    run.save()
    await _emit(run, "run_finished", {"status": run.status.value})
    return run
