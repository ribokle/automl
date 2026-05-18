"""Ingestion agent.

Loads CSV -> DuckDB, runs the dbt build (schema + singular tests), runs the
in-code Great Expectations suites, profiles the canonical panel, and asks
the LLM for a structured narrative of findings.

Outputs:
- ingestion_report.json  (dbt + GE results)
- data_profile.json      (per-column stats + outlier counts)
- ingestion_findings.json (LLM-tagged summary / anomalies / recommendations)
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from core.agents.base import Agent
from core.data.dbt_runner import run_dbt_build
from core.data.ge_runner import run_ge_checks
from core.data.io import load_csv_to_duckdb
from core.data.ingestion_report import IngestionReport
from core.data.tools import detect_outliers, profile_table, sample_rows
from core.orchestrator.state import AgentResult, AgentStatus, ArtifactRef, RunState


SYSTEM_PROMPT = """You are the data-ingestion analyst for an automated CPG price/promo platform.
You receive: (1) an IngestionReport summarising dbt + Great Expectations results,
(2) a per-column profile of the canonical weekly panel, and (3) IQR-outlier counts
for price/units/distribution_acv.

Produce STRICT JSON with three keys:
- "summary": a 2-3 sentence plain-English description of the dataset.
- "anomalies": a list of objects {"tag": short-slug, "severity": "info"|"warn"|"error",
  "message": one-line description}. Only include genuine anomalies grounded in the
  inputs you were given - do not invent findings.
- "recommendations": a list of 1-4 short, concrete next steps for the downstream
  PPG-mapping and modelling agents.

Reply with JSON only, no prose."""


def _dry_run_findings(profile: dict, report: IngestionReport, outliers: list[dict]) -> dict:
    anomalies: list[dict] = []
    for c in report.dbt + report.ge:
        if c.status == "fail":
            anomalies.append(
                {
                    "tag": f"{c.source}_{c.name}".lower().replace(".", "_")[:60],
                    "severity": c.severity.value,
                    "message": c.message[:160] if c.message else c.name,
                }
            )
    for o in outliers:
        if o["outlier_pct"] >= 1.0:
            anomalies.append(
                {
                    "tag": f"outliers_{o['column']}",
                    "severity": "warn" if o["outlier_pct"] < 5 else "info",
                    "message": f"{o['n_outliers']} IQR outliers in {o['column']} ({o['outlier_pct']}% of rows)",
                }
            )
    return {
        "summary": (
            f"Weekly CPG panel: {profile['row_count']} rows across "
            f"{len([c for c in profile['columns'] if c['name'] == 'sku'])} SKU column "
            f"and {len(profile['columns'])} total columns. "
            f"dbt {len(report.dbt)} checks, GE {len(report.ge)} checks."
        ),
        "anomalies": anomalies[:8],
        "recommendations": [
            "Confirm base_price is true regular-price baseline before modelling.",
            "Forward PPG mapping to the next agent using brand+category as a strong prior.",
        ],
    }


class IngestionAgent(Agent):
    name = "ingestion"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        csv_path = Path(run.data_path)
        duckdb_path = Path(run.duckdb_path)
        run_dir = Path(run.run_dir)

        row_count = await asyncio.to_thread(load_csv_to_duckdb, csv_path, duckdb_path)
        await self.emit(run, "tool_called", {"tool": "load_csv_to_duckdb", "rows": row_count})

        dbt_results = await asyncio.to_thread(run_dbt_build, duckdb_path)
        await self.emit(run, "tool_called", {"tool": "dbt_build", "checks": len(dbt_results)})

        try:
            ge_results = await asyncio.to_thread(run_ge_checks, duckdb_path, "panel")
            await self.emit(run, "tool_called", {"tool": "ge_checks", "checks": len(ge_results)})
        except Exception as exc:
            ge_results = []
            await self.emit(run, "tool_error", {"tool": "ge_checks", "error": str(exc)})

        report = IngestionReport(
            duckdb_path=str(duckdb_path),
            table="panel",
            row_count=row_count,
            dbt=dbt_results,
            ge=ge_results,
        )
        report_path = run_dir / "ingestion_report.json"
        report_path.write_text(report.model_dump_json(indent=2))
        result.artifacts.append(
            ArtifactRef(
                path=str(report_path),
                mime="application/json",
                agent=self.name,
                name="ingestion_report.json",
            )
        )

        profile = await asyncio.to_thread(profile_table, duckdb_path, "panel")
        outliers = [
            await asyncio.to_thread(detect_outliers, duckdb_path, col, "panel")
            for col in ("price", "units", "distribution_acv")
        ]
        samples = await asyncio.to_thread(sample_rows, duckdb_path, "panel", 5)
        profile_blob = {"profile": profile, "outliers": outliers, "sample": samples}
        profile_path = run_dir / "data_profile.json"
        profile_path.write_text(json.dumps(profile_blob, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(profile_path),
                mime="application/json",
                agent=self.name,
                name="data_profile.json",
            )
        )
        await self.emit(run, "tool_called", {"tool": "profile_table", "columns": len(profile["columns"])})

        user_prompt = json.dumps(
            {"ingestion_report": report.model_dump(mode="json"), "profile": profile, "outliers": outliers},
            default=str,
        )
        llm_resp = await asyncio.to_thread(
            self.call_llm,
            result,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=900,
        )
        findings: dict
        try:
            findings = json.loads(llm_resp.text) if not llm_resp.raw.get("dry_run") else _dry_run_findings(profile, report, outliers)
        except (json.JSONDecodeError, ValueError):
            findings = _dry_run_findings(profile, report, outliers)
            findings["summary"] = f"[LLM returned non-JSON; using deterministic fallback] {findings['summary']}"
        findings_path = run_dir / "ingestion_findings.json"
        findings_path.write_text(json.dumps(findings, indent=2))
        result.artifacts.append(
            ArtifactRef(
                path=str(findings_path),
                mime="application/json",
                agent=self.name,
                name="ingestion_findings.json",
            )
        )

        result.outputs = {
            "row_count": row_count,
            "dbt_failures": len([c for c in dbt_results if c.status == "fail"]),
            "ge_failures": len([c for c in ge_results if c.status == "fail"]),
            "ok": report.ok,
            "n_anomalies": len(findings.get("anomalies", [])),
        }
        result.reasoning = findings.get("summary", "")
        result.confidence = 0.95 if report.ok else 0.5
        if not report.ok:
            result.status = AgentStatus.failed
