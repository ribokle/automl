"""Insights agent.

Reads every upstream agent's artefact, assembles a structured
``ReportPayload``, generates an executive narrative via the LLM (with a
deterministic dry-run fallback), and emits:

- ``insights_summary.json`` — payload the UI renders + the next pipeline
  could re-consume.
- ``cost_summary.json`` — per-agent token / cost / duration rollup.
- ``report.html`` — self-contained executive report.
- ``report.pdf`` — same content rendered through WeasyPrint.

The agent never refits or recomputes — every number it surfaces traces
back to an artefact written by an earlier agent. That keeps the report
explainable and the LLM tightly grounded.
"""
from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.agents.base import Agent
from core.llm.cost import summarise_run
from core.orchestrator.state import AgentResult, ArtifactRef, RunState
from core.report.builder import build_html, build_pdf

SYSTEM_PROMPT = """You are the executive-summary writer for a CPG pricing
study. You receive: per-PPG recommended prices and promo states from a
constrained MILP, the validation verdict per PPG, the model winner per
PPG, and the total recommended revenue/margin. Return STRICT JSON:
{"headline": "<=280 chars exec summary calling out total revenue lift,
  any relaxed PPGs, and confidence by validation",
 "per_ppg": [{"ppg_id": "...", "rationale": "<=160 chars on the price
  choice and the validation verdict"}]}
JSON only. Cite only PPGs in the input."""


def _read_json(run_dir: Path, name: str) -> Any:
    path = run_dir / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _index_by_ppg(rows: list[dict] | None, key: str = "ppg_id") -> dict[str, dict]:
    if not rows:
        return {}
    return {r[key]: r for r in rows if key in r}


def _build_payload(
    run: RunState,
    headline: str,
    rationale_by_id: dict[str, str],
) -> tuple[dict[str, Any], list[dict], list[dict]]:
    run_dir = Path(run.run_dir)
    opt_table = _read_json(run_dir, "optimization_table.json") or []
    val_table = _read_json(run_dir, "validation_table.json") or []
    model_choice = _read_json(run_dir, "model_choice_summary.json") or []
    decomp_table = _read_json(run_dir, "decomposition_table.json") or []
    constraints = _read_json(run_dir, "optimization_constraints.json") or {}

    val_by_ppg = _index_by_ppg(val_table)
    model_by_ppg = _index_by_ppg(model_choice)

    n_feasible = sum(1 for r in opt_table if not r.get("relaxed"))
    n_relaxed = sum(1 for r in opt_table if r.get("relaxed"))
    total_revenue = sum(float(r.get("revenue", 0.0) or 0.0) for r in opt_table)
    total_margin = sum(float(r.get("margin", 0.0) or 0.0) for r in opt_table)

    n_pass = sum(1 for r in val_table if r.get("verdict") == "pass")
    n_warn = sum(1 for r in val_table if r.get("verdict") == "warn")
    n_fail = sum(1 for r in val_table if r.get("verdict") == "fail")

    per_ppg: list[dict] = []
    for r in opt_table:
        ppg_id = r["ppg_id"]
        val = val_by_ppg.get(ppg_id, {})
        mc = model_by_ppg.get(ppg_id, {})
        per_ppg.append(
            {
                "ppg_id": ppg_id,
                "recommended_price": r.get("price"),
                "price_multiplier": r.get("price_multiplier", 1.0),
                "promo": r.get("promo", 0),
                "units": r.get("units"),
                "revenue": r.get("revenue"),
                "margin": r.get("margin"),
                "relaxed": bool(r.get("relaxed", False)),
                "verdict": val.get("verdict", "warn"),
                "elasticity": mc.get("own_elasticity") or val.get("elasticity_mean"),
                "rationale": rationale_by_id.get(ppg_id, ""),
            }
        )

    per_agent_cost, totals = summarise_run(run)
    cost_rows = [a.to_dict() for a in per_agent_cost]

    payload: dict[str, Any] = {
        "run_id": run.id,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "objective": constraints.get("objective", "revenue"),
        "headline": headline,
        "kpis": {
            "n_optimised": len(opt_table),
            "n_feasible": n_feasible,
            "n_relaxed": n_relaxed,
            "n_validated": len(val_table),
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
            "total_revenue": total_revenue,
            "total_margin": total_margin,
        },
        "recommendations": opt_table,
        "per_ppg": per_ppg,
        "validation": val_table,
        "decomposition": decomp_table[:24],  # cap report length
        "cost": cost_rows,
        "cost_totals": totals.to_dict(),
        "constraints": constraints,
    }
    return payload, cost_rows, [a.to_dict() for a in per_agent_cost]


def _dry_run_headline(payload: dict[str, Any]) -> str:
    k = payload["kpis"]
    return (
        f"Optimised {k['n_optimised']} PPGs against the {payload['objective']} objective. "
        f"Recommended revenue ${k['total_revenue']:,.0f} / margin ${k['total_margin']:,.0f}. "
        f"Validation: {k['n_pass']} pass / {k['n_warn']} warn / {k['n_fail']} fail; "
        f"{k['n_relaxed']} PPG(s) required constraint relaxation."
    )


def _dry_run_rationale(row: dict) -> str:
    pct = (float(row.get("price_multiplier", 1.0)) - 1.0) * 100
    direction = "raise" if pct >= 0 else "cut"
    promo = "promo on" if row.get("promo") == 1 else "promo off"
    relaxed = "; comp-gap relaxed" if row.get("relaxed") else ""
    return f"{direction} {abs(pct):.1f}% · {promo} · verdict {row.get('verdict', 'warn')}{relaxed}"


class InsightsAgent(Agent):
    name = "insights"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        run_dir = Path(run.run_dir)

        payload, cost_rows, _ = await asyncio.to_thread(_build_payload, run, "", {})
        headline, rationales = self._narrate(result, payload)

        if not headline:
            headline = _dry_run_headline(payload)
            rationales = [
                {"ppg_id": r["ppg_id"], "rationale": _dry_run_rationale(r)}
                for r in payload["per_ppg"]
            ]
        rationale_by_id = {r["ppg_id"]: r["rationale"] for r in rationales}

        payload, _, _ = await asyncio.to_thread(_build_payload, run, headline, rationale_by_id)

        summary_path = run_dir / "insights_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(summary_path),
                mime="application/json",
                agent=self.name,
                name=summary_path.name,
            )
        )

        cost_path = run_dir / "cost_summary.json"
        cost_path.write_text(
            json.dumps(
                {"per_agent": cost_rows, "totals": payload["cost_totals"]},
                indent=2,
                default=str,
            )
        )
        result.artifacts.append(
            ArtifactRef(
                path=str(cost_path),
                mime="application/json",
                agent=self.name,
                name=cost_path.name,
            )
        )

        html = await asyncio.to_thread(build_html, payload)
        html_path = run_dir / "report.html"
        html_path.write_text(html, encoding="utf-8")
        result.artifacts.append(
            ArtifactRef(
                path=str(html_path),
                mime="text/html",
                agent=self.name,
                name=html_path.name,
            )
        )

        try:
            pdf_bytes = await asyncio.to_thread(build_pdf, html)
            pdf_path = run_dir / "report.pdf"
            pdf_path.write_bytes(pdf_bytes)
            result.artifacts.append(
                ArtifactRef(
                    path=str(pdf_path),
                    mime="application/pdf",
                    agent=self.name,
                    name=pdf_path.name,
                )
            )
            pdf_ok = True
        except Exception as exc:  # noqa: BLE001
            pdf_ok = False
            result.outputs["pdf_error"] = str(exc)

        k = payload["kpis"]
        result.outputs.update(
            {
                "n_ppgs": k["n_optimised"],
                "n_pass": k["n_pass"],
                "n_warn": k["n_warn"],
                "n_fail": k["n_fail"],
                "n_relaxed": k["n_relaxed"],
                "total_revenue": k["total_revenue"],
                "total_margin": k["total_margin"],
                "cost_usd": payload["cost_totals"]["cost_usd"],
                "pdf": pdf_ok,
            }
        )
        result.reasoning = headline
        result.confidence = (k["n_pass"] / k["n_validated"]) if k["n_validated"] else 0.0

    def _narrate(
        self, result: AgentResult, payload: dict[str, Any]
    ) -> tuple[str, list[dict]]:
        compact = {
            "objective": payload["objective"],
            "kpis": payload["kpis"],
            "per_ppg": [
                {
                    "ppg_id": p["ppg_id"],
                    "price_multiplier": p["price_multiplier"],
                    "promo": p["promo"],
                    "revenue": p["revenue"],
                    "margin": p["margin"],
                    "verdict": p["verdict"],
                    "elasticity": p["elasticity"],
                    "relaxed": p["relaxed"],
                }
                for p in payload["per_ppg"]
            ],
        }
        try:
            resp = self.call_llm(
                result,
                system=SYSTEM_PROMPT,
                user=json.dumps(compact, default=str),
                max_tokens=900,
                label="insights-exec-summary",
            )
            if resp.raw.get("dry_run"):
                return "", []
            blob = json.loads(resp.text)
            return str(blob.get("headline", "")), list(blob.get("per_ppg", []))
        except (json.JSONDecodeError, ValueError):
            return "", []
