"""Report builder + insights agent."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.agents.insights import InsightsAgent
from core.llm.cost import summarise_run
from core.orchestrator.state import AgentResult, AgentStatus, RunState
from core.report.builder import build_html, build_pdf


@pytest.fixture(scope="session")
def weasyprint_ok() -> bool:
    """True only when WeasyPrint's native GTK/Pango/Cairo libs are available."""
    try:
        from weasyprint import HTML
        HTML(string="<p>ok</p>").write_pdf()
        return True
    except Exception:
        return False


def _payload(**overrides):
    base = {
        "run_id": "run-test",
        "generated_at": "2026-05-19 10:00 UTC",
        "objective": "revenue",
        "headline": "Solid lift on 3 PPGs; 1 needed comp-gap relaxation.",
        "kpis": {
            "n_optimised": 2,
            "n_feasible": 1,
            "n_relaxed": 1,
            "n_validated": 2,
            "n_pass": 1,
            "n_warn": 1,
            "n_fail": 0,
            "total_revenue": 230_000.0,
            "total_margin": 90_000.0,
        },
        "recommendations": [
            {
                "ppg_id": "PPG_A",
                "objective": "revenue",
                "price_multiplier": 0.95,
                "price": 4.75,
                "base_price": 5.00,
                "promo": 1,
                "units": 20_000,
                "revenue": 95_000.0,
                "margin": 40_000.0,
                "feasible_strict": True,
                "relaxed": False,
                "model_kind": "loglog_ols",
            },
            {
                "ppg_id": "PPG_B",
                "objective": "revenue",
                "price_multiplier": 1.15,
                "price": 1.15,
                "base_price": 1.00,
                "promo": 0,
                "units": 30_000,
                "revenue": 135_000.0,
                "margin": 50_000.0,
                "feasible_strict": False,
                "relaxed": True,
                "model_kind": "semilog_ols",
            },
        ],
        "per_ppg": [
            {
                "ppg_id": "PPG_A",
                "recommended_price": 4.75,
                "price_multiplier": 0.95,
                "promo": 1,
                "units": 20_000,
                "revenue": 95_000.0,
                "margin": 40_000.0,
                "relaxed": False,
                "verdict": "pass",
                "elasticity": -2.3,
                "rationale": "drop 5% promo on; verdict pass",
            },
            {
                "ppg_id": "PPG_B",
                "recommended_price": 1.15,
                "price_multiplier": 1.15,
                "promo": 0,
                "units": 30_000,
                "revenue": 135_000.0,
                "margin": 50_000.0,
                "relaxed": True,
                "verdict": "warn",
                "elasticity": -1.1,
                "rationale": "raise 15% promo off; comp-gap relaxed",
            },
        ],
        "validation": [
            {
                "ppg_id": "PPG_A",
                "winner": "loglog_ols",
                "verdict": "pass",
                "sign_stability": 1.0,
                "wape_mean": 0.05,
                "elasticity_mean": -2.3,
                "elasticity_cv": 0.2,
                "n_folds": 4,
            },
            {
                "ppg_id": "PPG_B",
                "winner": "semilog_ols",
                "verdict": "warn",
                "sign_stability": 0.75,
                "wape_mean": 0.08,
                "elasticity_mean": -1.1,
                "elasticity_cv": 0.6,
                "n_folds": 4,
            },
        ],
        "decomposition": [
            {"ppg_id": "PPG_A", "group": "price", "due_units": 10000, "share_of_lift": 0.5},
            {"ppg_id": "PPG_A", "group": "promo", "due_units": 6000, "share_of_lift": 0.3},
        ],
        "cost": [
            {
                "agent": "modeling",
                "status": "done",
                "tokens_in": 1500,
                "tokens_out": 500,
                "cost_usd": 0.0234,
                "duration_s": 12.4,
                "duration_str": "12.4 s",
            }
        ],
        "cost_totals": {
            "tokens_in": 1500,
            "tokens_out": 500,
            "cost_usd": 0.0234,
            "duration_s": 12.4,
            "duration_str": "12.4 s",
        },
        "constraints": {"objective": "revenue", "price_ladder": [0.9, 1.0, 1.1]},
    }
    base.update(overrides)
    return base


def test_build_html_renders_required_sections() -> None:
    html = build_html(_payload())
    assert "Executive Report" in html
    assert "Recommendations" in html
    assert "Validation" in html
    assert "Cost &amp; runtime" in html
    assert "PPG_A" in html
    assert "PPG_B" in html
    assert "relaxed" in html
    assert "pass" in html


def test_build_html_signed_delta_filter() -> None:
    html = build_html(_payload())
    assert "-5.0%" in html
    assert "+15.0%" in html


def test_build_pdf_returns_bytes(weasyprint_ok: bool) -> None:
    if not weasyprint_ok:
        pytest.skip("WeasyPrint native libs not available on this platform")
    pdf = build_pdf(build_html(_payload()))
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def _new_run(tmp_path: Path) -> RunState:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    state.run_dir = str(run_dir.resolve())
    state.agents["insights"] = AgentResult(agent="insights", status=AgentStatus.pending)
    return state


def _seed_upstream_artifacts(run_dir: Path) -> None:
    (run_dir / "optimization_table.json").write_text(
        json.dumps(
            [
                {
                    "ppg_id": "PPG_A",
                    "objective": "revenue",
                    "price_multiplier": 0.95,
                    "price": 4.75,
                    "base_price": 5.0,
                    "promo": 1,
                    "units": 20000,
                    "revenue": 95000.0,
                    "margin": 40000.0,
                    "feasible_strict": True,
                    "relaxed": False,
                    "model_kind": "loglog_ols",
                }
            ]
        )
    )
    (run_dir / "validation_table.json").write_text(
        json.dumps(
            [
                {
                    "ppg_id": "PPG_A",
                    "winner": "loglog_ols",
                    "verdict": "pass",
                    "sign_stability": 1.0,
                    "wape_mean": 0.05,
                    "elasticity_mean": -2.3,
                    "elasticity_cv": 0.2,
                    "n_folds": 4,
                }
            ]
        )
    )
    (run_dir / "model_choice_summary.json").write_text(
        json.dumps([{"ppg_id": "PPG_A", "winner": "loglog_ols", "own_elasticity": -2.3, "verdict": "pass"}])
    )
    (run_dir / "decomposition_table.json").write_text(
        json.dumps([{"ppg_id": "PPG_A", "group": "price", "due_units": 1000, "share_of_lift": 0.6}])
    )
    (run_dir / "optimization_constraints.json").write_text(
        json.dumps({"objective": "revenue", "price_ladder": [0.9, 1.0, 1.1]})
    )


def test_insights_agent_writes_artifacts(tmp_path: Path, monkeypatch, weasyprint_ok: bool) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = _new_run(tmp_path)
    _seed_upstream_artifacts(Path(state.run_dir))
    asyncio.run(InsightsAgent().run(state))

    run_dir = Path(state.run_dir)
    required = ["insights_summary.json", "cost_summary.json", "report.html"]
    if weasyprint_ok:
        required.append("report.pdf")
    for name in required:
        assert (run_dir / name).exists(), f"missing {name}"

    summary = json.loads((run_dir / "insights_summary.json").read_text())
    assert summary["run_id"] == state.id
    assert summary["kpis"]["n_optimised"] == 1
    assert summary["kpis"]["n_pass"] == 1
    assert summary["per_ppg"][0]["rationale"]  # filled by dry-run fallback

    cost = json.loads((run_dir / "cost_summary.json").read_text())
    assert "per_agent" in cost and "totals" in cost
    assert any(a["agent"] == "insights" for a in cost["per_agent"])

    if weasyprint_ok:
        pdf_bytes = (run_dir / "report.pdf").read_bytes()
        assert pdf_bytes.startswith(b"%PDF")

    outputs = state.agents["insights"].outputs
    assert outputs["n_ppgs"] == 1
    assert outputs["n_pass"] == 1
    assert outputs["pdf"] is weasyprint_ok


def test_insights_agent_dry_run_headline_used_when_no_llm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = _new_run(tmp_path)
    _seed_upstream_artifacts(Path(state.run_dir))
    asyncio.run(InsightsAgent().run(state))
    headline = state.agents["insights"].reasoning
    assert "Optimised 1 PPGs" in headline


def test_summarise_run_aggregates_per_agent(tmp_path: Path) -> None:
    state = _new_run(tmp_path)
    state.agents["modeling"].tokens_in = 1000
    state.agents["modeling"].tokens_out = 200
    state.agents["modeling"].cost_usd = 0.0123
    state.agents["modeling"].started_at = datetime.utcnow()
    state.agents["modeling"].finished_at = state.agents["modeling"].started_at + timedelta(
        seconds=15.5
    )
    state.agents["modeling"].status = AgentStatus.done

    state.agents["optimization"].tokens_in = 500
    state.agents["optimization"].tokens_out = 100
    state.agents["optimization"].cost_usd = 0.0050
    state.agents["optimization"].status = AgentStatus.done

    per_agent, totals = summarise_run(state)
    by_name = {a.agent: a for a in per_agent}
    assert by_name["modeling"].tokens_in == 1000
    assert by_name["modeling"].duration_s == pytest.approx(15.5, abs=0.1)
    assert by_name["modeling"].duration_str.endswith(" s")

    assert totals.tokens_in == 1500
    assert totals.cost_usd == pytest.approx(0.0173)
