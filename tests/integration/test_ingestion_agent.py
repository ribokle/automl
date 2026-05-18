"""Real-time integration test for the ingestion agent.

Replaces a shallow smoke test with a full agent run on the synthetic panel:
the agent loads the CSV into a fresh DuckDB, runs ``dbt build`` (schema +
singular tests), runs the in-code Great Expectations suites, profiles the
mart, and asks the LLM for narrated findings. All artefacts the downstream
agents depend on are asserted on disk.

The same test is parametrised over the four LLM providers (dry_run, api,
oauth, cli). Each non-dry-run provider auto-skips when its credentials are
absent so this stays CI-safe — opt into live calls by exporting the
relevant env var (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``) or
ensuring the ``claude`` binary is on PATH and setting ``RUN_LIVE_LLM=true``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from core.agents.ingestion import IngestionAgent
from core.llm.client import AnthropicClient, LLMProvider
from core.orchestrator.state import AgentStatus, RunState
from synthetic.generator import write_panel


def _live_llm_opt_in() -> bool:
    return os.environ.get("RUN_LIVE_LLM", "").strip().lower() in ("1", "true", "yes", "on")


def _skip_unless_creds(provider: LLMProvider) -> None:
    """Skip the parametrised case when its credentials/binary aren't available."""
    if provider == LLMProvider.DRY_RUN:
        return
    if not _live_llm_opt_in():
        pytest.skip(f"set RUN_LIVE_LLM=true to exercise provider={provider.value}")
    if provider == LLMProvider.API and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    if provider == LLMProvider.OAUTH and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        pytest.skip("ANTHROPIC_AUTH_TOKEN not set")
    if provider == LLMProvider.CLI and not shutil.which("claude"):
        pytest.skip("claude CLI not on PATH")


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Seed a fresh synthetic panel once per module; reused across providers."""
    seed_dir = tmp_path_factory.mktemp("synthetic_seed")
    csv = seed_dir / "panel.csv"
    truth = seed_dir / "truth.json"
    write_panel(csv, truth, seed=42)
    return csv


def _new_run(tmp_path: Path, csv: Path) -> RunState:
    run_dir = tmp_path / "run"
    run = RunState.new(data_path=str(csv), run_dir=run_dir)
    return run


@pytest.mark.parametrize(
    "provider",
    [
        LLMProvider.DRY_RUN,
        pytest.param(LLMProvider.API, marks=pytest.mark.live_llm),
        pytest.param(LLMProvider.OAUTH, marks=pytest.mark.live_llm),
        pytest.param(LLMProvider.CLI, marks=pytest.mark.live_llm),
    ],
)
async def test_ingestion_agent_end_to_end(
    provider: LLMProvider, tmp_path: Path, synthetic_csv: Path
) -> None:
    _skip_unless_creds(provider)

    run = _new_run(tmp_path, synthetic_csv)
    agent = IngestionAgent(llm=AnthropicClient(provider=provider))
    result = await agent.run(run)

    assert result.status == AgentStatus.done, result.error
    assert result.outputs["row_count"] > 0
    assert result.outputs["ok"] is True, "no error-severity dbt or GE failures expected on synthetic"

    run_dir = Path(run.run_dir)
    for name in (
        "ingestion_report.json",
        "data_profile.json",
        "ingestion_findings.json",
        "coverage_grid.json",
        "weekly_trend.json",
        "quality_results.json",
    ):
        assert (run_dir / name).exists(), f"{name} should be on disk after ingestion"

    report = json.loads((run_dir / "ingestion_report.json").read_text())
    assert report["table"] == "panel"
    assert report["row_count"] == result.outputs["row_count"]
    assert len(report["dbt"]) > 0, "dbt must surface at least one test result"
    error_checks = [c for c in report["dbt"] + report["ge"] if c["severity"] == "error" and c["status"] == "fail"]
    assert not error_checks, f"synthetic produced error-severity failures: {error_checks}"

    findings = json.loads((run_dir / "ingestion_findings.json").read_text())
    for key in ("summary", "anomalies", "recommendations"):
        assert key in findings, f"findings missing {key}"
    assert isinstance(findings["summary"], str) and findings["summary"]

    trace_path = run_dir / "ingestion_llm_trace.json"
    assert trace_path.exists(), "LLM trace must be captured under any provider"
    trace = json.loads(trace_path.read_text())
    assert trace["agent"] == "ingestion"
    assert len(trace["calls"]) == 1
    call = trace["calls"][0]
    if provider == LLMProvider.DRY_RUN:
        assert call["dry_run"] is True
        assert call["tokens_in"] == 0 and call["tokens_out"] == 0
    else:
        assert call["dry_run"] is False
        assert call["tokens_in"] > 0
        assert call["tokens_out"] > 0
        assert call["response"], "live providers must return non-empty text"

    assert (run_dir / "warehouse.duckdb").exists()
