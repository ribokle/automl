"""Integration test for the advanced_eda agent.

Spins up a fresh DuckDB warehouse from the synthetic panel via the ingestion
agent, then runs the advanced_eda agent end-to-end and asserts every artefact
is on disk with the expected schema."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.agents.advanced_eda import AdvancedEDAAgent
from core.agents.ingestion import IngestionAgent
from core.llm.client import AnthropicClient, LLMProvider
from core.orchestrator.state import AgentStatus, RunState
from synthetic.generator import write_panel


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    seed_dir = tmp_path_factory.mktemp("adveda_seed")
    csv = seed_dir / "panel.csv"
    truth = seed_dir / "truth.json"
    write_panel(csv, truth, seed=42)
    return csv


@pytest.fixture(scope="module")
async def warehouse_run(tmp_path_factory: pytest.TempPathFactory, synthetic_csv: Path) -> RunState:
    """Run the ingestion agent once per module so we share the warehouse."""
    run_dir = tmp_path_factory.mktemp("adveda_run")
    run = RunState.new(data_path=str(synthetic_csv), run_dir=run_dir)
    agent = IngestionAgent(llm=AnthropicClient(provider=LLMProvider.DRY_RUN))
    result = await agent.run(run)
    assert result.status == AgentStatus.done, result.error
    return run


async def test_advanced_eda_agent_end_to_end(warehouse_run: RunState) -> None:
    agent = AdvancedEDAAgent(llm=AnthropicClient(provider=LLMProvider.DRY_RUN))
    result = await agent.run(warehouse_run)

    assert result.status == AgentStatus.done, result.error
    assert result.outputs["n_ppgs_top_k"] > 0
    assert result.outputs["n_change_points"] >= 0
    assert 0.0 <= result.outputs["stationarity_pass_rate"] <= 1.0

    run_dir = Path(warehouse_run.run_dir)
    expected = (
        "advanced_eda_report.json",
        "time_series_diagnostics.json",
        "temporal_anomalies.json",
        "change_points.json",
        "distribution_report.json",
        "promo_lift_sketches.json",
        "cross_ppg_correlation.json",
        "pareto_abc.json",
        "price_ladder.json",
        "promo_calendar.json",
        "holiday_lift.json",
        "cardinality_report.json",
        "advanced_eda_charts.json",
    )
    for name in expected:
        assert (run_dir / name).exists(), f"{name} not on disk"


async def test_report_shape(warehouse_run: RunState) -> None:
    agent = AdvancedEDAAgent(llm=AnthropicClient(provider=LLMProvider.DRY_RUN))
    await agent.run(warehouse_run)
    run_dir = Path(warehouse_run.run_dir)
    report = json.loads((run_dir / "advanced_eda_report.json").read_text())
    for key in ("summary", "findings", "narrative", "compute_caps", "artifact_index"):
        assert key in report
    assert isinstance(report["findings"], list) and report["findings"]
    assert isinstance(report["narrative"], str) and report["narrative"]


async def test_time_series_diagnostics_shape(warehouse_run: RunState) -> None:
    run_dir = Path(warehouse_run.run_dir)
    blob = json.loads((run_dir / "time_series_diagnostics.json").read_text())
    assert "diagnostics" in blob
    diag_list = blob["diagnostics"]
    assert isinstance(diag_list, list)
    assert diag_list, "expected at least one PPG diagnosed on the synthetic panel"
    first = diag_list[0]
    for key in ("ppg_id", "n_weeks", "stl", "acf_pacf", "stationarity"):
        assert key in first
    # STL keys on a 104-week synthetic series
    assert first["stl"]["available"] is True
    assert len(first["stl"]["observed"]) > 50


async def test_anomalies_shape(warehouse_run: RunState) -> None:
    run_dir = Path(warehouse_run.run_dir)
    blob = json.loads((run_dir / "temporal_anomalies.json").read_text())
    assert "n_total" in blob and "by_type" in blob and "rows" in blob
    for t in ("stockout", "pantry_loading", "forward_buy", "isolation_forest"):
        assert t in blob["by_type"]
    for row in blob["rows"]:
        assert {"ppg_id", "week_start", "anomaly_type", "severity"}.issubset(row.keys())


async def test_pareto_abc_partitions(warehouse_run: RunState) -> None:
    run_dir = Path(warehouse_run.run_dir)
    blob = json.loads((run_dir / "pareto_abc.json").read_text())
    assert "ppg_abc" in blob and "sku" in blob
    classes = {r["abc_class"] for r in blob["ppg_abc"]}
    assert classes.issubset({"A", "B", "C"})
    assert blob["sku"]["revenue"]["cum_share"], "expected non-empty SKU lorenz curve"


async def test_price_ladder_caveat_present(warehouse_run: RunState) -> None:
    run_dir = Path(warehouse_run.run_dir)
    blob = json.loads((run_dir / "price_ladder.json").read_text())
    assert blob["per_ppg"], "expected at least one PPG's ladder"
    for row in blob["per_ppg"]:
        assert "caveat" in row
        assert "elasticity" not in row["caveat"].lower() or "not causal" in row["caveat"].lower()


async def test_charts_pack(warehouse_run: RunState) -> None:
    run_dir = Path(warehouse_run.run_dir)
    blob = json.loads((run_dir / "advanced_eda_charts.json").read_text())
    for key in ("stl", "acf_pacf", "anomaly_timeline", "cross_ppg", "promo_calendar"):
        assert key in blob
    assert isinstance(blob["stl"], list)


async def test_top_k_cap_obeyed() -> None:
    """A run with ``max_series=2`` must diagnose at most 2 PPGs even when the
    panel has more."""
    import tempfile

    from synthetic.generator import write_panel

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv = tmp_path / "panel.csv"
        write_panel(csv, tmp_path / "truth.json", seed=7)
        run_dir = tmp_path / "run"
        run = RunState.new(data_path=str(csv), run_dir=run_dir, options={"advanced_eda": {"max_series": 2}})
        ingest = IngestionAgent(llm=AnthropicClient(provider=LLMProvider.DRY_RUN))
        await ingest.run(run)
        agent = AdvancedEDAAgent(llm=AnthropicClient(provider=LLMProvider.DRY_RUN))
        result = await agent.run(run)
        assert result.outputs["n_ppgs_top_k"] == 2
        diag = json.loads((Path(run.run_dir) / "time_series_diagnostics.json").read_text())
        assert len(diag["diagnostics"]) == 2
