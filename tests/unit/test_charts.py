"""Phase 2a-1 backend: chart-ready data builders.

Drives the chart builders against a real DuckDB warehouse built from the
synthetic panel, plus a couple of graceful-degradation cases.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from core.data.charts import (
    corr_refined,
    coverage_grid,
    eda_corr_matrix,
    eligibility_bars,
    feature_histograms,
    ppg_price_box,
    ppg_scatter_behaviour,
    ppg_scatter_facet,
    ppg_scatter_tier,
    quality_results,
    weekly_trend,
)
from core.data.dbt_runner import run_dbt_build
from core.data.io import load_csv_to_duckdb
from core.data.ingestion_report import CheckResult, IngestionReport, Severity
from core.ppg.cluster import ClusterParams, cluster_ppgs
from core.ppg.features import aggregate_sku_features
from core.ppg.score import score_ppgs


@pytest.fixture(scope="module")
def panel_warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("warehouse")
    duckdb_path = out / "warehouse.duckdb"
    load_csv_to_duckdb(Path("data/synthetic.csv"), duckdb_path)
    run_dbt_build(duckdb_path)
    return duckdb_path


@pytest.fixture(scope="module")
def assignments(panel_warehouse: Path) -> pd.DataFrame:
    return cluster_ppgs(aggregate_sku_features(panel_warehouse), ClusterParams())


def test_coverage_grid_dense(panel_warehouse: Path) -> None:
    grid = coverage_grid(panel_warehouse)
    assert grid["n_total_cells"] == len(grid["skus"]) * len(grid["weeks"])
    assert grid["n_present_cells"] / grid["n_total_cells"] >= 0.95
    assert all(c in (0, 1) for row in grid["present"] for c in row)


def test_weekly_trend_shape(panel_warehouse: Path) -> None:
    t = weekly_trend(panel_warehouse)
    n = len(t["weeks"])
    assert n >= 50
    assert len(t["units"]) == len(t["price"]) == len(t["promo_share"]) == n
    assert all(0.0 <= p <= 1.0 for p in t["promo_share"])
    assert all(u > 0 for u in t["units"])


def test_quality_results_normalises_severity() -> None:
    report = IngestionReport(
        duckdb_path="/tmp/x.duckdb",
        table="panel",
        row_count=10,
        dbt=[
            CheckResult(source="dbt", name="unique_keys", status="pass", severity=Severity.error),
            CheckResult(source="dbt", name="discount_depth_in_range", status="fail", severity=Severity.warn, message="2912 rows out of band", failing_rows=2912),
        ],
        ge=[CheckResult(source="ge", name="price_between_0.5_10", status="pass", severity=Severity.error)],
    )
    q = quality_results(report)
    assert q["summary"] == {"pass": 2, "warn": 1, "fail": 0}
    by_name = {c["name"]: c for c in q["checks"]}
    assert by_name["discount_depth_in_range"]["status"] == "warn"
    assert by_name["discount_depth_in_range"]["failing_rows"] == 2912


def test_ppg_scatter_tier(assignments: pd.DataFrame) -> None:
    out = ppg_scatter_tier(assignments)
    assert "points" in out
    assert len(out["points"]) == len(assignments)
    assert all(p["x"] >= -0.5 and p["x"] <= 2.5 for p in out["points"])
    assert set(out["colours"]) >= set(assignments["ppg_id"].unique())


def test_ppg_scatter_behaviour(assignments: pd.DataFrame, panel_warehouse: Path) -> None:
    out = ppg_scatter_behaviour(assignments, panel_warehouse)
    assert len(out["points"]) >= 30
    assert all(-1.0 <= p["y"] <= 1.0 for p in out["points"])


def test_ppg_scatter_facet(assignments: pd.DataFrame) -> None:
    out = ppg_scatter_facet(assignments)
    assert len(out["facets"]) >= 1
    seen_skus = {p["sku"] for f in out["facets"] for p in f["points"]}
    assert seen_skus == set(assignments["sku"])


def test_ppg_price_box(assignments: pd.DataFrame, panel_warehouse: Path) -> None:
    out = ppg_price_box(assignments, panel_warehouse)
    boxes = out["boxes"]
    assert len(boxes) == int(assignments["ppg_id"].nunique())
    for b in boxes:
        assert b["min"] <= b["q1"] <= b["median"] <= b["q3"] <= b["max"]
        assert b["n"] > 0


def test_eligibility_bars(assignments: pd.DataFrame, panel_warehouse: Path) -> None:
    scores = score_ppgs(panel_warehouse, assignments)
    out = eligibility_bars(scores)
    assert out["threshold"] == 0.60
    assert len(out["bars"]) == len(scores)
    for bar in out["bars"]:
        contributions_sum = sum(bar["contributions"].values())
        assert abs(contributions_sum - bar["score"]) <= 0.05


def test_feature_histograms() -> None:
    df = pd.DataFrame(
        {
            "a": np.random.default_rng(0).normal(size=200),
            "b": np.random.default_rng(1).normal(loc=2, scale=0.5, size=200),
            "label": ["x"] * 200,
        }
    )
    out = feature_histograms(df, ["a", "b"], bins=10)
    assert out["bins"] == 10
    assert len(out["features"]) == 2
    for f in out["features"]:
        assert len(f["counts"]) == 10
        assert len(f["edges"]) == 11
        assert sum(f["counts"]) == 200


def test_corr_refined_shape() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100), "c": rng.normal(size=100)})
    out = corr_refined(df, ["a", "b", "c"])
    assert out["labels"] == ["a", "b", "c"]
    assert len(out["matrix"]) == 3
    for i in range(3):
        assert out["matrix"][i][i] == 1.0
        for j in range(3):
            assert -1.0 <= out["matrix"][i][j] <= 1.0


def test_eda_corr_matrix_reshape() -> None:
    pairwise = {
        "p": {"p": 1.0, "u": -0.94},
        "u": {"p": -0.94, "u": 1.0},
    }
    out = eda_corr_matrix(pairwise)
    assert out["labels"] == ["p", "u"]
    assert out["matrix"] == [[1.0, -0.94], [-0.94, 1.0]]


def test_graceful_degradation_missing_columns() -> None:
    bare = pd.DataFrame({"sku": ["A", "B"], "ppg_id": ["P1", "P2"]})
    assert ppg_scatter_tier(bare) == {"missing_columns": ["median_price", "pack_size"]}
    assert ppg_scatter_facet(bare) == {"missing_columns": ["brand", "category", "pack_size"]}
    assert eligibility_bars(pd.DataFrame({"ppg_id": ["x"]}))["missing_columns"]
