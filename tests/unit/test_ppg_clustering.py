"""End-to-end PPG-mapping verification.

Generates a fresh synthetic panel, runs the clustering algorithm, and asserts
the predicted PPG mapping recovers the embedded ground truth at >=95% SKU
agreement. This is the Phase 1 acceptance test from the build plan.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.data.io import load_csv_to_duckdb
from core.data.dbt_runner import run_dbt_build
from core.ppg.cluster import cluster_ppgs, label_match_accuracy
from core.ppg.features import aggregate_sku_features
from synthetic.generator import generate_panel


@pytest.fixture(scope="module")
def synthetic_panel(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame]:
    """Materialise a synthetic panel into DuckDB via the same dbt build the
    pipeline uses, so the test exercises the production path."""
    tmp = tmp_path_factory.mktemp("ppg")
    df, _truth = generate_panel(seed=42, n_weeks=52, skus_per_ppg=6)
    csv = tmp / "panel.csv"
    df.to_csv(csv, index=False)
    duckdb_path = tmp / "warehouse.duckdb"
    load_csv_to_duckdb(csv, duckdb_path)
    results = run_dbt_build(duckdb_path)
    assert any(r.status in {"pass", "warn"} for r in results), "dbt build produced no passing checks"
    return duckdb_path, df


def test_ppg_mapping_recovers_truth(synthetic_panel: tuple[Path, pd.DataFrame]) -> None:
    duckdb_path, df = synthetic_panel
    sku_features = aggregate_sku_features(duckdb_path)
    assignments = cluster_ppgs(sku_features)

    truth = df[["sku", "ppg_id"]].drop_duplicates().set_index("sku")["ppg_id"]
    merged = assignments.set_index("sku").join(truth.rename("truth_ppg"))
    assert merged["truth_ppg"].notna().all(), "every predicted SKU must have a truth label"

    score = label_match_accuracy(merged["ppg_id"], merged["truth_ppg"])
    assert score["accuracy"] >= 0.95, (
        f"PPG mapping recovered only {score['accuracy']:.2%} of truth "
        f"(matched {score['matched_pairs']}/{score['n_total']} SKUs)"
    )

    # Same number of unique PPGs in prediction as in truth.
    assert assignments["ppg_id"].nunique() == truth.nunique(), (
        f"predicted {assignments['ppg_id'].nunique()} PPGs vs truth {truth.nunique()}"
    )

    # Confidence should be high on this clean synthetic dataset.
    assert assignments["confidence"].mean() >= 0.85


def test_apply_mapping_rewrites_panel_ppg_id(synthetic_panel: tuple[Path, pd.DataFrame]) -> None:
    """After ppg_mapping runs, main.panel.ppg_id must reflect the clusterer's
    PPG_AUTO_* assignments, not the truth labels the panel was loaded with.
    Otherwise downstream agents that group by main.panel.ppg_id see a
    different label space than ppg_selection / ppg_mapping_table.json.
    """
    import duckdb

    from core.ppg.cluster import apply_mapping_to_panel

    duckdb_path, _df = synthetic_panel
    sku_features = aggregate_sku_features(duckdb_path)
    assignments = cluster_ppgs(sku_features)

    n_rows = apply_mapping_to_panel(duckdb_path, assignments)
    assert n_rows > 0

    con = duckdb.connect(str(duckdb_path))
    try:
        panel_ppgs = set(
            r[0] for r in con.execute("SELECT DISTINCT ppg_id FROM main.panel").fetchall()
        )
    finally:
        con.close()

    mapped_ppgs = set(assignments["ppg_id"].unique().tolist())
    assert panel_ppgs == mapped_ppgs, (
        f"panel ppg_ids {panel_ppgs} do not match mapping {mapped_ppgs}"
    )
