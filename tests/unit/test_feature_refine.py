"""Phase 2 acceptance test.

Drives a fresh synthetic panel through dbt + the Phase 2 feature pipeline and
asserts the refined feature set satisfies:
- max VIF < 10
- max |off-diagonal correlation| <= 0.95
- log_price is retained (downstream elasticity modelling needs it by name)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.data.dbt_runner import run_dbt_build
from core.data.io import load_csv_to_duckdb
from core.features.eda import ppg_week_aggregate
from core.features.engineering import ENGINEERED_COLUMNS, TARGET, build_features
from core.features.refine import refine_features
from synthetic.generator import generate_panel


@pytest.fixture(scope="module")
def features_frame(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    tmp = tmp_path_factory.mktemp("p2")
    df, _truth = generate_panel(seed=42, n_weeks=52, skus_per_ppg=6)
    csv = tmp / "panel.csv"
    df.to_csv(csv, index=False)
    duckdb_path = tmp / "warehouse.duckdb"
    load_csv_to_duckdb(csv, duckdb_path)
    results = run_dbt_build(duckdb_path)
    assert any(r.status in {"pass", "warn"} for r in results), "dbt build produced no passing checks"

    panel = ppg_week_aggregate(duckdb_path)
    feats = build_features(panel)
    assert TARGET in feats.columns
    assert len(feats) > 0
    return feats


def test_refined_features_pass_vif_and_corr(features_frame: pd.DataFrame) -> None:
    candidates = [c for c in ENGINEERED_COLUMNS if c != TARGET and c in features_frame.columns]
    refined = refine_features(
        features_frame,
        candidates,
        vif_threshold=10.0,
        corr_threshold=0.95,
        protected=["log_price"],
    )

    assert "log_price" in refined["kept"], "log_price must survive refinement"
    assert refined["max_vif"] < 10.0, (
        f"max VIF {refined['max_vif']:.2f} exceeds 10 — kept={refined['kept']}"
    )
    assert refined["max_abs_corr"] <= 0.95, (
        f"max |corr| {refined['max_abs_corr']:.3f} exceeds 0.95 — pair={refined['max_abs_corr_pair']}"
    )
    # We should have kept enough variables to model with.
    assert len(refined["kept"]) >= 6
