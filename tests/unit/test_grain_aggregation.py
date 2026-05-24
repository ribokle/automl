"""Tests for the configurable modelling-grain aggregation.

Confirms ``aggregate_features`` produces sensible shapes for each grain,
that ``build_features`` flows ``grain_unit`` through when present, and
that the chain (PPG_WEEK) backward-compat wrapper still drops it.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from core.features.eda import aggregate_features, ppg_week_aggregate
from core.features.engineering import build_features


@pytest.fixture
def panel_db(tmp_path: Path) -> Path:
    """Tiny synthetic panel: 3 stores x 2 PPGs x 10 weeks."""
    rows: list[dict] = []
    rng = np.random.default_rng(0)
    week_start = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    for store in ("store_A", "store_B", "store_C"):
        for ppg in ("PPG_01", "PPG_02"):
            for w in week_start:
                price = 5.0 + rng.normal(0, 0.5)
                rows.append(
                    {
                        "store_id": store,
                        "ppg_id": ppg,
                        "category": "test_cat",
                        "week_start": w.date(),
                        "units": int(rng.integers(20, 80)),
                        "price": price,
                        "base_price": 5.5,
                        "discount_depth": max(0.0, (5.5 - price) / 5.5),
                        "tpr_flag": int(price < 5.0),
                        "display_flag": 0,
                        "feature_flag": 0,
                        "distribution_acv": 100.0,
                        "competitor_price": price + rng.normal(0, 0.2),
                        "holiday": None,
                    }
                )
    df = pd.DataFrame(rows)
    db = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(db))
    try:
        con.register("panel_df", df)
        con.execute("CREATE SCHEMA IF NOT EXISTS main")
        con.execute("CREATE TABLE main.panel AS SELECT * FROM panel_df")
    finally:
        con.close()
    return db


def test_ppg_week_aggregate_unchanged_shape(panel_db: Path) -> None:
    # Back-compat wrapper drops grain_unit so existing callers see the
    # historical column set.
    df = ppg_week_aggregate(panel_db)
    assert "grain_unit" not in df.columns
    # 2 PPGs x 10 weeks
    assert len(df) == 20


def test_aggregate_features_default_grain_has_chain_grain_unit(panel_db: Path) -> None:
    df = aggregate_features(panel_db, grain="ppg_week")
    assert set(df["grain_unit"].unique()) == {"chain"}
    assert len(df) == 20


def test_aggregate_features_store_ppg_week(panel_db: Path) -> None:
    df = aggregate_features(panel_db, grain="store_ppg_week")
    # 3 stores x 2 PPGs x 10 weeks
    assert len(df) == 60
    assert set(df["grain_unit"].unique()) == {"store_A", "store_B", "store_C"}


def test_aggregate_features_store_category_week(panel_db: Path) -> None:
    df = aggregate_features(panel_db, grain="store_category_week")
    # 3 stores x 1 category x 10 weeks
    assert len(df) == 30
    assert set(df["grain_unit"].unique()) == {"store_A", "store_B", "store_C"}
    assert set(df["ppg_id"].unique()) == {"test_cat"}


def test_aggregate_features_category_week(panel_db: Path) -> None:
    df = aggregate_features(panel_db, grain="category_week")
    # 1 category x 10 weeks
    assert len(df) == 10
    assert set(df["grain_unit"].unique()) == {"chain"}
    assert set(df["ppg_id"].unique()) == {"test_cat"}


def test_aggregate_features_brand_week(tmp_path: Path) -> None:
    # Seed a fresh warehouse with a brand column populated so the
    # brand-grain SQL has something to group on.
    import pandas as pd
    import duckdb
    rows: list[dict] = []
    weeks = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    for store in ("s1", "s2"):
        for brand in ("acme", "globex"):
            for w in weeks:
                rows.append(
                    {
                        "store_id": store,
                        "ppg_id": f"{brand}_ppg",
                        "category": "cat",
                        "brand": brand,
                        "sku": f"{brand}_x",
                        "week_start": w.date(),
                        "units": 50,
                        "price": 4.0,
                        "base_price": 4.0,
                        "discount_depth": 0.0,
                        "tpr_flag": 0,
                        "display_flag": 0,
                        "feature_flag": 0,
                        "distribution_acv": 100.0,
                        "competitor_price": 4.0,
                        "holiday": None,
                    }
                )
    df = pd.DataFrame(rows)
    db = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(db))
    try:
        con.register("panel_df", df)
        con.execute("CREATE SCHEMA IF NOT EXISTS main")
        con.execute("CREATE TABLE main.panel AS SELECT * FROM panel_df")
    finally:
        con.close()

    out = aggregate_features(db, grain="brand_week")
    # 2 brands x 10 weeks; grain_unit = "chain"
    assert len(out) == 20
    assert set(out["grain_unit"].unique()) == {"chain"}
    assert set(out["ppg_id"].unique()) == {"acme", "globex"}

    store_out = aggregate_features(db, grain="store_brand_week")
    # 2 stores x 2 brands x 10 weeks
    assert len(store_out) == 40
    assert set(store_out["grain_unit"].unique()) == {"s1", "s2"}


def test_aggregate_features_rejects_unknown_grain(panel_db: Path) -> None:
    with pytest.raises(ValueError, match="unsupported grain"):
        aggregate_features(panel_db, grain="bogus")


def test_build_features_carries_grain_unit_when_present(panel_db: Path) -> None:
    panel = aggregate_features(panel_db, grain="store_ppg_week")
    feats = build_features(panel)
    assert "grain_unit" in feats.columns
    # Each (store, PPG) is its own group for lags — first 4 weeks per group drop.
    expected_cells = 3 * 2  # stores x PPGs
    weeks_kept = 10 - 4
    assert len(feats) == expected_cells * weeks_kept


def test_build_features_no_grain_unit_when_absent(panel_db: Path) -> None:
    panel = ppg_week_aggregate(panel_db)
    feats = build_features(panel)
    assert "grain_unit" not in feats.columns
    # 2 PPGs, 10 weeks - 4 lags = 12 rows.
    assert len(feats) == 12
