"""Decision-support catalogue for the grain selector.

Confirms ``list_grain_options`` reports the right ``available`` /
``recommended`` flags for representative panel shapes.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from core.features.grain_options import list_grain_options


def _seed_panel(tmp_path: Path, *, n_stores: int, brands: list[str], categories: list[str], n_weeks: int) -> Path:
    """Build a tiny in-memory panel exercising one or more axes."""
    rows: list[dict] = []
    weeks = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    store_ids = [f"store_{i:03d}" for i in range(n_stores)]
    for store in store_ids:
        for brand in brands:
            for cat in categories:
                for w in weeks:
                    rows.append(
                        {
                            "store_id": store,
                            "ppg_id": f"{brand}_{cat}",
                            "category": cat,
                            "brand": brand,
                            "sku": f"{brand}_{cat}_a",
                            "week_start": w.date(),
                            "units": 10,
                            "price": 1.0,
                            "base_price": 1.0,
                            "discount_depth": 0.0,
                            "tpr_flag": 0,
                            "display_flag": 0,
                            "feature_flag": 0,
                            "distribution_acv": 100.0,
                            "competitor_price": 1.0,
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


def test_six_grains_returned(tmp_path: Path) -> None:
    db = _seed_panel(
        tmp_path, n_stores=5, brands=["A", "B", "C"], categories=["cat1", "cat2"], n_weeks=40
    )
    opts = list_grain_options(db)
    ids = {o.id for o in opts}
    assert ids == {
        "ppg_week",
        "store_ppg_week",
        "category_week",
        "store_category_week",
        "brand_week",
        "store_brand_week",
    }


def test_brand_grains_unavailable_with_one_brand(tmp_path: Path) -> None:
    db = _seed_panel(
        tmp_path, n_stores=3, brands=["only_one"], categories=["cat1", "cat2"], n_weeks=40
    )
    opts = {o.id: o for o in list_grain_options(db)}
    assert opts["brand_week"].available is False
    assert opts["store_brand_week"].available is False
    assert opts["category_week"].available is True


def test_store_grains_unavailable_with_one_store(tmp_path: Path) -> None:
    db = _seed_panel(
        tmp_path, n_stores=1, brands=["A", "B"], categories=["cat1", "cat2"], n_weeks=40
    )
    opts = {o.id: o for o in list_grain_options(db)}
    assert opts["store_ppg_week"].available is False
    assert opts["store_brand_week"].available is False
    assert opts["store_category_week"].available is False
    assert opts["ppg_week"].available is True


def test_short_panel_marks_grains_unavailable(tmp_path: Path) -> None:
    db = _seed_panel(
        tmp_path, n_stores=3, brands=["A", "B"], categories=["cat1"], n_weeks=10
    )
    opts = list_grain_options(db)
    # Every grain should be unavailable because the panel has <30 weeks
    # of data per cell — there's no statistical leg to stand on.
    assert all(not o.available for o in opts)


def test_recommendation_picks_in_sweet_spot(tmp_path: Path) -> None:
    db = _seed_panel(
        tmp_path, n_stores=10, brands=["A", "B", "C", "D"], categories=["cat1", "cat2"], n_weeks=60
    )
    opts = list_grain_options(db)
    # At most one recommendation per spatial column (chain vs store).
    chain_recs = [o for o in opts if o.recommended and o.spatial_axis == "chain"]
    store_recs = [o for o in opts if o.recommended and o.spatial_axis == "store"]
    assert len(chain_recs) <= 1
    assert len(store_recs) <= 1
    # And the recommendation must be on an available grain with cells
    # inside the configured sweet spot.
    for o in chain_recs + store_recs:
        assert o.available
        assert 25 <= o.expected_cells <= 500


def test_expected_cells_match_axis_product(tmp_path: Path) -> None:
    db = _seed_panel(
        tmp_path, n_stores=4, brands=["A", "B"], categories=["cat1", "cat2"], n_weeks=40
    )
    opts = {o.id: o for o in list_grain_options(db)}
    # store_brand_week = n_stores * n_brands = 4 * 2 = 8
    assert opts["store_brand_week"].expected_cells == 8
    # brand_week (chain) = n_brands = 2
    assert opts["brand_week"].expected_cells == 2
    # store_ppg_week = n_stores * n_ppgs (=4*4 since 2 brands × 2 cats)
    assert opts["store_ppg_week"].expected_cells == 16
