"""Within-PPG-week competitor-price proxy.

When the raw panel ships no competitor series, we compute the
quantity-weighted mean price of the OTHER SKUs in the same PPG-week as
a defensible substitute. This tests the SQL helper itself and the
PPG_WEEK / STORE_PPG_WEEK grain variants.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from core.features.competitor import compute_competitor_proxy


@pytest.fixture
def panel_db(tmp_path: Path) -> Path:
    # Two SKUs per PPG so each has at least one "other" SKU to average over.
    rows = [
        # week 1
        {"store_id": "s1", "ppg_id": "PPG", "sku": "A", "week_start": "2024-01-01",
         "units": 100, "price": 5.0, "base_price": 5.0, "discount_depth": 0.0,
         "tpr_flag": 0, "display_flag": 0, "feature_flag": 0,
         "distribution_acv": 100.0, "competitor_price": None, "holiday": None},
        {"store_id": "s1", "ppg_id": "PPG", "sku": "B", "week_start": "2024-01-01",
         "units": 200, "price": 7.0, "base_price": 7.0, "discount_depth": 0.0,
         "tpr_flag": 0, "display_flag": 0, "feature_flag": 0,
         "distribution_acv": 100.0, "competitor_price": None, "holiday": None},
        # week 2
        {"store_id": "s1", "ppg_id": "PPG", "sku": "A", "week_start": "2024-01-08",
         "units": 100, "price": 4.0, "base_price": 5.0, "discount_depth": 0.2,
         "tpr_flag": 1, "display_flag": 0, "feature_flag": 0,
         "distribution_acv": 100.0, "competitor_price": None, "holiday": None},
        {"store_id": "s1", "ppg_id": "PPG", "sku": "B", "week_start": "2024-01-08",
         "units": 100, "price": 8.0, "base_price": 7.0, "discount_depth": -0.143,
         "tpr_flag": 0, "display_flag": 0, "feature_flag": 0,
         "distribution_acv": 100.0, "competitor_price": None, "holiday": None},
    ]
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


def test_proxy_ppg_week_excludes_focal_sku(panel_db: Path) -> None:
    proxy = compute_competitor_proxy(panel_db, grain="ppg_week")
    # 2 weeks worth of rows.
    assert len(proxy) == 2
    # In week 1: SKU A's competitor mean is just price(B) = 7.0; SKU B's is
    # price(A) = 5.0. Per-PPG-week proxy averages those: (7.0 + 5.0) / 2 = 6.0.
    week1 = proxy[proxy["week_start"].astype(str) == "2024-01-01"]
    assert pytest.approx(float(week1["competitor_price"].iloc[0]), abs=0.01) == 6.0


def test_proxy_store_ppg_week_keys_by_store(panel_db: Path) -> None:
    proxy = compute_competitor_proxy(panel_db, grain="store_ppg_week")
    assert set(proxy["grain_unit"].unique()) == {"s1"}
    # Same per-week count.
    assert len(proxy) == 2


def test_proxy_rejects_unknown_grain(panel_db: Path) -> None:
    with pytest.raises(ValueError, match="unsupported grain"):
        compute_competitor_proxy(panel_db, grain="bogus")
