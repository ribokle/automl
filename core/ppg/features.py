"""Per-SKU feature aggregation for PPG clustering.

Aggregates the weekly panel down to one row per SKU with the attributes that
characterise a Price-Pack Group: brand, category, typical pack tier, typical
price, and volume.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


PACK_ORDINAL: dict[str, int] = {"small": 1, "medium": 2, "large": 3}


def aggregate_sku_features(duckdb_path: Path, table: str = "panel") -> pd.DataFrame:
    """Return one row per SKU with the columns used for PPG clustering.

    Columns: sku, brand, category, pack_size, pack_ordinal, median_price,
    median_base_price, total_units, n_weeks, n_stores.
    """
    con = duckdb.connect(str(duckdb_path))
    try:
        df = con.execute(
            f"""
            select
              sku,
              any_value(brand)            as brand,
              any_value(category)         as category,
              any_value(pack_size)        as pack_size,
              median(price)               as median_price,
              median(base_price)          as median_base_price,
              sum(units)                  as total_units,
              count(distinct week_start)  as n_weeks,
              count(distinct store_id)    as n_stores
            from main.{table}
            group by sku
            order by sku
            """
        ).df()
    finally:
        con.close()
    df["pack_ordinal"] = df["pack_size"].map(PACK_ORDINAL).fillna(2).astype(int)
    return df
