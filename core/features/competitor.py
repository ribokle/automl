"""Within-PPG-week competitor-price proxy.

The canonical schema carries a ``competitor_price`` column for each
(SKU, store, week) row, but most scanner-data loaders (Dominick's
included) have no separate competitor series and set the column to
NaN. Without a competitor signal the engineered ``log_price_gap``
collapses to zero and the modelling stage loses the only feature that
distinguishes "we priced expensively" from "the whole category went
up".

This module supplies a defensible proxy: for each (PPG, week), the
quantity-weighted mean price of the OTHER SKUs in the same PPG (and
store, when available). Falls back to the chain-week mean across the
PPG when a (store, week) cell has only one SKU.

This is a per-PPG-week (or per-store-PPG-week) aggregate, not a
per-row substitute — it lives next to the modelling features, not in
the panel mart.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def compute_competitor_proxy(
    duckdb_path: Path,
    *,
    grain: str = "ppg_week",
    table: str = "main.panel",
) -> pd.DataFrame:
    """Return a frame keyed by (grain_unit, ppg_id, week_start) carrying
    a ``competitor_price`` column = chain-week-PPG mean of OTHER SKUs.

    ``grain`` is one of ``ppg_week`` (grain_unit = ``"chain"``) or
    ``store_ppg_week`` (grain_unit = the store_id).
    """
    if grain == "ppg_week":
        sql = f"""
        WITH base AS (
            SELECT
              ppg_id,
              week_start,
              sku,
              SUM(units) AS sku_units,
              SUM(units * price) AS sku_revenue
            FROM {table}
            WHERE ppg_id IS NOT NULL AND price > 0 AND units > 0
            GROUP BY 1, 2, 3
        ),
        totals AS (
            SELECT
              ppg_id,
              week_start,
              SUM(sku_units) AS total_units,
              SUM(sku_revenue) AS total_revenue
            FROM base
            GROUP BY 1, 2
        ),
        excl AS (
            SELECT
              b.ppg_id,
              b.week_start,
              b.sku,
              (t.total_revenue - b.sku_revenue) / NULLIF(t.total_units - b.sku_units, 0)
                AS competitor_price_sku
            FROM base b
            JOIN totals t USING (ppg_id, week_start)
        )
        SELECT
          'chain' AS grain_unit,
          ppg_id,
          week_start,
          AVG(competitor_price_sku) AS competitor_price
        FROM excl
        WHERE competitor_price_sku IS NOT NULL
        GROUP BY 1, 2, 3
        ORDER BY 2, 3
        """
    elif grain == "store_ppg_week":
        sql = f"""
        WITH base AS (
            SELECT
              store_id,
              ppg_id,
              week_start,
              sku,
              SUM(units) AS sku_units,
              SUM(units * price) AS sku_revenue
            FROM {table}
            WHERE ppg_id IS NOT NULL AND price > 0 AND units > 0
            GROUP BY 1, 2, 3, 4
        ),
        totals AS (
            SELECT
              store_id, ppg_id, week_start,
              SUM(sku_units) AS total_units,
              SUM(sku_revenue) AS total_revenue
            FROM base
            GROUP BY 1, 2, 3
        ),
        excl AS (
            SELECT
              b.store_id, b.ppg_id, b.week_start, b.sku,
              (t.total_revenue - b.sku_revenue) / NULLIF(t.total_units - b.sku_units, 0)
                AS competitor_price_sku
            FROM base b
            JOIN totals t USING (store_id, ppg_id, week_start)
        )
        SELECT
          store_id AS grain_unit,
          ppg_id,
          week_start,
          AVG(competitor_price_sku) AS competitor_price
        FROM excl
        WHERE competitor_price_sku IS NOT NULL
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    else:
        raise ValueError(f"unsupported grain {grain!r} for competitor proxy")

    # Use the default read/write open: opening with read_only=True conflicts
    # with the read/write handles other agents in the same process hold.
    # The SQL is a SELECT only.
    con = duckdb.connect(str(duckdb_path))
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def coverage(proxy: pd.DataFrame, n_total: int) -> float:
    """Fraction of expected (grain_unit, ppg, week) cells with a proxy value."""
    if n_total <= 0:
        return 0.0
    return float(len(proxy) / n_total)
