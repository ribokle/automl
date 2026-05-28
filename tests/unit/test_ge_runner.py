"""Great Expectations runner: error paths and baseline-driven drift suite."""
from __future__ import annotations

from pathlib import Path

import duckdb

from core.data.ge_runner import run_ge_checks


def _make_duckdb(path: Path, *, empty: bool = False) -> None:
    con = duckdb.connect(str(path))
    try:
        schema = """
            CREATE SCHEMA IF NOT EXISTS main;
            CREATE TABLE IF NOT EXISTS main.panel (
                sku VARCHAR,
                week_start DATE,
                store_id VARCHAR,
                region VARCHAR,
                units INTEGER,
                price DOUBLE,
                base_price DOUBLE,
                tpr_flag INTEGER,
                display_flag INTEGER,
                feature_flag INTEGER,
                distribution_acv DOUBLE
            )
        """
        con.execute(schema)
        if not empty:
            con.execute("""
                INSERT INTO main.panel
                SELECT
                    'SKU' || i::VARCHAR AS sku,
                    DATE '2024-01-01' + (i % 104) * INTERVAL '7 days' AS week_start,
                    'S1' AS store_id,
                    'EAST' AS region,
                    100 + (i % 50) AS units,
                    2.99 + (i % 10) * 0.1 AS price,
                    3.49 AS base_price,
                    0 AS tpr_flag,
                    0 AS display_flag,
                    0 AS feature_flag,
                    80.0 AS distribution_acv
                FROM range(600) t(i)
            """)
    finally:
        con.close()


def test_ge_runner_empty_mart_returns_failures(tmp_path: Path) -> None:
    """An empty `panel` table should yield at least one failed expectation
    (row-count check) rather than raising an unhandled exception."""
    db = tmp_path / "warehouse.duckdb"
    _make_duckdb(db, empty=True)

    results = run_ge_checks(db, table="panel")

    # The volume suite includes ExpectTableRowCountToBeBetween(min_value=500)
    # which must fail on an empty table.
    assert isinstance(results, list)
    assert any(r.status == "fail" for r in results), (
        "Expected at least one failed check for an empty panel table"
    )


def test_ge_runner_healthy_mart_mostly_passes(tmp_path: Path) -> None:
    """A well-formed panel should produce mostly-passing checks."""
    db = tmp_path / "warehouse.duckdb"
    _make_duckdb(db, empty=False)

    results = run_ge_checks(db, table="panel")

    assert isinstance(results, list)
    passed = sum(1 for r in results if r.status == "pass")
    assert passed > 0


def test_ge_runner_with_baseline_adds_drift_suite(tmp_path: Path) -> None:
    """When a baseline JSON exists, the drift suite produces check results."""
    import json

    db = tmp_path / "warehouse.duckdb"
    _make_duckdb(db, empty=False)

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "row_count": 600,
        "columns": {
            "price": {"mean": 3.44, "std": 0.3, "q25": 3.0, "q50": 3.44, "q75": 3.89, "q95": 4.2},
            "units": {"mean": 124.5, "std": 15.0, "q25": 100.0, "q50": 124.0, "q75": 148.0, "q95": 149.0},
        },
    }))

    results_without = run_ge_checks(db, table="panel")
    results_with = run_ge_checks(db, table="panel", baseline_path=baseline)

    assert len(results_with) > len(results_without), (
        "Drift suite should add extra check results when baseline is present"
    )
