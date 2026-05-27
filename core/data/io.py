"""Panel -> DuckDB loaders. Land the raw panel as `main.raw_panel`."""
from __future__ import annotations

from pathlib import Path

import duckdb


def load_csv_to_duckdb(csv_path: Path, duckdb_path: Path, table: str = "raw_panel") -> int:
    """Load a CSV into DuckDB and return the row count."""
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(f"create or replace table main.{table} as select * from read_csv_auto(?, header=true)", [str(csv_path)])
        # ppg_id is optional in the schema (filled in later by the ppg_mapping
        # agent). Ensure the column exists so downstream dbt casts don't fail.
        con.execute(f"alter table main.{table} add column if not exists ppg_id varchar")
        (rows,) = con.execute(f"select count(*) from main.{table}").fetchone()
        return int(rows)
    finally:
        con.close()


def load_parquet_dir_to_duckdb(parquet_dir: Path, duckdb_path: Path, table: str = "raw_panel") -> int:
    """Load a directory of per-category Parquet files into DuckDB.

    Reads all ``*.parquet`` files under ``parquet_dir`` in a single DuckDB
    ``read_parquet`` glob — schema is embedded in the files so no type
    inference overhead.  Roughly 10-20x faster than the CSV path for the
    same row count.
    """
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    glob = (parquet_dir / "*.parquet").as_posix()
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(
            f"CREATE OR REPLACE TABLE main.{table} AS SELECT * FROM read_parquet(?)",
            [glob],
        )
        con.execute(f"ALTER TABLE main.{table} ADD COLUMN IF NOT EXISTS ppg_id VARCHAR")
        (rows,) = con.execute(f"SELECT count(*) FROM main.{table}").fetchone()
        return int(rows)
    finally:
        con.close()
