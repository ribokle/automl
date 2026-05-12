"""CSV -> DuckDB loader. Lands the raw panel as `main.raw_panel`."""
from __future__ import annotations

from pathlib import Path

import duckdb


def load_csv_to_duckdb(csv_path: Path, duckdb_path: Path, table: str = "raw_panel") -> int:
    """Load a CSV into DuckDB and return the row count."""
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(f"create or replace table main.{table} as select * from read_csv_auto(?, header=true)", [str(csv_path)])
        (rows,) = con.execute(f"select count(*) from main.{table}").fetchone()
        return int(rows)
    finally:
        con.close()
