"""Profiling tools for the ingestion agent.

Each tool is a plain function that talks to DuckDB; the agent composes them
to build a structured profile of the canonical panel before narrating
findings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


def _connect(duckdb_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(duckdb_path))


def profile_table(duckdb_path: Path, table: str = "panel") -> dict[str, Any]:
    """Return row count, column count, and per-column summary stats."""
    con = _connect(duckdb_path)
    try:
        n_rows = con.execute(f"select count(*) from main.{table}").fetchone()[0]
        cols = con.execute(
            f"select column_name, data_type from information_schema.columns "
            f"where table_schema='main' and table_name='{table}' order by ordinal_position"
        ).fetchall()
        columns: list[dict[str, Any]] = []
        for name, dtype in cols:
            null_pct = con.execute(
                f'select 100.0 * sum(case when "{name}" is null then 1 else 0 end) / count(*) '
                f"from main.{table}"
            ).fetchone()[0]
            n_unique = con.execute(f'select count(distinct "{name}") from main.{table}').fetchone()[0]
            entry: dict[str, Any] = {
                "name": name,
                "dtype": dtype,
                "null_pct": round(float(null_pct or 0), 3),
                "n_unique": int(n_unique),
            }
            if any(t in dtype.upper() for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "BIGINT")):
                stats = con.execute(
                    f'select min("{name}"), max("{name}"), avg("{name}"), stddev_samp("{name}") '
                    f"from main.{table}"
                ).fetchone()
                entry.update(
                    {
                        "min": float(stats[0]) if stats[0] is not None else None,
                        "max": float(stats[1]) if stats[1] is not None else None,
                        "mean": float(stats[2]) if stats[2] is not None else None,
                        "std": float(stats[3]) if stats[3] is not None else None,
                    }
                )
            columns.append(entry)
        return {"table": table, "row_count": int(n_rows), "columns": columns}
    finally:
        con.close()


def sample_rows(duckdb_path: Path, table: str = "panel", n: int = 10) -> list[dict[str, Any]]:
    con = _connect(duckdb_path)
    try:
        df = con.execute(f"select * from main.{table} using sample {int(n)} rows").df()
        return df.to_dict(orient="records")
    finally:
        con.close()


def column_distribution(
    duckdb_path: Path,
    column: str,
    table: str = "panel",
    bins: int = 10,
) -> dict[str, Any]:
    """Equal-width histogram for a numeric column."""
    con = _connect(duckdb_path)
    try:
        lo, hi = con.execute(f'select min("{column}"), max("{column}") from main.{table}').fetchone()
        if lo is None or hi is None or hi == lo:
            return {"column": column, "bins": [], "counts": []}
        width = (hi - lo) / bins
        edges = [lo + i * width for i in range(bins + 1)]
        counts: list[int] = []
        for i in range(bins):
            lower, upper = edges[i], edges[i + 1]
            op = "<=" if i == bins - 1 else "<"
            n = con.execute(
                f'select count(*) from main.{table} '
                f'where "{column}" >= {lower} and "{column}" {op} {upper}'
            ).fetchone()[0]
            counts.append(int(n))
        return {"column": column, "edges": [float(x) for x in edges], "counts": counts}
    finally:
        con.close()


def detect_outliers(duckdb_path: Path, column: str, table: str = "panel") -> dict[str, Any]:
    """IQR-based outlier count for a numeric column."""
    con = _connect(duckdb_path)
    try:
        row = con.execute(
            f'select quantile_cont("{column}", 0.25), quantile_cont("{column}", 0.75) '
            f"from main.{table}"
        ).fetchone()
        q1, q3 = float(row[0]), float(row[1])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = con.execute(
            f'select count(*) from main.{table} '
            f'where "{column}" < {lo} or "{column}" > {hi}'
        ).fetchone()[0]
        n_total = con.execute(f"select count(*) from main.{table}").fetchone()[0]
        return {
            "column": column,
            "q1": q1,
            "q3": q3,
            "lower": lo,
            "upper": hi,
            "n_outliers": int(n_out),
            "outlier_pct": round(100.0 * n_out / n_total, 3) if n_total else 0.0,
        }
    finally:
        con.close()
