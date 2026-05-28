"""Query-spec validation and SQL composition for the chart playground.

The playground sends a structured ``QuerySpec`` describing what to read from
``main.panel`` (dimensions, measures, filters, order, limit). This module
validates the spec against :mod:`core.data.column_meta`, builds a
parameterised DuckDB query, and executes it read-only against a per-run
warehouse. No string concatenation of user input — every value is bound
through DuckDB parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, Field, model_validator

from core.data.column_meta import ALLOWED_AGGS, COLUMNS, is_dimension, is_measure

MAX_LIMIT = 50_000


class Measure(BaseModel):
    column: str
    agg: str = "sum"
    alias: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Measure:
        if not is_measure(self.column):
            raise ValueError(f"unknown measure column: {self.column}")
        if self.agg not in ALLOWED_AGGS:
            raise ValueError(f"unsupported aggregation: {self.agg}")
        return self

    def select_expr(self) -> str:
        agg = self.agg.upper().replace("COUNT_DISTINCT", "COUNT(DISTINCT")
        out_alias = self.alias or f"{self.agg}_{self.column}"
        if self.agg == "count_distinct":
            return f'COUNT(DISTINCT "{self.column}") AS "{out_alias}"'
        if self.agg == "count":
            return f'COUNT("{self.column}") AS "{out_alias}"'
        return f'{agg}("{self.column}") AS "{out_alias}"'

    @property
    def output_name(self) -> str:
        return self.alias or f"{self.agg}_{self.column}"


class Dimension(BaseModel):
    column: str
    alias: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Dimension:
        if not is_dimension(self.column):
            raise ValueError(f"unknown dimension column: {self.column}")
        return self

    def select_expr(self) -> str:
        out_alias = self.alias or self.column
        return f'"{self.column}" AS "{out_alias}"'

    @property
    def output_name(self) -> str:
        return self.alias or self.column


class RangeFilter(BaseModel):
    gte: str | float | int | None = None
    lte: str | float | int | None = None


class OrderClause(BaseModel):
    column: str
    asc: bool = True


class QuerySpec(BaseModel):
    dimensions: list[Dimension] = Field(default_factory=list)
    measures: list[Measure] = Field(default_factory=list)
    filters: dict[str, list[Any] | RangeFilter] = Field(default_factory=dict)
    order_by: list[OrderClause] = Field(default_factory=list)
    limit: int = 5_000

    @model_validator(mode="after")
    def _check(self) -> QuerySpec:
        if not self.dimensions and not self.measures:
            raise ValueError("query requires at least one dimension or measure")
        for col in self.filters:
            if col not in COLUMNS:
                raise ValueError(f"unknown filter column: {col}")
        if self.limit <= 0 or self.limit > MAX_LIMIT:
            raise ValueError(f"limit must be in (0, {MAX_LIMIT}]")
        for o in self.order_by:
            valid = {d.output_name for d in self.dimensions} | {m.output_name for m in self.measures}
            if o.column not in valid:
                raise ValueError(f"order_by column '{o.column}' must be a selected dim or measure alias")
        return self


def build_sql(spec: QuerySpec) -> tuple[str, list[Any]]:
    """Compose the SELECT for ``spec``. Returns (sql, params).

    Params are bound positionally through DuckDB's ``?`` placeholders.
    """
    select_parts: list[str] = []
    select_parts.extend(d.select_expr() for d in spec.dimensions)
    select_parts.extend(m.select_expr() for m in spec.measures)
    select_sql = ", ".join(select_parts) or "*"

    where_parts: list[str] = []
    params: list[Any] = []
    for col, val in spec.filters.items():
        if isinstance(val, RangeFilter):
            if val.gte is not None:
                where_parts.append(f'"{col}" >= ?')
                params.append(val.gte)
            if val.lte is not None:
                where_parts.append(f'"{col}" <= ?')
                params.append(val.lte)
        else:
            if not val:
                continue
            placeholders = ", ".join("?" for _ in val)
            where_parts.append(f'"{col}" IN ({placeholders})')
            params.extend(val)
    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    group_sql = ""
    if spec.dimensions and spec.measures:
        group_sql = "GROUP BY " + ", ".join(str(i + 1) for i in range(len(spec.dimensions)))

    order_sql = ""
    if spec.order_by:
        order_sql = "ORDER BY " + ", ".join(
            f'"{o.column}" {"ASC" if o.asc else "DESC"}' for o in spec.order_by
        )

    sql = f'SELECT {select_sql} FROM "main"."panel" {where_sql} {group_sql} {order_sql} LIMIT {spec.limit}'
    return " ".join(sql.split()), params


def execute(spec: QuerySpec, duckdb_path: Path) -> dict[str, Any]:
    """Run ``spec`` against ``duckdb_path`` read-only and return tidy JSON."""
    sql, params = build_sql(spec)
    con = duckdb.connect(str(duckdb_path))
    try:
        cur = con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        con.close()
    serialisable = [
        [v.isoformat() if hasattr(v, "isoformat") else v for v in row]
        for row in rows
    ]
    return {"columns": cols, "rows": serialisable, "n": len(rows)}


def distinct_values(column: str, duckdb_path: Path, limit: int = 500) -> list[Any]:
    """Helper for the filter-builder dropdowns."""
    if column not in COLUMNS:
        raise ValueError(f"unknown column: {column}")
    sql = f'SELECT DISTINCT "{column}" FROM "main"."panel" WHERE "{column}" IS NOT NULL ORDER BY 1 LIMIT {min(limit, MAX_LIMIT)}'
    con = duckdb.connect(str(duckdb_path))
    try:
        rows = con.execute(sql).fetchall()
    finally:
        con.close()
    return [r[0].isoformat() if hasattr(r[0], "isoformat") else r[0] for r in rows]
