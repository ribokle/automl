"""Chart-playground query endpoint.

Serves arbitrary aggregations against ``main.panel`` in the per-run
DuckDB, validated against the column allow-list in
:mod:`core.data.column_meta`. The advanced-EDA dashboard's playground UI
posts ``QuerySpec`` JSON here and renders the returned tidy table.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError

from api.auth import require_auth
from api.deps import get_run_dir
from core.data.column_meta import ALLOWED_AGGS, COLUMNS, DIMENSIONS, MEASURES, TIME_COLUMNS
from core.data.query import QuerySpec, distinct_values, execute

router = APIRouter(prefix="/runs", tags=["query"], dependencies=[Depends(require_auth)])


@router.get("/query/schema")
def schema() -> dict:
    """Column allow-list + aggregation menu for the playground UI."""
    return {
        "columns": COLUMNS,
        "dimensions": DIMENSIONS,
        "measures": MEASURES,
        "time_columns": TIME_COLUMNS,
        "aggregations": list(ALLOWED_AGGS),
    }


@router.post("/{run_id}/query")
def run_query(run_id: str, spec: dict) -> dict:
    base = (get_run_dir() / run_id).resolve()
    duckdb_path = base / "warehouse.duckdb"
    try:
        base.relative_to(get_run_dir().resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid run id") from None
    if not duckdb_path.exists():
        raise HTTPException(status_code=404, detail="run warehouse not found")
    try:
        parsed = QuerySpec.model_validate(spec)
    except ValidationError as e:
        msgs = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise HTTPException(status_code=400, detail="; ".join(msgs)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    try:
        return execute(parsed, duckdb_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{run_id}/query/distinct")
def run_distinct(run_id: str, column: str = Query(...)) -> list:
    base = (get_run_dir() / run_id).resolve()
    duckdb_path = base / "warehouse.duckdb"
    try:
        base.relative_to(get_run_dir().resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid run id") from None
    if not duckdb_path.exists():
        raise HTTPException(status_code=404, detail="run warehouse not found")
    try:
        return distinct_values(column, duckdb_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
