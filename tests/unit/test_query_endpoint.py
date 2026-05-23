"""End-to-end tests for the chart-playground query endpoint.

Stands up a real DuckDB warehouse from the synthetic panel via the existing
ingestion helpers, then hits the FastAPI route via TestClient and validates
the returned tidy shape.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.data.dbt_runner import run_dbt_build
from core.data.io import load_csv_to_duckdb
from synthetic.generator import write_panel


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    base = tmp_path_factory.mktemp("query_warehouse")
    run_id = "test_run_q1"
    run_dir = base / run_id
    run_dir.mkdir()
    csv = base / "panel.csv"
    write_panel(csv, base / "truth.json", seed=42)
    duckdb_path = run_dir / "warehouse.duckdb"
    load_csv_to_duckdb(csv, duckdb_path)
    run_dbt_build(duckdb_path)
    return base, run_id


@pytest.fixture()
def client(warehouse: tuple[Path, str]) -> TestClient:
    base, _ = warehouse
    os.environ["RUN_DIR"] = str(base)
    return TestClient(app)


def test_schema_endpoint_lists_columns(client: TestClient) -> None:
    res = client.get("/runs/query/schema")
    assert res.status_code == 200
    body = res.json()
    assert "units" in body["measures"]
    assert "category" in body["dimensions"]
    assert "week_start" in body["time_columns"]
    assert "sum" in body["aggregations"]


def test_query_groups_by_dimension(client: TestClient, warehouse: tuple[Path, str]) -> None:
    _, run_id = warehouse
    res = client.post(
        f"/runs/{run_id}/query",
        json={
            "dimensions": [{"column": "category"}],
            "measures": [{"column": "units", "agg": "sum", "alias": "u"}],
            "order_by": [{"column": "u", "asc": False}],
            "limit": 100,
        },
    )
    assert res.status_code == 200, res.json()
    body = res.json()
    assert body["columns"] == ["category", "u"]
    assert body["n"] > 0

    base, _ = warehouse
    con = duckdb.connect(str(base / run_id / "warehouse.duckdb"))
    expected = dict(con.execute('SELECT "category", SUM("units") FROM "main"."panel" GROUP BY 1').fetchall())
    con.close()
    for row in body["rows"]:
        cat, total = row
        assert expected[cat] == total


def test_query_range_filter(client: TestClient, warehouse: tuple[Path, str]) -> None:
    _, run_id = warehouse
    res = client.post(
        f"/runs/{run_id}/query",
        json={
            "dimensions": [{"column": "week_start"}],
            "measures": [{"column": "units", "agg": "sum"}],
            "filters": {"week_start": {"gte": "2023-06-01", "lte": "2023-08-31"}},
        },
    )
    assert res.status_code == 200, res.json()
    weeks = {row[0] for row in res.json()["rows"]}
    assert all("2023-06" <= w <= "2023-08-31" for w in weeks)


def test_query_unknown_column_returns_400(client: TestClient, warehouse: tuple[Path, str]) -> None:
    _, run_id = warehouse
    res = client.post(
        f"/runs/{run_id}/query",
        json={
            "dimensions": [{"column": "not_real"}],
            "measures": [{"column": "units", "agg": "sum"}],
        },
    )
    assert res.status_code == 400


def test_query_dimension_used_as_measure_returns_400(
    client: TestClient, warehouse: tuple[Path, str]
) -> None:
    _, run_id = warehouse
    res = client.post(
        f"/runs/{run_id}/query",
        json={"measures": [{"column": "category", "agg": "sum"}]},
    )
    assert res.status_code == 400


def test_query_missing_warehouse_returns_404(client: TestClient) -> None:
    res = client.post(
        "/runs/no_such_run/query",
        json={"measures": [{"column": "units", "agg": "sum"}]},
    )
    assert res.status_code == 404


def test_distinct_values(client: TestClient, warehouse: tuple[Path, str]) -> None:
    _, run_id = warehouse
    res = client.get(f"/runs/{run_id}/query/distinct?column=category")
    assert res.status_code == 200
    values = res.json()
    assert len(values) > 0
    assert all(isinstance(v, str) for v in values)


def test_distinct_unknown_column_returns_400(
    client: TestClient, warehouse: tuple[Path, str]
) -> None:
    _, run_id = warehouse
    res = client.get(f"/runs/{run_id}/query/distinct?column=not_a_col")
    assert res.status_code == 400


def test_top_n_limit_capped(client: TestClient, warehouse: tuple[Path, str]) -> None:
    _, run_id = warehouse
    res = client.post(
        f"/runs/{run_id}/query",
        json={
            "dimensions": [{"column": "week_start"}],
            "measures": [{"column": "units", "agg": "sum"}],
            "limit": 5,
        },
    )
    assert res.status_code == 200
    assert len(res.json()["rows"]) <= 5
