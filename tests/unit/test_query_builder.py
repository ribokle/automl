"""Unit tests for the chart-playground QuerySpec validator + SQL builder."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.data.query import Dimension, Measure, QuerySpec, RangeFilter, build_sql


def test_unknown_measure_rejected() -> None:
    with pytest.raises(ValidationError):
        Measure.model_validate({"column": "not_a_column", "agg": "sum"})


def test_unknown_aggregation_rejected() -> None:
    with pytest.raises(ValidationError):
        Measure.model_validate({"column": "units", "agg": "geometric_mean"})


def test_dimension_used_as_measure_rejected() -> None:
    with pytest.raises(ValidationError):
        Measure.model_validate({"column": "category", "agg": "sum"})


def test_measure_used_as_dimension_rejected() -> None:
    with pytest.raises(ValidationError):
        Dimension.model_validate({"column": "units"})


def test_empty_spec_rejected() -> None:
    with pytest.raises(ValidationError):
        QuerySpec.model_validate({})


def test_unknown_filter_column_rejected() -> None:
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(
            {
                "measures": [{"column": "units", "agg": "sum"}],
                "filters": {"not_a_col": ["x"]},
            }
        )


def test_unknown_limit_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(
            {"measures": [{"column": "units", "agg": "sum"}], "limit": 0}
        )
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(
            {"measures": [{"column": "units", "agg": "sum"}], "limit": 10**9}
        )


def test_order_by_must_reference_selected_alias() -> None:
    with pytest.raises(ValidationError):
        QuerySpec.model_validate(
            {
                "dimensions": [{"column": "category"}],
                "measures": [{"column": "units", "agg": "sum"}],
                "order_by": [{"column": "brand", "asc": True}],
            }
        )


def test_sql_has_groupby_when_both_dims_and_measures() -> None:
    spec = QuerySpec.model_validate(
        {
            "dimensions": [{"column": "category"}, {"column": "brand"}],
            "measures": [{"column": "units", "agg": "sum"}],
        }
    )
    sql, params = build_sql(spec)
    assert "GROUP BY 1, 2" in sql
    assert params == []


def test_sql_lists_filter_binds_positional_params() -> None:
    spec = QuerySpec.model_validate(
        {
            "measures": [{"column": "units", "agg": "sum"}],
            "filters": {"category": ["soda", "chips", "cereal"]},
        }
    )
    sql, params = build_sql(spec)
    assert 'IN (?, ?, ?)' in sql
    assert params == ["soda", "chips", "cereal"]


def test_sql_range_filter_binds_between() -> None:
    spec = QuerySpec.model_validate(
        {
            "dimensions": [{"column": "week_start"}],
            "measures": [{"column": "units", "agg": "sum"}],
            "filters": {
                "week_start": RangeFilter(gte="2023-01-01", lte="2023-06-01").model_dump()
            },
        }
    )
    sql, params = build_sql(spec)
    assert '"week_start" >= ?' in sql
    assert '"week_start" <= ?' in sql
    assert params == ["2023-01-01", "2023-06-01"]


def test_count_distinct_renders_correctly() -> None:
    spec = QuerySpec.model_validate(
        {"measures": [{"column": "units", "agg": "count_distinct", "alias": "n_units"}]}
    )
    sql, _ = build_sql(spec)
    assert 'COUNT(DISTINCT "units") AS "n_units"' in sql


def test_alias_used_in_select() -> None:
    spec = QuerySpec.model_validate(
        {
            "dimensions": [{"column": "week_start", "alias": "wk"}],
            "measures": [{"column": "units", "agg": "sum", "alias": "U"}],
        }
    )
    sql, _ = build_sql(spec)
    assert '"week_start" AS "wk"' in sql
    assert 'SUM("units") AS "U"' in sql


def test_limit_capped() -> None:
    spec = QuerySpec.model_validate(
        {"measures": [{"column": "units", "agg": "sum"}], "limit": 25}
    )
    sql, _ = build_sql(spec)
    assert sql.endswith("LIMIT 25")
