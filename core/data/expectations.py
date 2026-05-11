"""Code-defined Great Expectations suites for the canonical panel.

Returns lists of instantiated Expectation objects (GE 1.x preferred API)
keyed by suite name.
"""
from __future__ import annotations

from typing import Any

from great_expectations import expectations as gxe


def _volume() -> list[Any]:
    return [
        gxe.ExpectTableRowCountToBeBetween(min_value=500),
        gxe.ExpectColumnUniqueValueCountToBeBetween(column="sku", min_value=5),
        gxe.ExpectColumnUniqueValueCountToBeBetween(column="store_id", min_value=1),
    ]


def _distribution() -> list[Any]:
    return [
        gxe.ExpectColumnMeanToBeBetween(column="price", min_value=0.5, max_value=50.0),
        gxe.ExpectColumnStdevToBeBetween(column="price", min_value=0.0, max_value=20.0),
        gxe.ExpectColumnQuantileValuesToBeBetween(
            column="units",
            quantile_ranges={
                "quantiles": [0.5, 0.95],
                "value_ranges": [[0, 10000], [0, 50000]],
            },
        ),
        gxe.ExpectColumnValuesToBeBetween(
            column="distribution_acv", min_value=0, max_value=100
        ),
    ]


def _relationship() -> list[Any]:
    return [
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="base_price",
            column_B="price",
            or_equal=True,
            mostly=0.99,
        ),
    ]


def all_expectations() -> dict[str, list[Any]]:
    """Return suite name -> list of Expectation instances."""
    return {
        "panel_volume_suite": _volume(),
        "panel_distribution_suite": _distribution(),
        "panel_relationship_suite": _relationship(),
    }
