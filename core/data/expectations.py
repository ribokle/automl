"""Code-defined Great Expectations suites for the canonical panel.

Returns lists of instantiated Expectation objects (GE 1.x preferred API)
keyed by suite name.
"""
from __future__ import annotations

import json
from pathlib import Path
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


def _drift(baseline_path: Path | None) -> list[Any]:
    """Build drift expectations from a previously captured baseline snapshot.

    Returns an empty list when baseline_path is None or the file is absent
    (first-run skip — no baseline exists yet).
    """
    if baseline_path is None:
        return []
    try:
        baseline = json.loads(Path(baseline_path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    exps: list[Any] = []
    for col, stats in baseline.get("columns", {}).items():
        mean = stats.get("mean")
        if mean is not None and mean > 0:
            slack = abs(mean) * 0.4
            exps.append(
                gxe.ExpectColumnMeanToBeBetween(
                    column=col, min_value=mean - slack, max_value=mean + slack
                )
            )
        q25 = stats.get("q25")
        q75 = stats.get("q75")
        if q25 is not None and q75 is not None:
            slack25 = max(abs(q25) * 0.4, 0.01)
            slack75 = max(abs(q75) * 0.4, 0.01)
            exps.append(
                gxe.ExpectColumnQuantileValuesToBeBetween(
                    column=col,
                    quantile_ranges={
                        "quantiles": [0.25, 0.75],
                        "value_ranges": [
                            [q25 - slack25, q25 + slack25],
                            [q75 - slack75, q75 + slack75],
                        ],
                    },
                )
            )
    return exps


def _anomaly() -> list[Any]:
    return [
        # units must be non-negative (duplicate safety net alongside dbt test)
        gxe.ExpectColumnValuesToBeBetween(column="units", min_value=0),
        # price must stay positive
        gxe.ExpectColumnValuesToBeBetween(column="price", min_value=0.01),
        # distribution_acv bounded — catch sudden 0/100 spikes
        gxe.ExpectColumnValuesToBeBetween(
            column="distribution_acv", min_value=0, max_value=100, mostly=0.99
        ),
    ]


def all_expectations(baseline_path: Path | None = None) -> dict[str, list[Any]]:
    """Return suite name -> list of Expectation instances.

    ``baseline_path`` is the JSON file written by ``capture_baseline()``.
    When absent, the drift suite is empty (first-run skip).
    """
    return {
        "panel_volume_suite": _volume(),
        "panel_distribution_suite": _distribution(),
        "panel_relationship_suite": _relationship(),
        "panel_drift_suite": _drift(baseline_path),
        "panel_anomaly_suite": _anomaly(),
    }
