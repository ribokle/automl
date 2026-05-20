"""Tests for the published elasticity benchmark table."""
from __future__ import annotations

import pytest

from core.benchmarks.elasticity import (
    classify,
    load_elasticity_benchmarks,
    lookup_category,
)


def test_table_loads_with_known_categories() -> None:
    table = load_elasticity_benchmarks()
    assert -3.0 < table.grand_mean < -2.0  # Bijmolt 2005 grand mean is -2.62
    for key in ("soft_drinks", "beer", "yogurt", "cookies", "cheese"):
        assert key in table.categories
        b = table.categories[key]
        assert b.lo < b.mean < b.hi


def test_lookup_exact_keys() -> None:
    for key in ("yogurt", "beer", "soft_drinks"):
        b = lookup_category(key)
        assert b is not None, f"expected hit for {key}"
        assert b.category_key == key


def test_lookup_resolves_aliases() -> None:
    # Synthetic generator uses "Soda" / "Juice" / "Water" labels — verify
    # the alias table bridges those to the JMR keys.
    soda = lookup_category("Soda")
    assert soda is not None
    assert soda.category_key == "soft_drinks"

    juice = lookup_category("Juice")
    assert juice is not None
    assert juice.category_key == "bottled_juice"


def test_lookup_unknown_returns_none() -> None:
    assert lookup_category("rocket_fuel") is None
    assert lookup_category(None) is None
    assert lookup_category("") is None


@pytest.mark.parametrize(
    "elasticity, expected",
    [
        (-3.18, "in_band"),
        (-2.5, "in_band"),
        (-5.0, "out_band_high"),  # more elastic than band
        (-0.5, "out_band_low"),   # less elastic than band
    ],
)
def test_classify_soft_drinks(elasticity: float, expected: str) -> None:
    b = lookup_category("soft_drinks")
    assert b is not None
    assert classify(elasticity, b) == expected


def test_classify_no_benchmark() -> None:
    assert classify(-2.0, None) == "no_benchmark"


def test_classify_nan_elasticity_treated_as_no_benchmark() -> None:
    b = lookup_category("soft_drinks")
    assert b is not None
    assert classify(float("nan"), b) == "no_benchmark"
