"""Tests for the Dominick's -> panel-schema loader."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from core.data.loaders.dominicks import (
    WEEK_ANCHOR,
    _week_to_date,
    build_dominicks_panel,
)
from core.data.schema import REQUIRED_COLUMNS, PanelRow


def _write_fixture(tmp_path: Path) -> Path:
    """Hand-crafted yogurt + beer movement/UPC files under tmp_path."""
    raw = tmp_path / "dominicks-raw"
    raw.mkdir()

    # Yogurt — 3 UPCs x 2 stores x 6 weeks, one promo week per UPC
    yog_rows = []
    for upc in (12300000001, 12300000002, 12300000003):
        for store in (8, 9):
            for week in range(1, 7):
                is_promo = week == 4
                yog_rows.append(
                    {
                        "STORE": store,
                        "UPC": upc,
                        "WEEK": week,
                        "MOVE": 50 + week * 3 + (10 if is_promo else 0),
                        "QTY": 1,
                        "PRICE": 0.79 if is_promo else 0.99,
                        "SALE": "B" if is_promo else "",
                        "PROFIT": 22.5,
                        "OK": 1,
                    }
                )
    # Throw in one row with OK=0 to verify it gets filtered.
    yog_rows.append(
        {"STORE": 8, "UPC": 12300000001, "WEEK": 2, "MOVE": 99, "QTY": 1,
         "PRICE": 0.1, "SALE": "", "PROFIT": 0, "OK": 0}
    )
    pd.DataFrame(yog_rows).to_csv(raw / "wyog.csv", index=False)
    pd.DataFrame(
        [
            {"UPC": 12300000001, "DESCRIP": "DANNON FRUIT YOGURT 8OZ", "SIZE": "8 OZ", "COM_CODE": "Y01"},
            {"UPC": 12300000002, "DESCRIP": "YOPLAIT STRAWBERRY 6OZ", "SIZE": "6 OZ", "COM_CODE": "Y01"},
            {"UPC": 12300000003, "DESCRIP": "DANNON PLAIN 32OZ", "SIZE": "32 OZ", "COM_CODE": "Y02"},
        ]
    ).to_csv(raw / "upcyog.csv", index=False)

    # Beer — 2 UPCs x 1 store x 4 weeks, no promos
    beer_rows = []
    for upc in (18200000001, 18200000002):
        for week in range(1, 5):
            beer_rows.append(
                {
                    "STORE": 8,
                    "UPC": upc,
                    "WEEK": week,
                    "MOVE": 20 + week,
                    "QTY": 6,            # 6-pack -> per-unit price = PRICE/6
                    "PRICE": 5.99,
                    "SALE": "",
                    "PROFIT": 18.0,
                    "OK": 1,
                }
            )
    pd.DataFrame(beer_rows).to_csv(raw / "wber.csv", index=False)
    pd.DataFrame(
        [
            {"UPC": 18200000001, "DESCRIP": "BUDWEISER 6PK", "SIZE": "6 PK", "COM_CODE": "B01"},
            {"UPC": 18200000002, "DESCRIP": "MILLER LITE 6PK", "SIZE": "6 PK", "COM_CODE": "B01"},
        ]
    ).to_csv(raw / "upcber.csv", index=False)

    return raw


def test_week_to_date_anchor() -> None:
    assert _week_to_date(1) == WEEK_ANCHOR  # 1989-09-14 (Thursday)
    assert _week_to_date(2) == date(1989, 9, 21)
    assert _week_to_date(53) == date(1990, 9, 13)


def test_loader_produces_panel_schema(tmp_path: Path) -> None:
    raw = _write_fixture(tmp_path)
    panel = build_dominicks_panel(raw, categories=["yogurt", "beer"])

    for col in REQUIRED_COLUMNS:
        assert col in panel.columns, f"missing required column: {col}"

    # OK=0 row was filtered
    assert (panel["price"] > 0).all()
    assert (panel["base_price"] >= panel["price"]).all()
    assert panel["region"].unique().tolist() == ["Chicago"]
    assert set(panel["category"].unique()) == {"Yogurt", "Beer"}

    # tpr_flag: SALE='B' weeks marked promo, others not
    yogurt = panel[panel["category"] == "Yogurt"]
    assert int(yogurt["tpr_flag"].sum()) > 0  # week 4 promos
    assert int(yogurt[yogurt["tpr_flag"] == 1]["price"].max()) < 1.0  # promo price 0.79

    # Per-unit beer price = PRICE / QTY = 5.99 / 6 ~= 0.998
    beer = panel[panel["category"] == "Beer"]
    assert (beer["price"] - 5.99 / 6).abs().max() < 0.01


def test_loader_pydantic_round_trip(tmp_path: Path) -> None:
    raw = _write_fixture(tmp_path)
    panel = build_dominicks_panel(raw, categories=["yogurt"])
    # Sample a few rows and round-trip through the row contract.
    for row in panel.head(5).to_dict(orient="records"):
        PanelRow(**row)


def test_base_price_dominates_promo_weeks(tmp_path: Path) -> None:
    raw = _write_fixture(tmp_path)
    panel = build_dominicks_panel(raw, categories=["yogurt"])
    promo = panel[panel["tpr_flag"] == 1]
    assert (promo["base_price"] > promo["price"]).all(), (
        "base_price should sit above the promo price once the trailing "
        "non-promo max kicks in"
    )


def test_loader_skips_unknown_category(tmp_path: Path) -> None:
    raw = _write_fixture(tmp_path)
    with pytest.raises(KeyError):
        build_dominicks_panel(raw, categories=["unicorn_meat"])


def test_loader_raises_when_no_rows(tmp_path: Path) -> None:
    raw = tmp_path / "empty-raw"
    raw.mkdir()
    with pytest.raises(RuntimeError):
        build_dominicks_panel(raw, categories=["yogurt"])
