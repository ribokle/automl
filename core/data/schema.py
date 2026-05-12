"""Pydantic row models for the canonical panel.

Column-level validation lives in dbt (`dbt/automl_dbt/models/*.yml`) and the
Great Expectations suites. These Pydantic models exist only for in-process
boundaries (tool inputs, API payloads) where a typed row is convenient.
"""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


REQUIRED_COLUMNS: tuple[str, ...] = (
    "sku",
    "week_start",
    "store_id",
    "region",
    "units",
    "price",
    "base_price",
    "tpr_flag",
    "display_flag",
    "feature_flag",
    "distribution_acv",
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "holiday",
    "competitor_price",
    "category",
    "brand",
    "pack_size",
    "segment",
    "ppg_id",
)


class PanelRow(BaseModel):
    sku: str
    week_start: date
    store_id: str
    region: str
    units: int = Field(ge=0)
    price: float = Field(gt=0)
    base_price: float = Field(gt=0)
    tpr_flag: int = Field(ge=0, le=1)
    display_flag: int = Field(ge=0, le=1)
    feature_flag: int = Field(ge=0, le=1)
    distribution_acv: float = Field(ge=0, le=100)
    holiday: str | None = None
    competitor_price: float | None = None
    category: str | None = None
    brand: str | None = None
    pack_size: str | None = None
    segment: str | None = None
    ppg_id: str | None = None
