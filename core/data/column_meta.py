"""Per-column metadata used by the chart-playground query endpoint.

One module owns the playground's allow-list so the FastAPI route and the
frontend column pickers never disagree on which columns exist or what role
each one plays. Sourced from :mod:`core.data.schema` plus ``discount_depth``
which is materialised by the dbt mart (`dbt/automl_dbt/models/marts/panel.sql`)
but isn't in the raw schema.
"""
from __future__ import annotations

from typing import Literal, TypedDict


ColumnRole = Literal["dimension", "measure", "time"]
ColumnUnit = Literal["count", "dollars", "share", "ratio", "category", "date", "id"]


class ColumnMeta(TypedDict):
    role: ColumnRole
    unit: ColumnUnit
    label: str
    default_agg: str


COLUMNS: dict[str, ColumnMeta] = {
    "week_start": {"role": "time", "unit": "date", "label": "Week", "default_agg": "min"},
    "sku":        {"role": "dimension", "unit": "id",       "label": "SKU",        "default_agg": "count_distinct"},
    "store_id":   {"role": "dimension", "unit": "id",       "label": "Store",      "default_agg": "count_distinct"},
    "region":     {"role": "dimension", "unit": "category", "label": "Region",     "default_agg": "count_distinct"},
    "category":   {"role": "dimension", "unit": "category", "label": "Category",   "default_agg": "count_distinct"},
    "brand":      {"role": "dimension", "unit": "category", "label": "Brand",      "default_agg": "count_distinct"},
    "pack_size":  {"role": "dimension", "unit": "category", "label": "Pack size",  "default_agg": "count_distinct"},
    "segment":    {"role": "dimension", "unit": "category", "label": "Segment",    "default_agg": "count_distinct"},
    "ppg_id":     {"role": "dimension", "unit": "id",       "label": "PPG",        "default_agg": "count_distinct"},
    "holiday":    {"role": "dimension", "unit": "category", "label": "Holiday",    "default_agg": "count_distinct"},
    "tpr_flag":     {"role": "dimension", "unit": "category", "label": "TPR flag",     "default_agg": "avg"},
    "display_flag": {"role": "dimension", "unit": "category", "label": "Display flag", "default_agg": "avg"},
    "feature_flag": {"role": "dimension", "unit": "category", "label": "Feature flag", "default_agg": "avg"},
    "units":            {"role": "measure", "unit": "count",   "label": "Units",            "default_agg": "sum"},
    "price":            {"role": "measure", "unit": "dollars", "label": "Price",            "default_agg": "avg"},
    "base_price":       {"role": "measure", "unit": "dollars", "label": "Base price",       "default_agg": "avg"},
    "discount_depth":   {"role": "measure", "unit": "share",   "label": "Discount depth",   "default_agg": "avg"},
    "distribution_acv": {"role": "measure", "unit": "share",   "label": "Distribution ACV", "default_agg": "avg"},
    "competitor_price": {"role": "measure", "unit": "dollars", "label": "Competitor price", "default_agg": "avg"},
}

DIMENSIONS: list[str] = [c for c, m in COLUMNS.items() if m["role"] in ("dimension", "time")]
MEASURES: list[str] = [c for c, m in COLUMNS.items() if m["role"] == "measure"]
TIME_COLUMNS: list[str] = [c for c, m in COLUMNS.items() if m["role"] == "time"]

ALLOWED_AGGS: tuple[str, ...] = (
    "sum", "avg", "min", "max", "median", "count", "count_distinct",
)


def is_dimension(col: str) -> bool:
    return col in COLUMNS and COLUMNS[col]["role"] in ("dimension", "time")


def is_measure(col: str) -> bool:
    return col in COLUMNS and COLUMNS[col]["role"] == "measure"
