"""Decision-support for the modelling-grain selector.

After ingestion + ppg_mapping land, the UI shows the operator a grid of
the 6 supported grain combinations (chain × {PPG, category, brand}
× week, with or without the store axis). For each grain we report:

- ``expected_cells``: how many (grain_unit, ppg_id) pairs the
  modelling agent will iterate over. This is the dominant driver of
  pipeline runtime, so the operator should see it before they pick.
- ``available``: ``False`` when the underlying data is degenerate for
  this axis pair — e.g. a single-brand panel can't usefully fit
  ``brand_week``, a single-store dataset can't fit any ``store_*``
  grain.
- ``recommended``: a heuristic flag for the cell count sweet spot
  (25–500 cells). At most one recommended per axis pair so the UI
  has clear defaults.

The grain catalogue, including the recommended-band, lives here so
the UI never has to encode pipeline policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from core.config import ModellingGrain


_PRODUCT_LABELS = {"ppg": "PPG", "category": "Category", "brand": "Brand"}
_SPATIAL_LABELS = {"chain": "Chain", "store": "Store"}

# Axes for each grain id. Tuple is (spatial, product).
_GRAIN_AXES: dict[ModellingGrain, tuple[str, str]] = {
    ModellingGrain.PPG_WEEK: ("chain", "ppg"),
    ModellingGrain.STORE_PPG_WEEK: ("store", "ppg"),
    ModellingGrain.CATEGORY_WEEK: ("chain", "category"),
    ModellingGrain.STORE_CATEGORY_WEEK: ("store", "category"),
    ModellingGrain.BRAND_WEEK: ("chain", "brand"),
    ModellingGrain.STORE_BRAND_WEEK: ("store", "brand"),
}

# Cell-count sweet spot for the recommendation flag. Below 25 the
# modelling stage will likely skip most cells (each needs >=20 rows);
# above 500 the run gets slow and unwieldy in the UI.
_RECOMMEND_LOW = 25
_RECOMMEND_HIGH = 500


@dataclass(frozen=True)
class GrainOption:
    id: str               # ModellingGrain value, e.g. "ppg_week"
    label: str            # display label, e.g. "Store × Brand × week"
    spatial_axis: str     # "chain" | "store"
    product_axis: str     # "ppg" | "category" | "brand"
    expected_cells: int   # number of (grain_unit, ppg_id) pairs
    expected_rows_per_cell: float  # median weeks per cell (approx)
    available: bool
    recommended: bool
    reason: str           # one-line "why available / not"


@dataclass(frozen=True)
class PanelShape:
    n_stores: int
    n_ppgs: int
    n_categories: int
    n_brands: int
    n_weeks: int


def _query_shape(duckdb_path: Path, table: str = "main.panel") -> PanelShape:
    con = duckdb.connect(str(duckdb_path))
    try:
        row = con.execute(
            f"""
            SELECT
              COUNT(DISTINCT store_id),
              COUNT(DISTINCT ppg_id),
              COUNT(DISTINCT category),
              COUNT(DISTINCT brand),
              COUNT(DISTINCT week_start)
            FROM {table}
            """
        ).fetchone()
    finally:
        con.close()
    return PanelShape(
        n_stores=int(row[0] or 0),
        n_ppgs=int(row[1] or 0),
        n_categories=int(row[2] or 0),
        n_brands=int(row[3] or 0),
        n_weeks=int(row[4] or 0),
    )


def _product_count(shape: PanelShape, product_axis: str) -> int:
    if product_axis == "ppg":
        return shape.n_ppgs
    if product_axis == "category":
        return shape.n_categories
    if product_axis == "brand":
        return shape.n_brands
    raise ValueError(f"unknown product axis {product_axis!r}")


def _spatial_count(shape: PanelShape, spatial_axis: str) -> int:
    return shape.n_stores if spatial_axis == "store" else 1


def _grain_label(spatial: str, product: str) -> str:
    parts: list[str] = []
    if spatial == "store":
        parts.append(_SPATIAL_LABELS["store"])
    parts.append(_PRODUCT_LABELS[product])
    parts.append("week")
    return " × ".join(parts)


def list_grain_options(duckdb_path: Path, table: str = "main.panel") -> list[GrainOption]:
    """Build the 6-entry catalogue the UI selector renders.

    Order matches the visual grid the operator sees: rows are product
    axis (PPG → Category → Brand), columns are spatial axis (chain →
    store). The first recommended grain per spatial column wins (so the
    operator gets clear defaults without competing recommendations).
    """
    shape = _query_shape(duckdb_path, table)
    options: list[GrainOption] = []

    for grain, (spatial, product) in _GRAIN_AXES.items():
        spatial_n = _spatial_count(shape, spatial)
        product_n = _product_count(shape, product)
        expected = int(spatial_n * product_n)
        available = product_n >= 2 and spatial_n >= 1
        if spatial == "store" and shape.n_stores < 2:
            available = False
            reason = "needs >=2 stores"
        elif product_n < 2:
            reason = f"only {product_n} {product} value in this panel"
            available = False
        elif shape.n_weeks < 30:
            available = False
            reason = f"only {shape.n_weeks} weeks of data"
        else:
            reason = (
                f"{spatial_n} × {product_n} cells; "
                f"~{shape.n_weeks} weeks per cell"
            )

        options.append(
            GrainOption(
                id=grain.value,
                label=_grain_label(spatial, product),
                spatial_axis=spatial,
                product_axis=product,
                expected_cells=expected,
                expected_rows_per_cell=float(shape.n_weeks),
                available=available,
                recommended=False,  # filled below per spatial column
                reason=reason,
            )
        )

    # Mark a single recommendation per spatial column. Pick the
    # available grain whose expected_cells sits in [low, high]; if
    # multiple, prefer the one closest to the geometric mean.
    def _score(opt: GrainOption) -> float:
        if not opt.available:
            return float("inf")
        if not (_RECOMMEND_LOW <= opt.expected_cells <= _RECOMMEND_HIGH):
            return float("inf")
        # log-distance from the sweet-spot midpoint
        from math import log
        midpoint = (_RECOMMEND_LOW * _RECOMMEND_HIGH) ** 0.5
        return abs(log(opt.expected_cells) - log(midpoint))

    recommended: list[GrainOption] = []
    for spatial in ("chain", "store"):
        column = [o for o in options if o.spatial_axis == spatial]
        column.sort(key=_score)
        if column and _score(column[0]) != float("inf"):
            recommended.append(column[0])

    flagged = {id(o) for o in recommended}
    return [
        # frozen=True so rebuild a new instance with recommended set
        GrainOption(
            id=o.id,
            label=o.label,
            spatial_axis=o.spatial_axis,
            product_axis=o.product_axis,
            expected_cells=o.expected_cells,
            expected_rows_per_cell=o.expected_rows_per_cell,
            available=o.available,
            recommended=(id(o) in flagged),
            reason=o.reason,
        )
        for o in options
    ]


def to_payload(options: list[GrainOption], shape: PanelShape | None = None) -> dict:
    """JSON-serialisable payload for the artifact + UI."""
    return {
        "shape": (
            {
                "n_stores": shape.n_stores,
                "n_ppgs": shape.n_ppgs,
                "n_categories": shape.n_categories,
                "n_brands": shape.n_brands,
                "n_weeks": shape.n_weeks,
            }
            if shape
            else None
        ),
        "options": [
            {
                "id": o.id,
                "label": o.label,
                "spatial_axis": o.spatial_axis,
                "product_axis": o.product_axis,
                "expected_cells": o.expected_cells,
                "expected_rows_per_cell": o.expected_rows_per_cell,
                "available": o.available,
                "recommended": o.recommended,
                "reason": o.reason,
            }
            for o in options
        ],
    }
