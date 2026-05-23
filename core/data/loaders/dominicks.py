"""Adapter: Dominick's Finer Foods scanner panel -> canonical panel schema.

The Dominick's archive (Kilts Center, University of Chicago Booth) publishes
weekly store x UPC movement for ~9 years across 29 categories. Each category
ships as a pair of CSVs:

    w<code>.csv     STORE,UPC,WEEK,MOVE,QTY,PRICE,SALE,PROFIT,OK   (long panel)
    upc<code>.csv   COM_UPC,UPC,DESCRIP,SIZE,CASE,NITEM           (UPC dictionary)

This loader reads one or more categories, applies the Hoch-1995 / Dominick's
calendar conventions (week 1 starts Thursday 1989-09-14), computes a
non-promo trailing-max ``base_price``, and writes a DataFrame in the panel
schema defined by ``core/data/schema.py``.

The Kilts data-use agreement forbids redistribution, so the raw files are
expected to live under ``data/dominicks-raw/`` (gitignored) on the operator's
own machine. The loader is a pure transform: no network, no LLM, no DuckDB.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.data.loaders.dominicks_categories import (
    CATEGORIES,
    DominicksCategory,
    resolve,
)

log = logging.getLogger(__name__)

# Dominick's week 1 starts Thursday 1989-09-14.
WEEK_ANCHOR: date = date(1989, 9, 14)

# Heuristic regex to pull a brand token from a Dominick's DESCRIP string.
# DESCRIP fields are space-separated all-caps with the brand as the first
# token or two, then a description, then size info.  We take the first
# alphabetic token of >=2 chars. This is good enough for clustering and the
# UI; the modeller doesn't depend on it.
_BRAND_TOKEN_RE = re.compile(r"^([A-Z][A-Z&\-']{1,})")


@dataclass(frozen=True)
class LoaderOptions:
    raw_dir: Path
    categories: tuple[DominicksCategory, ...]
    stores: tuple[int, ...] | None = None
    start_week: int = 1
    end_week: int | None = None
    base_price_window: int = 13


def _week_to_date(week: int) -> date:
    return WEEK_ANCHOR + timedelta(days=7 * (int(week) - 1))


def _find_file(raw_dir: Path, prefix: str, code: str) -> Path:
    """Find ``{prefix}{code}.csv`` anywhere under raw_dir (case-insensitive)."""
    target = f"{prefix}{code}.csv".lower()
    for candidate in raw_dir.rglob("*.csv"):
        if candidate.name.lower() == target:
            return candidate
    raise FileNotFoundError(
        f"Could not find {target} under {raw_dir}. Place the Dominick's "
        f"archive contents under {raw_dir}/ (any nesting is fine)."
    )


def _read_movement(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=lambda c: c.upper() in {"STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "SALE", "PROFIT", "OK"},
        dtype={"SALE": "string"},
        encoding="latin-1",
    )
    df.columns = [c.upper() for c in df.columns]
    return df


def _read_upc_dict(path: Path) -> pd.DataFrame:
    # Dominick's UPC dictionaries are ISO-8859 (SAS-export era); latin-1
    # decodes every byte without raising.
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.upper() for c in df.columns]
    keep = [c for c in ("UPC", "DESCRIP", "SIZE", "NITEM", "COM_CODE") if c in df.columns]
    return df[keep].copy()


def _infer_brand(descrip: str | float) -> str:
    if not isinstance(descrip, str):
        return "unknown"
    m = _BRAND_TOKEN_RE.match(descrip.strip().upper())
    if not m:
        return "unknown"
    return m.group(1).lower()


def _pack_size(size: str | float) -> str:
    if isinstance(size, str) and size.strip():
        return size.strip().lower()
    return "single"


def _compute_base_price(df: pd.DataFrame, window: int) -> pd.Series:
    """Trailing-max of non-promo price per (sku, store_id).

    On any week whose ``tpr_flag`` is 1, the price observation is masked
    out before the rolling max; the most-recent non-promo price wins.
    Falls back to the current ``price`` when no non-promo week has been
    seen yet (first weeks of a new SKU x store pair).
    """
    df = df.sort_values(["sku", "store_id", "week_start"])
    masked = df["price"].where(df["tpr_flag"] == 0)
    rolled = (
        masked.groupby([df["sku"], df["store_id"]])
        .transform(lambda s: s.rolling(window=window, min_periods=1).max())
    )
    return rolled.fillna(df["price"])


def _attach_holidays(week_starts: Iterable[date]) -> pd.Series:
    import holidays

    weeks = list(week_starts)
    if not weeks:
        return pd.Series([], dtype="object")
    years = range(min(w.year for w in weeks), max(w.year for w in weeks) + 1)
    us = holidays.UnitedStates(years=list(years))
    out: list[str | None] = []
    for w in weeks:
        hit: str | None = None
        for offset in range(7):
            d = w + timedelta(days=offset)
            name = us.get(d)
            if name:
                hit = name
                break
        out.append(hit)
    return pd.Series(out, dtype="object")


def _build_one_category(
    cat: DominicksCategory, opts: LoaderOptions
) -> pd.DataFrame:
    move_path = _find_file(opts.raw_dir, "w", cat.code[1:])
    upc_path = _find_file(opts.raw_dir, "upc", cat.code[1:])
    log.info("dominicks: reading %s (%s) <- %s", cat.label, cat.code, move_path)

    move = _read_movement(move_path)
    upc = _read_upc_dict(upc_path)

    move = move.dropna(subset=["STORE", "UPC", "WEEK", "MOVE", "PRICE"])
    move = move[(move["OK"] == 1) & (move["MOVE"] > 0) & (move["PRICE"] > 0)]
    if "QTY" in move.columns:
        move["QTY"] = move["QTY"].fillna(1).clip(lower=1)
    else:
        move["QTY"] = 1

    if opts.stores:
        move = move[move["STORE"].astype(int).isin(opts.stores)]
    move = move[move["WEEK"].astype(int) >= int(opts.start_week)]
    if opts.end_week is not None:
        move = move[move["WEEK"].astype(int) <= int(opts.end_week)]

    if move.empty:
        return move.assign()  # empty frame; downstream concat handles it

    panel = pd.DataFrame(
        {
            "sku": move["UPC"].astype(np.int64).map(lambda u: f"upc_{u:011d}"),
            "store_id": move["STORE"].astype(int).map(lambda s: f"store_{s:03d}"),
            "week_start": move["WEEK"].astype(int).map(_week_to_date),
            "region": "Chicago",
            "units": move["MOVE"].astype(float).round().clip(lower=0).astype(int),
            "price": (move["PRICE"].astype(float) / move["QTY"].astype(float)).round(4),
            "tpr_flag": move["SALE"].fillna("").str.upper().isin({"B", "S", "C"}).astype(int),
            "display_flag": 0,
            "feature_flag": 0,
            "distribution_acv": 100.0,
            "competitor_price": np.nan,
            "category": cat.display,
        }
    )

    upc_lookup = upc.set_index("UPC")
    brand_by_upc = upc_lookup.get("DESCRIP", pd.Series(dtype="object")).map(_infer_brand)
    size_by_upc = upc_lookup.get("SIZE", pd.Series(dtype="object")).map(_pack_size)
    segment_by_upc = upc_lookup.get("COM_CODE", pd.Series(dtype="object")).astype("string")

    raw_upc = move["UPC"].astype(np.int64)
    panel["brand"] = raw_upc.map(brand_by_upc).fillna("unknown")
    panel["pack_size"] = raw_upc.map(size_by_upc).fillna("single")
    panel["segment"] = raw_upc.map(segment_by_upc).fillna("")

    panel["base_price"] = _compute_base_price(panel, opts.base_price_window)
    # Promo weeks can otherwise drag base_price below the live price; the
    # canonical mart computes ``discount_depth = 1 - price/base_price`` and
    # expects base_price >= price.
    panel["base_price"] = panel[["base_price", "price"]].max(axis=1)
    holiday_series = _attach_holidays(panel["week_start"])
    holiday_series.index = panel.index
    panel["holiday"] = holiday_series.astype(object).where(holiday_series.notna(), None)
    return panel


def build_dominicks_panel(
    raw_dir: Path,
    categories: list[str] | None = None,
    stores: list[int] | None = None,
    start_week: int = 1,
    end_week: int | None = None,
    base_price_window: int = 13,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Dominick's raw dir not found: {raw_dir}")

    if categories:
        cats = tuple(resolve(c) for c in categories)
    else:
        cats = CATEGORIES

    opts = LoaderOptions(
        raw_dir=raw_dir,
        categories=cats,
        stores=tuple(stores) if stores else None,
        start_week=start_week,
        end_week=end_week,
        base_price_window=base_price_window,
    )

    pieces: list[pd.DataFrame] = []
    for cat in cats:
        try:
            piece = _build_one_category(cat, opts)
        except FileNotFoundError as exc:
            log.warning("dominicks: skipping %s — %s", cat.label, exc)
            continue
        if not piece.empty:
            pieces.append(piece)

    if not pieces:
        raise RuntimeError(
            "Dominick's loader produced no rows. Check that the raw archive "
            f"is present under {raw_dir} and that the requested categories "
            "have movement files."
        )

    panel = pd.concat(pieces, ignore_index=True)
    panel = panel.sort_values(["sku", "store_id", "week_start"]).reset_index(drop=True)

    column_order = [
        "sku", "week_start", "store_id", "region",
        "units", "price", "base_price",
        "tpr_flag", "display_flag", "feature_flag",
        "distribution_acv",
        "category", "brand", "pack_size", "segment",
        "competitor_price", "holiday",
    ]
    return panel[column_order]
