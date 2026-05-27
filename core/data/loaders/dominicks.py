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
own machine. The loader is a pure transform: no network, no LLM.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
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

_COLUMN_ORDER = [
    "sku", "week_start", "store_id", "region",
    "units", "price", "base_price",
    "tpr_flag", "display_flag", "feature_flag",
    "distribution_acv",
    "category", "brand", "pack_size", "segment",
    "competitor_price", "holiday",
]


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
    """Find ``{prefix}{code}.csv`` (or .txt) anywhere under raw_dir (case-insensitive).

    Some Kilts-distributed UPC dictionaries ship with a .txt extension despite
    being comma-separated; both extensions are accepted.
    """
    stem = f"{prefix}{code}".lower()
    for candidate in raw_dir.rglob("*"):
        if candidate.suffix.lower() in {".csv", ".txt"} and candidate.stem.lower() == stem:
            return candidate
    raise FileNotFoundError(
        f"Could not find {stem}.csv under {raw_dir}. Place the Dominick's "
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


def _compute_base_price(df: pd.DataFrame, window: int, conn: duckdb.DuckDBPyConnection) -> pd.Series:
    """Trailing-max of non-promo price per (sku, store_id) via DuckDB window function.

    On any week whose ``tpr_flag`` is 1, the price observation is excluded from
    the rolling max; the most-recent non-promo price wins. Falls back to the
    current ``price`` when no non-promo week exists yet in the trailing window.
    """
    df_sorted = df.sort_values(["sku", "store_id", "week_start"])
    conn.register("_bp", df_sorted[["sku", "store_id", "week_start", "price", "tpr_flag"]])
    arr = conn.execute(f"""
        SELECT COALESCE(
            MAX(CASE WHEN tpr_flag = 0 THEN price END)
                OVER (
                    PARTITION BY sku, store_id
                    ORDER BY week_start
                    ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
                ),
            price
        ) AS base_price
        FROM _bp
        ORDER BY sku, store_id, week_start
    """).fetchnumpy()["base_price"]
    conn.unregister("_bp")
    return pd.Series(arr, index=df_sorted.index)


def _attach_holidays(week_starts: pd.Series) -> pd.Series:
    """Return a Series mapping each week_start to the nearest US holiday name (or NaN).

    Builds a lookup dict over unique dates (~500 across the 9-year Dominick's
    span) then broadcasts via Series.map — O(unique_weeks * 7) instead of
    O(rows * 7).
    """
    import holidays

    dates = pd.to_datetime(week_starts).dt.date
    unique = dates.unique()
    if len(unique) == 0:
        return pd.Series([], dtype="object", index=week_starts.index)
    years = range(min(d.year for d in unique), max(d.year for d in unique) + 1)
    us = holidays.UnitedStates(years=list(years))
    lookup: dict = {}
    for w in unique:
        for offset in range(7):
            name = us.get(w + timedelta(days=offset))
            if name:
                lookup[w] = name
                break
    return dates.map(lookup)


def _build_one_category(
    cat: DominicksCategory, opts: LoaderOptions, conn: duckdb.DuckDBPyConnection
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

    # Vectorised string formatting — avoids per-row Python lambda overhead.
    upc_arr = move["UPC"].to_numpy(dtype=np.int64)
    sto_arr = move["STORE"].to_numpy(dtype=np.int64)
    sku_col  = np.char.add("upc_",   np.char.zfill(upc_arr.astype("U12"), 11))
    stor_col = np.char.add("store_", np.char.zfill(sto_arr.astype("U4"),   3))

    panel = pd.DataFrame(
        {
            "sku": sku_col,
            "store_id": stor_col,
            "week_start": pd.to_datetime(
                move["WEEK"].astype(int).map(_week_to_date)
            ),
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

    try:
        upc_lookup = upc.set_index("UPC")
    except KeyError:
        # UPC file is present but unparseable (e.g. upcrfj.txt ships as
        # fixed-width with no header). Fall back to empty enrichment so the
        # movement data still makes it into the panel.
        log.warning(
            "dominicks: %s UPC file lacks a UPC column — brand/size/segment "
            "will default to unknown/single/'' for this category",
            cat.label,
        )
        upc_lookup = pd.DataFrame(index=pd.Index([], name="UPC"))
    brand_by_upc = upc_lookup.get("DESCRIP", pd.Series(dtype="object")).map(_infer_brand)
    size_by_upc = upc_lookup.get("SIZE", pd.Series(dtype="object")).map(_pack_size)
    segment_by_upc = upc_lookup.get("COM_CODE", pd.Series(dtype="object")).astype("string")

    raw_upc = move["UPC"].astype(np.int64)
    panel["brand"] = raw_upc.map(brand_by_upc).fillna("unknown")
    panel["pack_size"] = raw_upc.map(size_by_upc).fillna("single")
    panel["segment"] = raw_upc.map(segment_by_upc).fillna("")

    panel["base_price"] = _compute_base_price(panel, opts.base_price_window, conn)
    # Promo weeks can otherwise drag base_price below the live price; the
    # canonical mart computes ``discount_depth = 1 - price/base_price`` and
    # expects base_price >= price.
    panel["base_price"] = panel[["base_price", "price"]].max(axis=1)

    holiday_series = _attach_holidays(panel["week_start"])
    panel["holiday"] = holiday_series.astype(object).where(holiday_series.notna(), None)

    return panel[_COLUMN_ORDER]


def coverage_report(panel: pd.DataFrame) -> dict[str, list[str]]:
    """List columns the loader emitted with zero variance.

    Dominick's movement files don't ship display / feature / ACV — the
    loader hardcodes them so the canonical schema stays valid, but those
    columns then carry no information for the modelling stage. Surfacing
    them here lets the CLI warn and the EDA dashboard render a
    "constant-by-design" badge instead of silently dropping them in
    feature_refine.
    """
    constant: list[str] = []
    all_null: list[str] = []
    for col in panel.columns:
        s = panel[col]
        if s.isna().all():
            all_null.append(col)
        elif s.dropna().nunique() <= 1:
            constant.append(col)
    return {"constant_columns": sorted(constant), "all_null_columns": sorted(all_null)}


def build_dominicks_panel(
    raw_dir: Path,
    categories: list[str] | None = None,
    stores: list[int] | None = None,
    start_week: int = 1,
    end_week: int | None = None,
    base_price_window: int = 13,
    out_dir: Path | None = None,
) -> pd.DataFrame | dict:
    """Build the Dominick's panel.

    When ``out_dir`` is None (default): collects all category DataFrames,
    concatenates, and returns a single DataFrame — existing behaviour used
    by tests and small-category runs.

    When ``out_dir`` is a Path: streams one category at a time, writing each
    as a Snappy-compressed Parquet file under ``out_dir/{category_label}.parquet``
    via DuckDB.  Returns a summary dict instead of a DataFrame so the full
    99M-row panel is never held in memory simultaneously.
    """
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

    conn = duckdb.connect()

    if out_dir is not None:
        out_dir = Path(out_dir)
        if out_dir.exists() and not out_dir.is_dir():
            raise ValueError(
                f"{out_dir} exists as a file. Delete it and re-run — "
                "prepare-dominicks now writes a directory of Parquet files, not a single CSV."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        row_count = 0
        cat_labels: list[str] = []
        last_piece: pd.DataFrame | None = None

        for cat in cats:
            try:
                piece = _build_one_category(cat, opts, conn)
            except Exception as exc:
                log.warning("dominicks: skipping %s — %s", cat.label, exc)
                continue
            if piece.empty:
                continue
            parquet_path = out_dir / f"{cat.label}.parquet"
            conn.register("_p", piece)
            conn.execute(
                f"COPY _p TO '{parquet_path.as_posix()}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
            )
            conn.unregister("_p")
            cat_labels.append(piece["category"].iat[0])
            row_count += len(piece)
            last_piece = piece

        conn.close()

        if last_piece is None:
            raise RuntimeError(
                "Dominick's loader produced no rows. Check that the raw archive "
                f"is present under {raw_dir} and that the requested categories "
                "have movement files."
            )
        return {"rows": row_count, "categories": cat_labels, "_last_piece": last_piece}

    # Default path: collect all pieces and return a single DataFrame.
    pieces: list[pd.DataFrame] = []
    for cat in cats:
        try:
            piece = _build_one_category(cat, opts, conn)
        except Exception as exc:
            log.warning("dominicks: skipping %s — %s", cat.label, exc)
            continue
        if not piece.empty:
            pieces.append(piece)

    conn.close()

    if not pieces:
        raise RuntimeError(
            "Dominick's loader produced no rows. Check that the raw archive "
            f"is present under {raw_dir} and that the requested categories "
            "have movement files."
        )

    return pd.concat(pieces, ignore_index=True)
