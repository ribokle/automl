"""Synthetic CPG panel generator with embedded ground-truth elasticities.

Produces a weekly SKU x store panel with realistic price, promo, and seasonal
structure. The relationship between units and the drivers is governed by the
constants in `elasticity_spec.PPGS`, which downstream model agents must
recover within validation tolerances.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic.elasticity_spec import PPGS, PPGTruth, truth_dict

STORES = ["S01", "S02", "S03", "S04", "S05"]
REGION_OF = {"S01": "EAST", "S02": "EAST", "S03": "WEST", "S04": "WEST", "S05": "CENTRAL"}
PACK_SIZES = ["small", "medium", "large"]
SEGMENTS = ["value", "mainstream", "premium"]


@dataclass(frozen=True)
class SKUSpec:
    sku: str
    ppg_id: str
    category: str
    brand: str
    pack_size: str
    segment: str
    base_price: float
    base_units: float


def _build_sku_universe(rng: np.random.Generator, skus_per_ppg: int = 6) -> list[SKUSpec]:
    skus: list[SKUSpec] = []
    for ppg in PPGS:
        for i in range(skus_per_ppg):
            pack = PACK_SIZES[i % 3]
            seg = SEGMENTS[i % 3]
            price_factor = {"small": 0.85, "medium": 1.0, "large": 1.25}[pack]
            unit_factor = {"small": 1.2, "medium": 1.0, "large": 0.7}[pack]
            seg_factor = {"value": 0.9, "mainstream": 1.0, "premium": 1.15}[seg]
            skus.append(
                SKUSpec(
                    sku=f"{ppg.ppg_id}-{i + 1:02d}",
                    ppg_id=ppg.ppg_id,
                    category=ppg.category,
                    brand=ppg.brand,
                    pack_size=pack,
                    segment=seg,
                    base_price=round(ppg.base_price * price_factor * seg_factor, 2),
                    base_units=ppg.base_units * unit_factor * (0.85 + 0.3 * rng.random()),
                )
            )
    return skus


def _weeks(n_weeks: int, start: date = date(2023, 1, 2)) -> list[date]:
    """Monday-anchored weekly index."""
    # Force start to a Monday.
    start = start - timedelta(days=start.weekday())
    return [start + timedelta(weeks=w) for w in range(n_weeks)]


def _seasonality(week_idx: np.ndarray, amp: float) -> np.ndarray:
    return 1.0 + amp * np.sin(2 * np.pi * week_idx / 52.0)


def _holiday_calendar(weeks: list[date]) -> pd.Series:
    """Light-touch holiday markers; uses US weeks around Thanksgiving, Christmas, July 4, Memorial, Labor."""
    markers: dict[date, str] = {}
    for w in weeks:
        y = w.year
        # Thanksgiving (4th Thursday of November)
        nov1 = date(y, 11, 1)
        thanksgiving = nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21)
        if abs((w - thanksgiving).days) <= 6:
            markers[w] = "thanksgiving"
        elif w.month == 12 and 14 <= w.day <= 28:
            markers[w] = "christmas"
        elif w.month == 7 and w.day <= 7:
            markers[w] = "july4"
        elif w.month == 5 and w.day >= 25:
            markers[w] = "memorial"
        elif w.month == 9 and w.day <= 7:
            markers[w] = "labor"
    return pd.Series(markers)


def _draw_promo_pattern(rng: np.random.Generator, n_weeks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tpr_flag, display_flag, feature_flag) arrays."""
    tpr = (rng.random(n_weeks) < 0.18).astype(int)
    # display tends to co-occur with TPR ~60% of the time
    display = np.where(tpr == 1, (rng.random(n_weeks) < 0.6).astype(int), (rng.random(n_weeks) < 0.05).astype(int))
    feature = np.where(tpr == 1, (rng.random(n_weeks) < 0.4).astype(int), (rng.random(n_weeks) < 0.03).astype(int))
    return tpr, display, feature


def _competitor_price(base_price: float, n_weeks: int, rng: np.random.Generator) -> np.ndarray:
    drift = np.cumsum(rng.normal(0, 0.002, size=n_weeks))
    noise = rng.normal(0, 0.02, size=n_weeks)
    return base_price * (1.0 + drift + noise)


def _distribution_acv(n_weeks: int, rng: np.random.Generator) -> np.ndarray:
    start = 70 + 20 * rng.random()
    walk = np.cumsum(rng.normal(0, 0.5, size=n_weeks))
    series = np.clip(start + walk, 50, 99)
    return series


def generate_panel(
    seed: int = 42,
    n_weeks: int = 104,
    skus_per_ppg: int = 6,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    skus = _build_sku_universe(rng, skus_per_ppg=skus_per_ppg)
    weeks = _weeks(n_weeks)
    holidays = _holiday_calendar(weeks)
    ppg_lookup: dict[str, PPGTruth] = {p.ppg_id: p for p in PPGS}

    week_idx = np.arange(n_weeks, dtype=float)
    trend = 1.0 + 0.001 * week_idx  # mild upward trend

    rows: list[dict] = []
    for sku in skus:
        ppg = ppg_lookup[sku.ppg_id]
        season = _seasonality(week_idx, ppg.seasonality_amp)
        for store in STORES:
            tpr, display, feature = _draw_promo_pattern(rng, n_weeks)
            discount_depth = np.where(tpr == 1, 0.10 + 0.20 * rng.random(n_weeks), 0.0)
            price = sku.base_price * (1.0 - discount_depth)
            # off-promo random price jitter (e.g., shelf reprice)
            jitter = rng.normal(0, ppg.price_jitter_pct, size=n_weeks)
            price = np.where(tpr == 1, price, sku.base_price * (1.0 + jitter))
            price = np.clip(price, 0.05 * sku.base_price, None)

            comp_price = _competitor_price(sku.base_price, n_weeks, rng)
            acv = _distribution_acv(n_weeks, rng)
            store_factor = 0.7 + 0.3 * (hash((sku.sku, store)) % 100) / 100.0

            # Demand model: log-linear with shocks.
            log_price_ratio = np.log(price / sku.base_price)
            log_volume = (
                np.log(sku.base_units * store_factor)
                + ppg.own_elasticity * log_price_ratio
                + ppg.tpr_lift_log * tpr
                + np.log1p(ppg.display_lift_pct) * display
                + np.log1p(ppg.feature_lift_pct) * feature
                + np.log(season)
                + np.log(trend)
                + np.log(acv / 90.0)
                + rng.normal(0, 0.08, size=n_weeks)
            )
            units = np.maximum(0, np.round(np.exp(log_volume))).astype(int)

            for w, week_date in enumerate(weeks):
                rows.append(
                    {
                        "sku": sku.sku,
                        "week_start": week_date.isoformat(),
                        "store_id": store,
                        "region": REGION_OF[store],
                        "category": sku.category,
                        "brand": sku.brand,
                        "pack_size": sku.pack_size,
                        "segment": sku.segment,
                        "ppg_id": sku.ppg_id,
                        "units": int(units[w]),
                        "price": round(float(price[w]), 4),
                        "base_price": float(sku.base_price),
                        "tpr_flag": int(tpr[w]),
                        "display_flag": int(display[w]),
                        "feature_flag": int(feature[w]),
                        "distribution_acv": round(float(acv[w]), 2),
                        "competitor_price": round(float(comp_price[w]), 4),
                        "holiday": holidays.get(week_date, None),
                    }
                )

    df = pd.DataFrame(rows)
    truth = {
        "seed": seed,
        "n_weeks": n_weeks,
        "n_skus": len(skus),
        "n_stores": len(STORES),
        "ppgs": truth_dict(),
        "tolerances": {
            "own_elasticity_pct": 0.20,
            "wape_loglog": 0.15,
            "wape_lightgbm": 0.12,
            "ppgs_with_correct_sign": 8,
            "ppgs_within_tolerance": 7,
        },
    }
    return df, truth


def write_panel(out_csv: Path, out_truth: Path, seed: int = 42) -> None:
    df, truth = generate_panel(seed=seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    out_truth.write_text(json.dumps(truth, indent=2))


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    write_panel(repo / "data" / "synthetic.csv", repo / "synthetic" / "truth.json")
