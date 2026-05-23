"""Published own-price elasticity benchmarks.

Sources (see ``core/benchmarks/data/elasticity.json`` for full citations):

- Hoch, Kim, Montgomery & Rossi (1995), *Determinants of Store-Level Price
  Elasticity* — 18 Dominick's chain-level category elasticities. The
  primary reference for any category that exists in Dominick's.
- Bijmolt, van Heerde & Pieters (2005) — meta-analysis of 1,851
  elasticities; provides the grand-mean fallback (-2.62) and a handful of
  Hoch-uncovered categories (yogurt, salty snacks, ice cream, coffee,
  paper products).

These numbers are static, public, and small enough to bake in. Validation
joins on PPG category (via ``ppg_mapping_table.json``) and reports whether
each PPG's recovered elasticity falls inside the published category band.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "data" / "elasticity.json"


@dataclass(frozen=True)
class CategoryBenchmark:
    category_key: str
    display_name: str
    mean: float
    lo: float
    hi: float
    source: str

    def contains(self, elasticity: float) -> bool:
        return self.lo <= float(elasticity) <= self.hi


@dataclass(frozen=True)
class ElasticityBenchmarkTable:
    grand_mean: float
    grand_mean_source: str
    categories: dict[str, CategoryBenchmark]
    aliases: dict[str, str]
    sources: tuple[dict, ...]

    def lookup(self, category: str | None) -> CategoryBenchmark | None:
        if not category:
            return None
        key = _normalise(category)
        if key in self.categories:
            return self.categories[key]
        if key in self.aliases:
            return self.categories.get(self.aliases[key])
        # last-chance substring match against keys + display names
        for cat_key, cat in self.categories.items():
            if key in cat_key or key in _normalise(cat.display_name):
                return cat
        return None


def _normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.strip().lower()).strip("_")


@lru_cache(maxsize=1)
def load_elasticity_benchmarks() -> ElasticityBenchmarkTable:
    blob = json.loads(_DATA_PATH.read_text())
    cats = {
        key: CategoryBenchmark(
            category_key=key,
            display_name=v["display_name"],
            mean=float(v["mean"]),
            lo=float(v["lo"]),
            hi=float(v["hi"]),
            source=v["source"],
        )
        for key, v in blob["categories"].items()
    }
    aliases = {_normalise(k): v for k, v in blob.get("aliases", {}).items()}
    return ElasticityBenchmarkTable(
        grand_mean=float(blob["meta"]["grand_mean"]),
        grand_mean_source=blob["meta"]["grand_mean_source"],
        categories=cats,
        aliases=aliases,
        sources=tuple(blob["meta"].get("sources", [])),
    )


def lookup_category(category: str | None) -> CategoryBenchmark | None:
    return load_elasticity_benchmarks().lookup(category)


def classify(elasticity: float, bench: CategoryBenchmark | None) -> str:
    """Return one of ``in_band``, ``out_band_low``, ``out_band_high``, ``no_benchmark``.

    ``out_band_low`` = less elastic than the published band (elasticity above hi).
    ``out_band_high`` = more elastic than the published band (elasticity below lo).
    A NaN elasticity (no usable folds) collapses to ``no_benchmark``.
    """
    if bench is None:
        return "no_benchmark"
    e = float(elasticity)
    if e != e:  # NaN
        return "no_benchmark"
    if e > bench.hi:
        return "out_band_low"
    if e < bench.lo:
        return "out_band_high"
    return "in_band"
