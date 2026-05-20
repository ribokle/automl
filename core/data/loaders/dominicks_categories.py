"""Dominick's category code -> (panel label, elasticity-benchmark key).

The Dominick's archive ships one movement file per category, named after a
4-letter code (``wyog.csv`` = yogurt, ``wber.csv`` = beer). This map lets the
loader find files by friendly name and lets the validation benchmark join
on a category key without the user having to memorise Hoch (1995)'s
labelling.

The benchmark key is the lookup into ``core/benchmarks/data/elasticity.json``.
Some Dominick's categories don't have a Hoch-1995 entry; those fall back to
the closest Bijmolt-2005 bucket.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DominicksCategory:
    code: str  # 4-char Dominick's file prefix, e.g., "wyog"
    label: str  # friendly key the loader's --categories flag accepts
    display: str  # human-readable category name written into the panel
    benchmark_key: str  # key into core/benchmarks/data/elasticity.json


CATEGORIES: tuple[DominicksCategory, ...] = (
    DominicksCategory("wana", "analgesics",          "Analgesics",          "analgesics"),
    DominicksCategory("wbat", "bath_soap",           "Bath soap",           "bath_soap"),
    DominicksCategory("wbnt", "bathroom_tissue",     "Bathroom tissue",     "paper_products"),
    DominicksCategory("wber", "beer",                "Beer",                "beer"),
    DominicksCategory("wbjc", "bottled_juice",       "Bottled juice",       "bottled_juice"),
    DominicksCategory("wcer", "cereal",              "Cereal",              "cereal"),
    DominicksCategory("wche", "cheese",              "Cheese",              "cheese"),
    DominicksCategory("wcig", "cigarettes",          "Cigarettes",          "cigarettes"),
    DominicksCategory("wcoo", "cookies",             "Cookies",             "cookies"),
    DominicksCategory("wcra", "crackers",            "Crackers",            "crackers"),
    DominicksCategory("wdid", "dish_detergent",      "Dish detergent",      "detergent"),
    DominicksCategory("wfre", "frozen_entrees",      "Frozen entrees",      "frozen_entrees"),
    DominicksCategory("wfrd", "frozen_dinners",      "Frozen dinners",      "frozen_entrees"),
    DominicksCategory("wfrj", "frozen_juice",        "Frozen juice",        "frozen_juice"),
    DominicksCategory("wfsf", "fabric_softener",     "Fabric softener",     "laundry_soap"),
    DominicksCategory("wlnd", "laundry_detergent",   "Laundry detergent",   "detergent"),
    DominicksCategory("woat", "oatmeal",             "Oatmeal",             "cereal"),
    DominicksCategory("wptw", "paper_towels",        "Paper towels",        "paper_products"),
    DominicksCategory("wrfj", "refrigerated_juice",  "Refrigerated juice",  "refrigerated_juice"),
    DominicksCategory("wsdr", "soft_drinks",         "Soft drinks",         "soft_drinks"),
    DominicksCategory("wsna", "snack_crackers",      "Snack crackers",      "snack_crackers"),
    DominicksCategory("wsoa", "soaps",               "Soaps",               "laundry_soap"),
    DominicksCategory("wsou", "canned_soup",         "Canned soup",         "canned_soup"),
    DominicksCategory("wtbr", "toothbrushes",        "Toothbrushes",        "toothbrushes"),
    DominicksCategory("wtna", "tuna",                "Tuna",                "tuna"),
    DominicksCategory("wtpa", "toothpaste",          "Toothpaste",          "toothpaste"),
    DominicksCategory("wyog", "yogurt",              "Yogurt",              "yogurt"),
)


BY_LABEL: dict[str, DominicksCategory] = {c.label: c for c in CATEGORIES}
BY_CODE: dict[str, DominicksCategory] = {c.code: c for c in CATEGORIES}


def resolve(label_or_code: str) -> DominicksCategory:
    key = label_or_code.strip().lower()
    if key in BY_LABEL:
        return BY_LABEL[key]
    if key in BY_CODE:
        return BY_CODE[key]
    raise KeyError(
        f"Unknown Dominick's category: {label_or_code!r}. "
        f"Known labels: {sorted(BY_LABEL)}; codes: {sorted(BY_CODE)}."
    )
