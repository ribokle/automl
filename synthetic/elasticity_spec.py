"""Ground-truth elasticity / lift parameters per PPG.

These are the values the panel is generated from. Downstream model agents
must recover them within the tolerances defined in `synthetic/truth.json`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class PPGTruth:
    ppg_id: str
    category: str
    brand: str
    own_elasticity: float
    tpr_lift_log: float
    display_lift_pct: float
    feature_lift_pct: float
    seasonality_amp: float
    base_units: float
    base_price: float
    price_jitter_pct: float


PPGS: list[PPGTruth] = [
    PPGTruth("PPG01", "soda",      "AlphaCola",   -2.4, 0.55, 0.22, 0.15, 0.18, 1200, 1.99, 0.04),
    PPGTruth("PPG02", "soda",      "BetaPop",     -1.9, 0.40, 0.18, 0.12, 0.14,  800, 2.49, 0.05),
    PPGTruth("PPG03", "chips",     "CrispCo",     -2.8, 0.65, 0.25, 0.18, 0.10,  600, 3.49, 0.05),
    PPGTruth("PPG04", "chips",     "GoldKern",    -2.1, 0.45, 0.20, 0.14, 0.08,  450, 3.99, 0.06),
    PPGTruth("PPG05", "cereal",    "MorningOat",  -1.6, 0.35, 0.15, 0.10, 0.06,  300, 4.49, 0.04),
    PPGTruth("PPG06", "cereal",    "FlakeStar",   -1.4, 0.30, 0.12, 0.08, 0.05,  240, 4.99, 0.04),
    PPGTruth("PPG07", "icecream",  "FrostyPint",  -3.2, 0.75, 0.30, 0.20, 0.40,  180, 5.99, 0.07),
    PPGTruth("PPG08", "coffee",    "BrewMaster",  -1.2, 0.25, 0.10, 0.08, 0.05,  220, 7.99, 0.03),
]


def truth_dict() -> dict:
    return {p.ppg_id: asdict(p) for p in PPGS}
