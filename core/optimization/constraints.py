"""Optimization constraints + per-PPG inputs.

Three business constraints define the feasible region:

- **Price ladder** — chosen price must equal ``base_price * m`` for some
  multiplier ``m`` in a discrete, finite set (typically nickel-spaced
  multipliers like 0.85, 0.90, 0.95, 1.00, ...). Stops the MILP from
  picking $4.97 when the merchandiser only stocks $4.99.
- **Margin floor** — chosen price minus cost must be at least
  ``margin_floor_pct * base_price``. ``cost = cog_pct * base_price``;
  ``cog_pct`` defaults to 0.55 (same placeholder the simulator uses).
- **Competitive gap** — chosen price must stay within ``comp_gap_pct``
  of the competitor's reference price. Skipped when competitor data is
  missing on the PPG.

The MILP additionally caps the price move at ``[1 - max_decrease,
1 + max_increase]`` multipliers regardless of the ladder; this is the
"don't shock the shelf" guardrail.
"""
from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_LADDER: tuple[float, ...] = (
    0.85, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.15,
)


@dataclass
class OptimizationConstraints:
    price_ladder: tuple[float, ...] = DEFAULT_LADDER
    promo_states: tuple[int, ...] = (0, 1)
    cog_pct: float = 0.55
    margin_floor_pct: float = 0.05
    comp_gap_pct: float = 0.15
    max_decrease: float = 0.20
    max_increase: float = 0.20
    objective: str = "revenue"  # "revenue" | "margin"

    def __post_init__(self) -> None:
        if self.objective not in ("revenue", "margin"):
            raise ValueError(f"objective must be 'revenue' or 'margin', got {self.objective!r}")
        if not self.price_ladder:
            raise ValueError("price_ladder must contain at least one multiplier")
        if not self.promo_states:
            raise ValueError("promo_states must contain at least one state")
        if self.cog_pct < 0 or self.cog_pct >= 1:
            raise ValueError("cog_pct must be in [0, 1)")
        if self.margin_floor_pct < 0:
            raise ValueError("margin_floor_pct must be non-negative")

    def to_dict(self) -> dict:
        return {
            "price_ladder": list(self.price_ladder),
            "promo_states": list(self.promo_states),
            "cog_pct": self.cog_pct,
            "margin_floor_pct": self.margin_floor_pct,
            "comp_gap_pct": self.comp_gap_pct,
            "max_decrease": self.max_decrease,
            "max_increase": self.max_increase,
            "objective": self.objective,
        }


@dataclass
class PPGOptInputs:
    """Per-PPG inputs the optimisers need.

    ``competitor_price`` is optional — when ``None`` the competitive-gap
    constraint is skipped for this PPG. The agent fills this from the
    feature frame.
    """

    ppg_id: str
    model_kind: str
    coefficients: dict[str, float]
    base_price: float
    context: dict[str, float] = field(default_factory=dict)
    competitor_price: float | None = None
