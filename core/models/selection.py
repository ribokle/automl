"""Shared winner-selection helpers for elasticity candidates.

Single source of truth for "which candidate wins" and "is a fit good enough to
stop escalating", so the legacy modelling agent and the router's escalation
loop rank candidates identically. All thresholds are passed in by the caller
(resolved from config) — no magic numbers live here.
"""
from __future__ import annotations

import math

from core.models.base import ElasticityFit


def test_wape(fit: ElasticityFit) -> float:
    val = fit.diagnostics.get("test_wape")
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return float("inf")
    return float(val)


def fit_acceptable(
    fit: ElasticityFit,
    *,
    magnitude_ceiling: float,
    wape_floor: float,
) -> bool:
    """A candidate is good enough to stop escalating: right sign, in magnitude
    band, and hold-out WAPE at or under the escalation floor."""
    return (
        fit.sign_ok
        and abs(fit.own_elasticity) <= magnitude_ceiling
        and test_wape(fit) <= wape_floor
    )


def pick_winner(
    attempts: list[ElasticityFit],
    *,
    magnitude_ceiling: float,
) -> ElasticityFit:
    """Lowest hold-out WAPE among sign-correct AND in-magnitude fits, relaxing
    in order: (1) sign_ok AND |ε|<=ceiling, (2) sign_ok only, (3) anything."""
    sign_ok = [a for a in attempts if a.sign_ok]
    in_band = [a for a in sign_ok if abs(a.own_elasticity) <= magnitude_ceiling]
    pool = in_band or sign_ok or attempts
    return min(pool, key=test_wape)
