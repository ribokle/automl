"""Escalation loop: fit candidates in order, stop when one fits well enough.

Replaces the hardcoded loglog -> sign-retry semilog -> always-lightgbm sequence
with a config-bounded walk over the router's candidate list. Non-scalar
results (pure forecasters, demand systems without a usable diagonal) are
recorded but never enter winner ranking. ``selection.pick_winner`` chooses the
final winner among the candidates that actually produced a scalar elasticity,
so the relaxation semantics match the legacy agent exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from core.models.base import ElasticityFit
from core.models.library import registry
from core.models.library.base import FitContext
from core.models.result import to_elasticity_fit
from core.models.selection import fit_acceptable, pick_winner


@dataclass
class EscalationResult:
    candidates: list[str]
    attempts: list[ElasticityFit] = field(default_factory=list)
    winner: ElasticityFit | None = None
    non_scalar: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": list(self.candidates),
            "attempts": [a.to_dict() for a in self.attempts],
            "winner_model": self.winner.model if self.winner else None,
            "winner": self.winner.to_dict() if self.winner else None,
            "non_scalar": list(self.non_scalar),
            "errors": dict(self.errors),
        }


def run_escalation(
    candidates: list[str],
    frame: pd.DataFrame,
    ctx: FitContext,
    *,
    max_candidates: int,
    magnitude_ceiling: float,
    wape_floor: float,
    hparams: dict[str, dict[str, Any]] | None = None,
) -> EscalationResult:
    res = EscalationResult(candidates=list(candidates))
    for key in candidates[:max_candidates]:
        if not registry.has(key):
            continue
        plugin = registry.get(key)
        if not plugin.is_available():
            continue
        cand_ctx = replace(ctx, hparams=(hparams or {}).get(key, {}))
        try:
            model_result = plugin.fit(frame, cand_ctx)
        except Exception as exc:  # noqa: BLE001 — record + continue to next candidate
            res.errors[key] = f"{type(exc).__name__}: {exc}"
            continue
        fit = to_elasticity_fit(model_result)
        if fit is None:
            res.non_scalar.append(key)
            continue
        res.attempts.append(fit)
        if fit_acceptable(fit, magnitude_ceiling=magnitude_ceiling, wape_floor=wape_floor):
            break
    if res.attempts:
        res.winner = pick_winner(res.attempts, magnitude_ceiling=magnitude_ceiling)
    return res
