"""Deterministic rules router.

Pure function of (problem type, data profile): same inputs always yield the
same ordered candidate list. This is the dry-run / no-API-key fallback for the
LLM router. All thresholds come from ``RouterSettings`` — nothing hardcoded.
The raw preference list is filtered through the registry's availability set so
candidates whose dependency is missing are dropped silently, and the legacy
trio is always appended as a guaranteed-available tail.
"""
from __future__ import annotations

from core.config import RouterSettings, get_settings
from core.models.library import registry
from core.models.library.diagnostics import DataProfile
from core.models.result import ProblemType
from core.models.router.base import LEGACY_TAIL


class DeterministicRouter:
    def __init__(self, settings: RouterSettings | None = None) -> None:
        self._s = settings or get_settings().router

    def _preferences(self, problem: ProblemType, profile: DataProfile) -> list[str]:
        s = self._s
        if problem == ProblemType.OWN_ELASTICITY:
            if profile.is_panel and profile.n_entities >= s.panel_min_entities:
                return ["fixed_effects", "loglog_ols", "lightgbm"]
            if profile.n_obs < s.small_n_threshold:
                return ["ridge", "loglog_ols", "lasso", "huber"]
            return ["loglog_ols", "elasticnet", "double_ml", "iv_2sls", "lightgbm"]
        if problem == ProblemType.CROSS_PRICE:
            return ["crossprice_loglog", "aids", "nested_logit"]
        if problem == ProblemType.FORECAST:
            if profile.seasonality_detected:
                return ["sarimax", "holt_winters", "ets", "state_space", "prophet"]
            return ["arimax", "ets", "state_space", "prophet"]
        if problem == ProblemType.PROMO_UPLIFT:
            return ["baseline_uplift", "uplift_causal_tree", "lightgbm"]
        if problem == ProblemType.DEMAND_SYSTEM:
            return ["crossprice_loglog", "aids", "blp", "nested_logit"]
        if problem == ProblemType.PANEL:
            return ["fixed_effects", "random_effects"]
        return list(LEGACY_TAIL)

    def select(
        self,
        problem: ProblemType,
        profile: DataProfile,
        enabled: set[str] | None = None,
    ) -> list[str]:
        ordered: list[str] = []
        for key in (*self._preferences(problem, profile), *LEGACY_TAIL):
            if key not in ordered:
                ordered.append(key)
        avail = registry.available_keys(enabled=enabled)
        return [k for k in ordered if k in avail]
