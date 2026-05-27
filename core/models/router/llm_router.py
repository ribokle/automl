"""LLM-driven router with deterministic fallback.

The async LLM call lives in the modelling agent; this class stays pure and
testable. ``build_prompt`` renders the system+user messages, ``select`` takes
the model's raw text (or ``None`` for dry-run), parses a strict-JSON candidate
list, validates every key against the available set, and falls back to the
``DeterministicRouter`` on dry-run, parse failure, or any unknown key.
"""
from __future__ import annotations

import json

from core.config import RouterSettings, get_settings
from core.models.library import registry
from core.models.library.diagnostics import DataProfile
from core.models.result import ProblemType
from core.models.router.base import LEGACY_TAIL
from core.models.router.rules import DeterministicRouter

_SYSTEM = (
    "You are a model-selection router for retail price/promo demand modelling. "
    "Given a problem type and a data-diagnostics profile, choose an ordered set "
    "of models (best first) from the provided catalog of available model keys. "
    "Respond with STRICT JSON only: {\"candidates\": [\"key1\", ...], "
    "\"rationale\": \"...\"}. Choose only from the provided keys."
)


class LLMRouter:
    def __init__(
        self,
        fallback: DeterministicRouter | None = None,
        settings: RouterSettings | None = None,
    ) -> None:
        self._s = settings or get_settings().router
        self._fallback = fallback or DeterministicRouter(self._s)

    @staticmethod
    def build_prompt(
        problem: ProblemType,
        profile: DataProfile,
        available: list[str],
    ) -> tuple[str, str]:
        user = json.dumps(
            {
                "problem_type": problem.value,
                "data_profile": profile.to_dict(),
                "available_models": sorted(available),
            },
            indent=2,
            default=str,
        )
        return _SYSTEM, user

    def _parse(self, text: str, valid: set[str]) -> list[str] | None:
        try:
            payload = json.loads(text)
            raw = payload["candidates"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        if not isinstance(raw, list):
            return None
        ordered = [k for k in raw if isinstance(k, str) and k in valid]
        return ordered or None

    def select(
        self,
        problem: ProblemType,
        profile: DataProfile,
        enabled: set[str] | None = None,
        *,
        llm_text: str | None = None,
    ) -> list[str]:
        avail = registry.available_keys(enabled=enabled)
        if not llm_text:
            return self._fallback.select(problem, profile, enabled)
        parsed = self._parse(llm_text, avail)
        if parsed is None:
            return self._fallback.select(problem, profile, enabled)
        for key in LEGACY_TAIL:
            if key in avail and key not in parsed:
                parsed.append(key)
        return parsed
