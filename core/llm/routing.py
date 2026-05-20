"""Per-agent model routing.

Defaults: heavy reasoning agents use Opus, everything else uses Sonnet. Both
model names and the heavy-agent set are configurable through ``core.config``.

Override precedence (highest first):

1. ``MODEL_<AGENT>`` env var (e.g. ``MODEL_PPG_MAPPING=claude-sonnet-4-6``).
2. ``ANTHROPIC_MODEL_OPUS`` / ``ANTHROPIC_MODEL_SONNET`` env vars for the
   role-level defaults.
3. The hard-coded defaults in ``core.config.Settings``.
"""
from __future__ import annotations

from core.config import get_settings

OPUS_AGENTS: frozenset[str] = frozenset(
    {"ppg_mapping", "ppg_selection", "modeling", "decomposition", "optimization", "insights"}
)


def model_for(agent: str) -> str:
    s = get_settings()
    override = s.model_overrides.get(agent)
    if override:
        return override
    return s.anthropic_model_opus if agent in OPUS_AGENTS else s.anthropic_model_sonnet
