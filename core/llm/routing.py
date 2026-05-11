"""Per-agent model routing."""
from __future__ import annotations

OPUS = "claude-opus-4-7"
SONNET = "claude-sonnet-4-6"

# Heavy reasoning agents -> opus; lighter narration / structured agents -> sonnet.
OPUS_AGENTS: frozenset[str] = frozenset(
    {"ppg_mapping", "ppg_selection", "modeling", "decomposition", "insights"}
)


def model_for(agent: str) -> str:
    return OPUS if agent in OPUS_AGENTS else SONNET
