"""Anthropic client wrapper.

Phase 0 scope: a thin wrapper that supports a `dry_run` mode (no API call,
returns a canned response) so the rest of the system can be exercised without
spending tokens. Real tool-use loops land in Phase 1.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from core.llm.cost import estimate_usd


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    raw: dict[str, Any]


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        dry_run: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        env_dry = os.environ.get("LLM_DRY_RUN", "").lower() in ("1", "true", "yes")
        self.dry_run = env_dry if dry_run is None else dry_run
        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def message(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 1024,
        cache_system: bool = True,
    ) -> LLMResponse:
        """One-shot message; tool-use loop is added in Phase 1."""
        if self.dry_run or not self.api_key:
            return LLMResponse(
                text=f"[dry_run] {user[:80]}...",
                model=model,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                raw={"dry_run": True},
            )

        client = self._ensure_client()
        sys_blocks = [{"type": "text", "text": system}]
        if cache_system:
            sys_blocks[0]["cache_control"] = {"type": "ephemeral"}

        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=sys_blocks,
            messages=[{"role": "user", "content": user}],
        )
        usage = resp.usage
        tokens_in = (
            getattr(usage, "input_tokens", 0)
            + getattr(usage, "cache_creation_input_tokens", 0)
            + getattr(usage, "cache_read_input_tokens", 0)
        )
        tokens_out = getattr(usage, "output_tokens", 0)
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return LLMResponse(
            text=text,
            model=model,
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
            cost_usd=estimate_usd(model, int(tokens_in), int(tokens_out)),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
