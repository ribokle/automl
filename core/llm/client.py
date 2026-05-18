"""Anthropic client wrapper with pluggable provider.

Four providers are supported so the same agent can be exercised under
different auth/billing surfaces:

- ``dry_run``  no network call, deterministic ``[dry_run] ...`` response.
                Used by default in CI and when no credentials are available.
- ``api``      direct Anthropic API with ``ANTHROPIC_API_KEY``.
- ``oauth``    direct Anthropic API with an OAuth bearer token from
                ``ANTHROPIC_AUTH_TOKEN`` (e.g. a Pro/Max subscription token
                obtained via ``claude setup-token``).
- ``cli``      shells out to the ``claude`` binary (Claude Code CLI). Uses
                whatever auth the local CLI has configured (OAuth keychain,
                API key, etc.). Useful when the test environment already has
                Claude Code installed but no environment-variable creds.

Provider selection precedence:

1. Constructor ``provider=`` arg.
2. ``LLM_PROVIDER`` env var.
3. ``LLM_DRY_RUN=true`` -> ``dry_run``.
4. ``ANTHROPIC_API_KEY`` present -> ``api``.
5. ``ANTHROPIC_AUTH_TOKEN`` present -> ``oauth``.
6. Fallback -> ``dry_run``.

``cli`` is never auto-selected: shelling out to the local Claude binary is
a side-effect that must be explicit (set ``LLM_PROVIDER=cli`` or pass
``provider=LLMProvider.CLI``).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.llm.cost import estimate_usd


class LLMProvider(str, Enum):
    DRY_RUN = "dry_run"
    API = "api"
    OAUTH = "oauth"
    CLI = "cli"


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    raw: dict[str, Any]
    provider: str = LLMProvider.DRY_RUN.value


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "yes", "on")


def detect_provider() -> LLMProvider:
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        try:
            return LLMProvider(explicit)
        except ValueError as exc:
            valid = ", ".join(p.value for p in LLMProvider)
            raise ValueError(f"LLM_PROVIDER={explicit!r} invalid; expected one of {valid}") from exc
    if _truthy(os.environ.get("LLM_DRY_RUN")):
        return LLMProvider.DRY_RUN
    if os.environ.get("ANTHROPIC_API_KEY"):
        return LLMProvider.API
    if os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return LLMProvider.OAUTH
    return LLMProvider.DRY_RUN


class AnthropicClient:
    def __init__(
        self,
        *,
        provider: LLMProvider | str | None = None,
        api_key: str | None = None,
        auth_token: str | None = None,
        dry_run: bool | None = None,
        cli_path: str | None = None,
        cli_timeout_s: int = 180,
    ) -> None:
        if dry_run is True:
            provider = LLMProvider.DRY_RUN
        elif provider is None:
            provider = detect_provider()
        elif isinstance(provider, str):
            provider = LLMProvider(provider)
        self.provider: LLMProvider = provider
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.auth_token = auth_token or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        self.cli_path = cli_path or shutil.which("claude") or "claude"
        self.cli_timeout_s = cli_timeout_s
        self._client: Any = None

    @property
    def dry_run(self) -> bool:
        return self.provider == LLMProvider.DRY_RUN

    def _ensure_sdk_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            if self.provider == LLMProvider.OAUTH:
                if not self.auth_token:
                    raise RuntimeError("LLMProvider.OAUTH requires ANTHROPIC_AUTH_TOKEN")
                self._client = Anthropic(auth_token=self.auth_token, api_key=None)
            else:
                if not self.api_key:
                    raise RuntimeError("LLMProvider.API requires ANTHROPIC_API_KEY")
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
        if self.provider == LLMProvider.DRY_RUN:
            return LLMResponse(
                text=f"[dry_run] {user[:80]}...",
                model=model,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                raw={"dry_run": True},
                provider=self.provider.value,
            )
        if self.provider in (LLMProvider.API, LLMProvider.OAUTH):
            return self._sdk_message(model, system, user, max_tokens, cache_system)
        if self.provider == LLMProvider.CLI:
            return self._cli_message(model, system, user)
        raise RuntimeError(f"unsupported provider: {self.provider}")

    def _sdk_message(
        self, model: str, system: str, user: str, max_tokens: int, cache_system: bool
    ) -> LLMResponse:
        client = self._ensure_sdk_client()
        sys_blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
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
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return LLMResponse(
            text=text,
            model=model,
            tokens_in=int(tokens_in),
            tokens_out=int(tokens_out),
            cost_usd=estimate_usd(model, int(tokens_in), int(tokens_out)),
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            provider=self.provider.value,
        )

    def _cli_message(self, model: str, system: str, user: str) -> LLMResponse:
        """Invoke the local ``claude`` CLI as a model gateway.

        Uses ``-p`` for non-interactive output, ``--output-format json`` for a
        parseable single-result blob, and ``--tools ""`` to disable the CLI's
        built-in tools so the response is a pure model completion. The user
        prompt is piped via stdin to avoid argv-length limits.
        """
        cmd = [
            self.cli_path,
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--permission-mode",
            "bypassPermissions",
            "--model",
            model,
            "--system-prompt",
            system,
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=user,
                capture_output=True,
                text=True,
                timeout=self.cli_timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"claude CLI not found at {self.cli_path!r}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"claude CLI timed out after {self.cli_timeout_s}s") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited {proc.returncode}: {proc.stderr.strip()[:400]}"
            )
        try:
            data: dict[str, Any] = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"claude CLI returned non-JSON: {proc.stdout[:200]!r}"
            ) from exc
        text = data.get("result", "")
        usage = data.get("usage", {}) or {}
        tokens_in = int(
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        tokens_out = int(usage.get("output_tokens", 0))
        cost = float(data.get("total_cost_usd", estimate_usd(model, tokens_in, tokens_out)))
        return LLMResponse(
            text=text,
            model=data.get("model", model),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            raw=data,
            provider=self.provider.value,
        )
