"""Token + USD cost accounting from anthropic responses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.orchestrator.state import RunState

# Prices per 1M tokens — placeholders, refined at integration time.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
}


def estimate_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    if model not in PRICES:
        return 0.0
    in_price, out_price = PRICES[model]
    return (tokens_in / 1_000_000) * in_price + (tokens_out / 1_000_000) * out_price


@dataclass
class AgentCost:
    agent: str
    status: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    duration_s: float | None
    duration_str: str
    provider: str = ""

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "status": self.status,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "duration_s": self.duration_s,
            "duration_str": self.duration_str,
            "provider": self.provider,
        }


@dataclass
class CostTotals:
    tokens_in: int
    tokens_out: int
    cost_usd: float
    duration_s: float
    duration_str: str
    provider: str = ""

    def to_dict(self) -> dict:
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "duration_s": self.duration_s,
            "duration_str": self.duration_str,
            "provider": self.provider,
        }


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "—"
    if seconds < 1:
        return f"{int(seconds * 1000)} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(int(seconds), 60)
    return f"{minutes}m {sec}s"


def summarise_run(run: "RunState") -> tuple[list[AgentCost], CostTotals]:
    """Roll up per-agent tokens, cost, and duration from a `RunState`.

    The base ``Agent`` class already populates ``tokens_in / tokens_out /
    cost_usd`` on each ``AgentResult`` via ``call_llm``. This walks the
    run-state map, formats durations, and produces a typed rollup for the
    insights agent + the cost dashboard UI.
    """
    per_agent: list[AgentCost] = []
    tot_in = tot_out = 0
    tot_cost = 0.0
    tot_seconds = 0.0
    providers_seen: set[str] = set()

    for name, ar in run.agents.items():
        duration: float | None = None
        if ar.started_at and ar.finished_at:
            duration = (ar.finished_at - ar.started_at).total_seconds()
            tot_seconds += duration
        status = ar.status.value if hasattr(ar.status, "value") else str(ar.status)
        per_agent.append(
            AgentCost(
                agent=name,
                status=status,
                tokens_in=ar.tokens_in,
                tokens_out=ar.tokens_out,
                cost_usd=ar.cost_usd,
                duration_s=duration,
                duration_str=_format_duration(duration),
                provider=getattr(ar, "provider", "") or "",
            )
        )
        tot_in += ar.tokens_in
        tot_out += ar.tokens_out
        tot_cost += ar.cost_usd
        prov = getattr(ar, "provider", "") or ""
        if prov:
            providers_seen.add(prov)

    # If every recorded provider matched, surface it on the totals so the
    # report header can show "Dry-run mode" without inspecting every row.
    totals_provider = next(iter(providers_seen)) if len(providers_seen) == 1 else (
        "mixed" if len(providers_seen) > 1 else ""
    )
    totals = CostTotals(
        tokens_in=tot_in,
        tokens_out=tot_out,
        cost_usd=tot_cost,
        duration_s=tot_seconds,
        duration_str=_format_duration(tot_seconds) if tot_seconds > 0 else "—",
        provider=totals_provider,
    )
    return per_agent, totals
