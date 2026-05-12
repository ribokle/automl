"""Token + USD cost accounting from anthropic responses."""
from __future__ import annotations

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
