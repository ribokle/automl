"""Driver groupings for decomposition.

Maps engineered feature columns to business-level categories so the UI
table doesn't render fifteen rows per PPG when the analyst only wants
five buckets. The mapping is intentionally explicit (not regex-driven)
because feature names are stable and ambiguity matters: a misassigned
column would silently mis-attribute lift.
"""
from __future__ import annotations

# Engineered features -> business driver category. Anything not listed
# falls into "other" so the agent surfaces unfamiliar columns instead of
# silently dropping them.
FEATURE_TO_GROUP: dict[str, str] = {
    "log_price": "price",
    "log_base_price": "price",
    "log_price_gap": "price",
    "discount_depth": "price",
    "tpr_share": "promo",
    "display_share": "promo",
    "feature_share": "promo",
    "log_distribution_acv": "distribution",
    "log_competitor_price": "competitor",
    "lag1_log_price": "lags",
    "lag1_log_units": "lags",
    "lag4_log_price": "lags",
    "week_sin": "seasonality",
    "week_cos": "seasonality",
    "is_holiday_week": "seasonality",
    # semi-log alias
    "price": "price",
}


GROUP_ORDER: list[str] = [
    "price",
    "promo",
    "distribution",
    "competitor",
    "seasonality",
    "lags",
    "other",
]


def group_for(feature: str) -> str:
    return FEATURE_TO_GROUP.get(feature, "other")
