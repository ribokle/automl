"""Feature engineering for the PPG × week analytics grain.

Builds the feature frame that downstream modelling consumes. All transforms
operate on the PPG-week aggregate from `core.features.eda.ppg_week_aggregate`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "log_units"

ENGINEERED_COLUMNS: list[str] = [
    "log_units",
    "log_price",
    "log_base_price",
    "discount_depth",
    "tpr_share",
    "display_share",
    "feature_share",
    "log_distribution_acv",
    "log_competitor_price",
    "log_price_gap",
    "lag1_log_price",
    "lag1_log_units",
    "lag4_log_price",
    "week_sin",
    "week_cos",
    "is_holiday_week",
]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Engineer per-PPG-week features. Drops the first 4 weeks per PPG to keep
    only rows where every lagged feature is observed."""
    df = panel.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df.sort_values(["ppg_id", "week_start"], inplace=True)

    df["log_units"] = np.log(df["units"].clip(lower=1.0))
    df["log_price"] = np.log(df["price"].clip(lower=0.01))
    df["log_base_price"] = np.log(df["base_price"].clip(lower=0.01))
    df["log_distribution_acv"] = np.log(df["distribution_acv"].clip(lower=0.01))
    df["log_competitor_price"] = np.log(df["competitor_price"].clip(lower=0.01))
    df["log_price_gap"] = df["log_price"] - df["log_competitor_price"]

    by_ppg = df.groupby("ppg_id", group_keys=False)
    df["lag1_log_price"] = by_ppg["log_price"].shift(1)
    df["lag1_log_units"] = by_ppg["log_units"].shift(1)
    df["lag4_log_price"] = by_ppg["log_price"].shift(4)

    week = df["week_start"].dt.isocalendar().week.astype(float)
    angle = 2.0 * np.pi * week / 52.0
    df["week_sin"] = np.sin(angle)
    df["week_cos"] = np.cos(angle)
    df["is_holiday_week"] = df["is_holiday_week"].astype(float)

    keep = ["ppg_id", "week_start"] + ENGINEERED_COLUMNS
    out = df[keep].dropna().reset_index(drop=True)
    return out
