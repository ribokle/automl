"""Due-to decomposition for elasticity models.

Splits observed units into ``base`` + per-driver contributions + residual.

The math is the same multiplicative log-space form for both log-log and
semi-log OLS, because both share the structure
``log_units = α + Σ βᵢ·xᵢ``. The only difference is the reference value
for each feature (continuous features baseline to their mean within the
PPG; promo/holiday dummies baseline to zero; ``log_price`` baselines to
``log_base_price`` so the price contribution reflects deviation from the
regular-price baseline).

Per-row decomposition:

    pred_log     = α + Σ βᵢ·xᵢ
    base_log     = α + Σ βᵢ·xᵢ_ref
    Δlog_i       = βᵢ·(xᵢ - xᵢ_ref)
    pred_units   = exp(pred_log)
    base_units   = exp(base_log)
    lift_units   = pred_units - base_units   (positive when drivers raise demand)

Each driver's unit-space contribution is allocated as
``lift_units · Δlog_i / Σ Δlog_i`` so the per-driver due-tos sum to
``lift_units`` exactly. When ``Σ Δlog ≈ 0`` (drivers cancel out) every
driver gets zero and any rounding is folded into the residual.

Residual = observed - predicted; it stays out of the driver split so the
agent can surface it as model-fit error rather than spurious "due to X".
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DUMMY_FEATURES: frozenset[str] = frozenset(
    {"tpr_share", "display_share", "feature_share", "is_holiday_week"}
)


def reference_values(
    frame: pd.DataFrame, features: Iterable[str], *, base_price_col: str = "log_base_price"
) -> dict[str, float]:
    """Reference (baseline) value per feature for the decomposition.

    Promo + holiday dummies baseline to 0 (off). ``log_price`` baselines to
    the per-row ``log_base_price`` when present, else to the PPG mean.
    Everything else baselines to its mean over the frame.
    """
    refs: dict[str, float] = {}
    for col in features:
        if col == "log_price" and base_price_col in frame.columns:
            refs[col] = float(frame[base_price_col].mean())
        elif col in DEFAULT_DUMMY_FEATURES:
            refs[col] = 0.0
        else:
            refs[col] = float(frame[col].mean()) if col in frame.columns else 0.0
    return refs


def decompose_ols_frame(
    frame: pd.DataFrame,
    coefficients: dict[str, float],
    *,
    observed_col: str = "log_units",
    base_price_col: str = "log_base_price",
) -> pd.DataFrame:
    """Per-row decomposition for an OLS fit.

    ``coefficients`` must include ``const`` plus a coefficient for every
    driver column present in ``frame``. Returns a DataFrame with one row
    per input row containing ``predicted``, ``base``, ``residual``,
    ``observed``, plus one column per driver (``due_<feature>``).
    """
    if "const" not in coefficients:
        raise ValueError("coefficients missing 'const' intercept")
    features = [c for c in coefficients if c != "const"]
    refs = reference_values(frame, features, base_price_col=base_price_col)

    work = frame.copy().reset_index(drop=True)
    const = float(coefficients["const"])

    pred_log = np.full(len(work), const, dtype=float)
    base_log = np.full(len(work), const, dtype=float)
    delta_log_by_feature: dict[str, np.ndarray] = {}
    for col in features:
        coef = float(coefficients[col])
        observed_x = work[col].astype(float).to_numpy() if col in work.columns else np.zeros(len(work))
        ref_x = refs[col]
        pred_log += coef * observed_x
        base_log += coef * ref_x
        delta_log_by_feature[col] = coef * (observed_x - ref_x)

    pred_units = np.exp(pred_log)
    base_units = np.exp(base_log)
    lift_units = pred_units - base_units

    total_delta = np.zeros(len(work), dtype=float)
    for col in features:
        total_delta = total_delta + delta_log_by_feature[col]

    out = pd.DataFrame(
        {
            "predicted": pred_units,
            "base": base_units,
            "lift": lift_units,
        }
    )
    if observed_col in work.columns:
        observed = np.exp(work[observed_col].astype(float).to_numpy())
        out["observed"] = observed
        out["residual"] = observed - pred_units

    safe_total = np.where(np.abs(total_delta) < 1e-9, np.nan, total_delta)
    for col in features:
        share = delta_log_by_feature[col] / safe_total
        share = np.where(np.isnan(share), 0.0, share)
        out[f"due_{col}"] = lift_units * share
    return out


def aggregate_to_groups(
    weekly: pd.DataFrame, features: list[str], grouping: dict[str, str]
) -> pd.DataFrame:
    """Roll up ``due_<feature>`` columns into ``due_group_<category>``.

    Sums driver contributions across features that share a category.
    """
    work = weekly.copy()
    groups: dict[str, list[str]] = {}
    for f in features:
        grp = grouping.get(f, "other")
        groups.setdefault(grp, []).append(f"due_{f}")
    for grp, cols in groups.items():
        existing = [c for c in cols if c in work.columns]
        work[f"due_group_{grp}"] = work[existing].sum(axis=1) if existing else 0.0
    return work


def summarise_ppg(weekly: pd.DataFrame, features: list[str], grouping: dict[str, str]) -> dict:
    """Aggregate one PPG's decomposition into a flat table-friendly row.

    Returns totals + per-feature + per-group dollar (unit) contributions
    and shares, plus a reconciliation diagnostic.
    """
    total_predicted = float(weekly["predicted"].sum())
    total_base = float(weekly["base"].sum())
    total_observed = float(weekly.get("observed", weekly["predicted"]).sum())
    total_lift = total_predicted - total_base

    per_feature_units = {f: float(weekly[f"due_{f}"].sum()) for f in features if f"due_{f}" in weekly.columns}
    per_feature_share = {
        f: (u / total_lift) if abs(total_lift) > 1e-9 else 0.0
        for f, u in per_feature_units.items()
    }

    per_group_units: dict[str, float] = {}
    for f, u in per_feature_units.items():
        per_group_units[grouping.get(f, "other")] = per_group_units.get(grouping.get(f, "other"), 0.0) + u
    per_group_share = {
        g: (u / total_lift) if abs(total_lift) > 1e-9 else 0.0
        for g, u in per_group_units.items()
    }

    reconciliation = (total_base + sum(per_feature_units.values())) - total_predicted
    reconciliation_pct = (reconciliation / total_predicted) if abs(total_predicted) > 1e-9 else 0.0

    return {
        "total_observed": total_observed,
        "total_predicted": total_predicted,
        "total_base": total_base,
        "total_lift": total_lift,
        "per_feature_units": per_feature_units,
        "per_feature_share": per_feature_share,
        "per_group_units": per_group_units,
        "per_group_share": per_group_share,
        "reconciliation_unit_error": reconciliation,
        "reconciliation_pct_error": reconciliation_pct,
    }


def is_finite_number(v: float) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(float(v)) and math.isfinite(float(v))
