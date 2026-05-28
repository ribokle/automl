"""Ablation decomposition for non-linear predictors.

The closed-form :mod:`core.decomp.due_to` path works for OLS because the
log-units decomposition is additive: ``log(units) = α + Σ βᵢ·xᵢ``. For
LightGBM (or any tree ensemble) there is no equivalent identity, so we
fall back to a group-wise ablation:

::

    pred_log         = predictor(observed_row)                      # full prediction
    base_log         = predictor(reference_row)                     # everything at ref
    delta_log_group  = predictor(group_observed, others_reference)  # one group "on"
                         - base_log

Per-group **unit-space** contributions are then allocated by sharing the
total lift ``exp(pred_log) - exp(base_log)`` proportionally across groups
by ``delta_log_group``. When all groups cancel (``Σ delta_log ≈ 0``) every
group gets zero and any rounding falls into the residual. This matches
the OLS path's identity so the per-PPG reconciliation diagnostic stays
meaningful end-to-end.

The output frame uses the same column shape as
:func:`core.decomp.due_to.decompose_ols_frame` — ``predicted``, ``base``,
``lift``, ``observed``, ``residual``, and one ``due_group_<g>`` column
per business group — so downstream summarisation can ignore which
attribution path ran.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from core.decomp.due_to import DEFAULT_DUMMY_FEATURES
from core.decomp.groups import GROUP_ORDER, group_for
from core.models.predictor import Predictor


def reference_frame(
    frame: pd.DataFrame, feature_cols: Iterable[str], *, base_price_col: str = "log_base_price"
) -> pd.DataFrame:
    """Build a "baseline" copy of ``frame`` for ablation.

    Continuous features baseline to their column mean. Promo + holiday
    dummies baseline to zero. ``log_price`` baselines to per-row
    ``log_base_price`` when present so the price contribution reflects
    deviation from the regular-price baseline. Columns absent from
    ``frame`` are added as zeros so the predictor sees a complete row.
    """
    ref = frame.copy().reset_index(drop=True)
    for col in feature_cols:
        if col == "log_price" and base_price_col in ref.columns:
            ref[col] = ref[base_price_col].astype(float).to_numpy()
            continue
        if col in DEFAULT_DUMMY_FEATURES:
            ref[col] = 0.0
            continue
        if col in ref.columns:
            ref[col] = float(ref[col].astype(float).mean())
        else:
            ref[col] = 0.0
    return ref


def _features_by_group(feature_cols: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for col in feature_cols:
        grouped.setdefault(group_for(col), []).append(col)
    return grouped


def decompose_via_ablation(
    predictor: Predictor,
    frame: pd.DataFrame,
    *,
    observed_col: str = "log_units",
    base_price_col: str = "log_base_price",
) -> pd.DataFrame:
    """Per-row group-ablation decomposition for any ``Predictor``.

    Returns a DataFrame with ``predicted``, ``base``, ``lift``,
    ``observed`` (when present), ``residual``, and one
    ``due_group_<group>`` column per business group present in the
    predictor's feature columns plus any zero-padded groups in
    :data:`core.decomp.groups.GROUP_ORDER` for shape consistency.
    """
    feature_cols = list(predictor.feature_cols)
    if not feature_cols:
        raise ValueError("predictor has no feature columns")

    work = frame.copy().reset_index(drop=True)
    ref = reference_frame(work, feature_cols, base_price_col=base_price_col)

    pred_log = predictor.predict_log(work)
    base_log = predictor.predict_log(ref)
    pred_units = np.exp(pred_log)
    base_units = np.exp(base_log)
    lift_units = pred_units - base_units

    features_by_group = _features_by_group(feature_cols)
    delta_log_by_group: dict[str, np.ndarray] = {}
    for group, cols_in_group in features_by_group.items():
        hybrid = ref.copy()
        for col in cols_in_group:
            if col in work.columns:
                hybrid[col] = work[col].astype(float).to_numpy()
        hybrid_log = predictor.predict_log(hybrid)
        delta_log_by_group[group] = hybrid_log - base_log

    total_delta = np.zeros(len(work), dtype=float)
    for arr in delta_log_by_group.values():
        total_delta = total_delta + arr
    safe_total = np.where(np.abs(total_delta) < 1e-9, np.nan, total_delta)

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

    for group in GROUP_ORDER:
        delta = delta_log_by_group.get(group)
        if delta is None:
            out[f"due_group_{group}"] = 0.0
            continue
        share = delta / safe_total
        share = np.where(np.isnan(share), 0.0, share)
        out[f"due_group_{group}"] = lift_units * share

    return out


def summarise_groups(weekly: pd.DataFrame) -> dict:
    """Aggregate ablation weekly into the same row shape as :func:`summarise_ppg`.

    Reports totals + per-group dollar (unit) contributions + shares +
    reconciliation diagnostic. Per-feature granularity isn't available
    here (ablation runs at group granularity) so ``per_feature_units`` and
    ``per_feature_share`` come back empty.
    """
    total_predicted = float(weekly["predicted"].sum())
    total_base = float(weekly["base"].sum())
    total_observed = float(weekly.get("observed", weekly["predicted"]).sum())
    total_lift = total_predicted - total_base

    per_group_units: dict[str, float] = {}
    for group in GROUP_ORDER:
        col = f"due_group_{group}"
        if col in weekly.columns:
            per_group_units[group] = float(weekly[col].sum())
    per_group_share = {
        g: (u / total_lift) if abs(total_lift) > 1e-9 else 0.0
        for g, u in per_group_units.items()
    }

    reconciliation = (total_base + sum(per_group_units.values())) - total_predicted
    reconciliation_pct = (reconciliation / total_predicted) if abs(total_predicted) > 1e-9 else 0.0

    return {
        "total_observed": total_observed,
        "total_predicted": total_predicted,
        "total_base": total_base,
        "total_lift": total_lift,
        "per_feature_units": {},
        "per_feature_share": {},
        "per_group_units": per_group_units,
        "per_group_share": per_group_share,
        "reconciliation_unit_error": reconciliation,
        "reconciliation_pct_error": reconciliation_pct,
    }
