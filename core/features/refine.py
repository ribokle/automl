"""Drop collinear features.

`refine_features` returns the set of features that satisfies:
- max VIF < `vif_threshold`
- max |off-diagonal correlation| <= `corr_threshold`

VIF for column i is 1 / (1 - R²_i), where R²_i comes from regressing column i
against the remaining columns with a least-squares fit. We compute it via
numpy's normal equations rather than depending on statsmodels.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _r2_against_rest(X: np.ndarray, i: int) -> float:
    n = X.shape[1]
    y = X[:, i]
    rest = np.delete(X, i, axis=1)
    rest_with_const = np.column_stack([np.ones(rest.shape[0]), rest])
    # Least-squares; rcond=None silences future-warning, returns coefs.
    coef, *_ = np.linalg.lstsq(rest_with_const, y, rcond=None)
    pred = rest_with_const @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def compute_vif(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    sub = df[columns].dropna().to_numpy(dtype=float)
    if sub.shape[1] < 2:
        return {c: 1.0 for c in columns}
    out: dict[str, float] = {}
    for i, c in enumerate(columns):
        r2 = _r2_against_rest(sub, i)
        out[c] = float("inf") if r2 >= 0.999999 else 1.0 / (1.0 - r2)
    return out


def max_abs_offdiag(corr: pd.DataFrame) -> tuple[str, str, float]:
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, 0.0)
    flat = np.argmax(np.abs(arr))
    i, j = divmod(int(flat), arr.shape[1])
    return corr.columns[i], corr.columns[j], float(arr[i, j])


def refine_features(
    df: pd.DataFrame,
    candidates: list[str],
    vif_threshold: float = 10.0,
    corr_threshold: float = 0.95,
    protected: list[str] | None = None,
) -> dict[str, Any]:
    """Iteratively drop until VIF and |corr| thresholds are met.

    Strategy: first prune any |corr| > corr_threshold pair (drop the column with
    higher mean absolute corr to the rest, ties broken alphabetically). Then
    drop the highest-VIF column until max(VIF) < vif_threshold.

    `protected` columns are never dropped — used to lock in primary modelling
    variables like log_price that downstream stages need by name.
    """
    kept = list(candidates)
    locked = set(protected or [])
    dropped: list[dict[str, Any]] = []

    def _droppable(*cols: str) -> tuple[str, ...]:
        return tuple(c for c in cols if c not in locked)

    while True:
        sub = df[kept].apply(pd.to_numeric, errors="coerce").dropna()
        if sub.shape[1] < 2:
            break
        corr = sub.corr()
        a, b, r = max_abs_offdiag(corr)
        if abs(r) <= corr_threshold:
            break
        candidates_for_drop = _droppable(a, b)
        if not candidates_for_drop:
            break
        if len(candidates_for_drop) == 1:
            drop = candidates_for_drop[0]
        else:
            mean_corr = corr.abs().mean()
            drop = a if (mean_corr[a], a) >= (mean_corr[b], b) else b
        dropped.append({"feature": drop, "reason": f"|corr|={abs(r):.2f} with {b if drop == a else a}"})
        kept.remove(drop)

    while True:
        if len(kept) < 2:
            break
        vif = compute_vif(df, kept)
        droppable = [c for c in kept if c not in locked]
        if not droppable:
            break
        worst = max(droppable, key=lambda k: vif[k])
        if vif[worst] < vif_threshold:
            break
        dropped.append({"feature": worst, "reason": f"VIF={vif[worst]:.1f}"})
        kept.remove(worst)

    sub = df[kept].apply(pd.to_numeric, errors="coerce").dropna()
    final_corr = sub.corr()
    final_vif = compute_vif(df, kept) if len(kept) >= 2 else {c: 1.0 for c in kept}
    a, b, r = max_abs_offdiag(final_corr) if len(kept) >= 2 else ("", "", 0.0)

    return {
        "kept": kept,
        "dropped": dropped,
        "vif": {k: float(v) for k, v in final_vif.items()},
        "max_vif": float(max(final_vif.values())) if final_vif else 0.0,
        "max_abs_corr": float(abs(r)),
        "max_abs_corr_pair": [a, b],
    }
