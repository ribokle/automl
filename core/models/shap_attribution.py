"""SHAP-style feature attribution for elasticity winners.

Two paths share the same output shape so the UI can render either:

- **LightGBM** uses the native ``predict(X, pred_contrib=True)`` path. The
  last column of the returned matrix is the bias / expected value; the
  remaining columns are exact tree-SHAP attributions that satisfy
  ``ŷ = base + Σ shapᵢ`` per row.
- **OLS** (log-log + semi-log) has no native SHAP, but the linear identity
  ``ŷ = α + Σ βᵢ·xᵢ`` is itself an exact additive attribution. We centre
  each contribution on the train-mean ``x̄ᵢ`` so the base value matches the
  ensemble interpretation (predicted at the mean of the training data) and
  ``shapᵢ = βᵢ·(xᵢ - x̄ᵢ)``.

Both paths return a JSON-serialisable summary:
``{base_value, mean_abs_shap[], mean_shap[], beeswarm[]}`` where
``beeswarm`` is a capped per-row sample so the UI doesn't have to download
hundreds of rows for the dot-plot.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


MAX_BEESWARM_ROWS = 100


def _mean_summaries(
    shap_matrix: np.ndarray, feature_names: list[str]
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    mean_signed = np.mean(shap_matrix, axis=0)
    order = np.argsort(-mean_abs)
    mean_abs_rows = [
        {"feature": feature_names[i], "value": float(mean_abs[i])} for i in order
    ]
    mean_shap_rows = [
        {"feature": feature_names[i], "value": float(mean_signed[i])} for i in order
    ]
    return mean_abs_rows, mean_shap_rows


def _beeswarm(
    shap_matrix: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    *,
    max_rows: int,
    rng_seed: int,
) -> list[dict[str, Any]]:
    n_rows = shap_matrix.shape[0]
    if n_rows == 0:
        return []
    if n_rows > max_rows:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(n_rows, size=max_rows, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_rows)
    rows: list[dict[str, Any]] = []
    for r in idx:
        rows.append(
            {
                "row": int(r),
                "values": [
                    {
                        "feature": feature_names[c],
                        "shap": float(shap_matrix[r, c]),
                        "x": float(feature_values[r, c]),
                    }
                    for c in range(shap_matrix.shape[1])
                ],
            }
        )
    return rows


def lightgbm_shap_summary(
    model: LGBMRegressor,
    X: pd.DataFrame,
    *,
    max_beeswarm: int = MAX_BEESWARM_ROWS,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Compute tree-SHAP for a fitted LightGBM regressor and return a summary."""
    contribs = np.asarray(model.predict(X, pred_contrib=True), dtype=float)
    feature_names = list(X.columns)
    # LightGBM appends the expected-value column at the end.
    shap_matrix = contribs[:, :-1]
    base_value = float(np.mean(contribs[:, -1]))
    mean_abs_rows, mean_shap_rows = _mean_summaries(shap_matrix, feature_names)
    beeswarm = _beeswarm(
        shap_matrix,
        X.to_numpy(dtype=float),
        feature_names,
        max_rows=max_beeswarm,
        rng_seed=rng_seed,
    )
    return {
        "method": "tree_shap",
        "n_rows": int(shap_matrix.shape[0]),
        "n_features": int(shap_matrix.shape[1]),
        "base_value": base_value,
        "mean_abs_shap": mean_abs_rows,
        "mean_shap": mean_shap_rows,
        "beeswarm": beeswarm,
    }


def ols_shap_summary(
    coefficients: dict[str, float],
    X: pd.DataFrame,
    feature_names: list[str],
    *,
    intercept_key: str = "const",
    max_beeswarm: int = MAX_BEESWARM_ROWS,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Centred per-row contributions for a linear model.

    ``ŷ = α + Σ βᵢ·xᵢ`` is rewritten as ``ŷ = (α + Σ βᵢ·x̄ᵢ) + Σ βᵢ·(xᵢ - x̄ᵢ)``
    so ``base_value`` is the model's prediction at the train-data mean and
    every row's contributions sum to ``ŷ - base_value``.
    """
    cols = [f for f in feature_names if f in X.columns]
    sub = X[cols].astype(float).dropna()
    if sub.empty:
        return {
            "method": "ols_centred",
            "n_rows": 0,
            "n_features": len(cols),
            "base_value": float(coefficients.get(intercept_key, 0.0)),
            "mean_abs_shap": [],
            "mean_shap": [],
            "beeswarm": [],
        }
    means = sub.mean(axis=0).to_numpy()
    values = sub.to_numpy()
    betas = np.array([float(coefficients.get(c, 0.0)) for c in cols])
    shap_matrix = (values - means) * betas  # row-wise centred contribution
    intercept = float(coefficients.get(intercept_key, 0.0))
    base_value = intercept + float(np.sum(betas * means))
    mean_abs_rows, mean_shap_rows = _mean_summaries(shap_matrix, cols)
    beeswarm = _beeswarm(
        shap_matrix, values, cols, max_rows=max_beeswarm, rng_seed=rng_seed
    )
    return {
        "method": "ols_centred",
        "n_rows": int(shap_matrix.shape[0]),
        "n_features": int(shap_matrix.shape[1]),
        "base_value": base_value,
        "mean_abs_shap": mean_abs_rows,
        "mean_shap": mean_shap_rows,
        "beeswarm": beeswarm,
    }
