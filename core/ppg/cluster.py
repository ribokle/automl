"""PPG clustering algorithm.

Strategy (mirrors how a CPG analyst would do this by hand):

1. **Hard partition** by `(brand, category)`. Two SKUs from different brands or
   different categories can never share a Price-Pack Group — this is an
   industry convention, not a learned constraint.
2. **Within each partition**, look for evidence of sub-groups driven by a
   distinct price tier or pack tier. Use silhouette-scored k-means over
   (log median_price, pack_ordinal) and split only if the best k>=2 silhouette
   exceeds `split_threshold` AND the price spread is meaningful.
3. **Assign PPG ids** as `PPG_AUTO_<n>` so the agent's mapping is clearly its
   own (later it can be reconciled against ground-truth PPG ids if those exist
   in the source data).
4. **Confidence** per SKU is `1 - (distance_to_own_centroid / distance_to_nearest_other_centroid)`,
   clipped to [0,1]. SKUs that are unambiguously inside their cluster score ~1.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusterParams:
    # The within-(brand, category) split fires only when ALL of these hold:
    #   - silhouette of the best k>=2 split >= split_threshold
    #   - smallest pairwise centroid gap (in log-price) >= min_centroid_gap_log
    # The second guard prevents splitting on within-PPG pack-size variation
    # (~+/-25% price effect) while still catching genuine premium/value tier
    # separation (~+50% price step or larger).
    split_threshold: float = 0.70
    min_centroid_gap_log: float = 0.50  # >= +65% price step between tiers
    max_k: int = 4
    random_state: int = 0


def _subcluster(features: np.ndarray, max_k: int, random_state: int) -> tuple[int, np.ndarray, float]:
    """Pick best k by silhouette; return (k, labels, score). k>=2 only."""
    n = features.shape[0]
    if n < 4:
        return 1, np.zeros(n, dtype=int), 0.0
    best_k, best_labels, best_score = 1, np.zeros(n, dtype=int), -1.0
    for k in range(2, min(max_k, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(features)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(features, labels))
        if score > best_score:
            best_k, best_labels, best_score = k, labels, score
    return best_k, best_labels, best_score


def cluster_ppgs(sku_features: pd.DataFrame, params: ClusterParams | None = None) -> pd.DataFrame:
    """Return SKU-level PPG assignments.

    Output columns: sku, brand, category, pack_size, ppg_id, confidence,
    rationale.
    """
    if params is None:
        params = ClusterParams()
    df = sku_features.copy()
    df["log_price"] = np.log(df["median_price"].clip(lower=1e-6))
    # Cluster on log-price only - pack-size variation within a brand/category
    # is a within-PPG attribute, not a tier signal.
    feature_cols = ["log_price"]

    assignments: list[dict[str, Any]] = []
    next_ppg_idx = 1

    for (brand, category), group in df.groupby(["brand", "category"], sort=True):
        X = group[feature_cols].to_numpy(dtype=float)
        n = X.shape[0]
        price_spread = float(X[:, 0].max() - X[:, 0].min())

        if n >= 4 and price_spread >= params.min_centroid_gap_log:
            k, labels, score = _subcluster(X, params.max_k, params.random_state)
        else:
            k, labels, score = 1, np.zeros(n, dtype=int), 0.0

        # Build provisional centroids and check the smallest pairwise gap.
        if k >= 2:
            prov_centroids = [X[labels == c].mean(axis=0) for c in range(k)]
            sorted_means = sorted(float(c[0]) for c in prov_centroids)
            min_gap = min(sorted_means[i + 1] - sorted_means[i] for i in range(len(sorted_means) - 1))
        else:
            min_gap = 0.0

        split = k >= 2 and score >= params.split_threshold and min_gap >= params.min_centroid_gap_log
        local_labels = labels if split else np.zeros(n, dtype=int)
        local_k = int(local_labels.max() + 1) if split else 1

        centroids: list[np.ndarray] = [X[local_labels == c].mean(axis=0) for c in range(local_k)]
        ppg_ids: list[str] = []
        for c in range(local_k):
            ppg_ids.append(f"PPG_AUTO_{next_ppg_idx:02d}")
            next_ppg_idx += 1

        for i, (_, sku_row) in enumerate(group.reset_index(drop=True).iterrows()):
            c = int(local_labels[i])
            d_own = float(np.linalg.norm(X[i] - centroids[c]))
            if local_k > 1:
                others = [float(np.linalg.norm(X[i] - centroids[j])) for j in range(local_k) if j != c]
                d_other = min(others)
                conf = max(0.0, min(1.0, 1.0 - (d_own / (d_other + 1e-9))))
            else:
                conf = 0.92  # single-cluster confidence is high but not 1.0
            ppg_id = ppg_ids[c]
            rationale = (
                f"{brand}/{category}"
                + (f" split into {local_k} price tiers (silhouette={score:.2f})" if split else "")
                + (
                    f"; centroid log_price={centroids[c][0]:.2f}"
                    if split
                    else f"; single price tier, log_price spread={price_spread:.2f}"
                )
            )
            assignments.append(
                {
                    "sku": sku_row["sku"],
                    "brand": brand,
                    "category": category,
                    "pack_size": sku_row["pack_size"],
                    "median_price": float(sku_row["median_price"]),
                    "ppg_id": ppg_id,
                    "confidence": round(float(conf), 3),
                    "rationale": rationale,
                }
            )

    return pd.DataFrame(assignments)


def label_match_accuracy(predicted: pd.Series, truth: pd.Series) -> dict[str, float]:
    """Cluster-label-agnostic accuracy via Hungarian-style assignment.

    Builds a confusion matrix between predicted and truth labels and finds the
    best one-to-one match. Returns {"accuracy": ..., "matched_pairs": int}.
    """
    from scipy.optimize import linear_sum_assignment

    pred_labels = sorted(predicted.unique())
    truth_labels = sorted(truth.unique())
    p_idx = {l: i for i, l in enumerate(pred_labels)}
    t_idx = {l: i for i, l in enumerate(truth_labels)}
    cm = np.zeros((len(pred_labels), len(truth_labels)), dtype=int)
    for p, t in zip(predicted, truth, strict=True):
        cm[p_idx[p], t_idx[t]] += 1
    # Hungarian maximises -> minimise (-cm).
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = int(cm[row_ind, col_ind].sum())
    total = int(cm.sum())
    return {"accuracy": matched / total if total else 0.0, "matched_pairs": matched, "n_total": total}


def apply_mapping_to_panel(
    duckdb_path: Path,
    assignments: pd.DataFrame,
    table: str = "main.panel",
) -> int:
    """Rewrite ``main.panel.ppg_id`` from the per-SKU assignments.

    The clusterer invents fresh ``PPG_AUTO_*`` ids that don't match the
    panel's original ``ppg_id`` column (whatever it was loaded with —
    truth labels on the synthetic data, often empty or sku-derived on
    real data). Every downstream agent groups by ``main.panel.ppg_id``
    via ``ppg_week_aggregate``, so unless we propagate the mapping the
    modelling / decomposition / simulation stages silently see zero
    rows per PPG.

    Returns the number of panel rows whose ``ppg_id`` was set. Unmatched
    SKUs (none under normal flow) keep their original value.
    """
    if "sku" not in assignments.columns or "ppg_id" not in assignments.columns:
        raise ValueError("assignments must have sku + ppg_id columns")

    con = duckdb.connect(str(duckdb_path))
    try:
        con.register("assignments_df", assignments[["sku", "ppg_id"]])
        con.execute(
            f"""
            UPDATE {table} AS p
            SET ppg_id = a.ppg_id
            FROM assignments_df AS a
            WHERE p.sku = a.sku
            """
        )
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        con.close()
