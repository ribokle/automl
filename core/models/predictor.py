"""Shared prediction abstraction for downstream stages.

Phase 4a's decomposition / simulation / optimisation / validation stages
all branched on ``winner_model`` and only supported OLS. Phase 4b adds a
``Predictor`` interface so the agents stop caring whether the winner is
OLS or LightGBM: they ask the predictor for ``predict_log`` /
``predict_units`` and for cell-level scoring.

OLS predictors evaluate the closed-form ``α + Σ βᵢ·xᵢ`` directly from
the modelling agent's saved coefficients. LightGBM predictors refit the
booster on the PPG's training slice (deterministic given ``random_state=0``
inside :mod:`core.models.lightgbm_model`) and wrap the trained estimator.
Refitting is a few hundred milliseconds per PPG; persisting + reloading
the booster would be marginally faster but adds a feature-column-ordering
contract this codebase doesn't otherwise need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.models.metrics import chronological_split


OLS_KINDS = {"loglog_ols", "semilog_ols"}


@dataclass
class Predictor:
    """Unified wrapper around the winning model for one PPG."""

    ppg_id: str
    model_kind: str
    feature_cols: list[str]
    coefficients: dict[str, float] = field(default_factory=dict)
    booster: Any = None  # LGBMRegressor when model_kind == "lightgbm"

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict ``log_units`` for every row in ``frame``."""
        if self.model_kind in OLS_KINDS:
            return _predict_log_ols(self.coefficients, frame)
        if self.model_kind == "lightgbm":
            if self.booster is None:
                raise RuntimeError("lightgbm predictor missing booster")
            X = _design_for_lightgbm(frame, self.feature_cols)
            return np.asarray(self.booster.predict(X), dtype=float)
        raise ValueError(f"unsupported model_kind={self.model_kind!r}")

    def predict_units(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(frame))


def _predict_log_ols(coefs: dict[str, float], frame: pd.DataFrame) -> np.ndarray:
    """Closed-form ``α + Σ βᵢ·xᵢ`` over rows. Missing columns contribute zero."""
    if "const" not in coefs:
        raise ValueError("coefficients missing 'const' intercept")
    log_units = np.full(len(frame), float(coefs["const"]), dtype=float)
    for col, beta in coefs.items():
        if col == "const":
            continue
        if col in frame.columns:
            log_units += float(beta) * frame[col].astype(float).to_numpy()
    return log_units


def _design_for_lightgbm(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Project ``frame`` onto the LightGBM feature columns, filling missing
    columns with zero so the booster never sees ``NaN`` predictors.
    """
    X = pd.DataFrame(index=frame.index)
    for col in cols:
        if col in frame.columns:
            X[col] = frame[col].astype(float).to_numpy()
        else:
            X[col] = 0.0
    return X


def build_predictor(
    modeling_row: dict,
    frame: pd.DataFrame,
    controls: list[str],
    *,
    test_ratio: float | None = None,
) -> Predictor:
    """Construct a predictor for one PPG given the modelling agent's row.

    For OLS winners we lift the coefficients straight off the winner blob.
    For LightGBM winners we refit on the same train slice the modelling
    agent used (chronological 80/20) unless ``test_ratio`` is overridden;
    pass ``test_ratio=0.0`` to fit on the full frame (used for in-sample
    decomposition where every observed week must be attributed).
    """
    ppg_id = str(modeling_row["ppg_id"])
    winner = modeling_row.get("winner") or {}
    kind = str(modeling_row.get("winner_model") or winner.get("model") or "")

    if kind in OLS_KINDS:
        coefs = {k: float(v) for k, v in (winner.get("coefficients") or {}).items()}
        if not coefs:
            raise ValueError(f"{ppg_id}: OLS winner has no coefficients")
        feature_cols = [c for c in coefs if c != "const"]
        return Predictor(
            ppg_id=ppg_id,
            model_kind=kind,
            feature_cols=feature_cols,
            coefficients=coefs,
        )

    if kind == "lightgbm":
        if test_ratio is None:
            train, _ = chronological_split(frame, test_ratio=0.2)
        elif test_ratio <= 0.0:
            train = frame
        else:
            train, _ = chronological_split(frame, test_ratio=test_ratio)
        feature_cols, booster = _fit_lightgbm_booster(train, controls)
        return Predictor(
            ppg_id=ppg_id,
            model_kind=kind,
            feature_cols=feature_cols,
            booster=booster,
        )

    raise ValueError(f"unsupported winner_model={kind!r} for {ppg_id}")


def _fit_lightgbm_booster(
    frame: pd.DataFrame, controls: list[str]
) -> tuple[list[str], Any]:
    """Refit LightGBM with the same hyper-parameters + column filter as
    :func:`core.models.lightgbm_model.fit_lightgbm`.

    Returns ``(feature_cols, fitted_estimator)``. The estimator is the bare
    ``LGBMRegressor`` so the predictor can call ``.predict`` directly. The
    hyper-parameters are coupled to ``fit_lightgbm`` by hand; if they drift,
    both call sites need to change together.
    """
    from lightgbm import LGBMRegressor

    usable = [c for c in controls if c in frame.columns and c not in ("log_price", "log_units")]
    usable = [c for c in usable if frame[c].nunique(dropna=True) > 1]
    feature_cols = ["log_price"] + usable
    sub = frame[["log_units", *feature_cols]].dropna()
    X = sub[feature_cols].astype(float).copy()
    y = sub["log_units"].astype(float).to_numpy()
    # Mirror fit_lightgbm's monotone constraint on log_price (column 0).
    monotone_constraints = [-1] + [0] * (len(feature_cols) - 1)
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        random_state=0,
        monotone_constraints=monotone_constraints,
        verbosity=-1,
    )
    model.fit(X, y)
    return feature_cols, model
