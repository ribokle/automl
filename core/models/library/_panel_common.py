"""Shared panel-data fitting for the panel family (optional dep: linearmodels).

Panel models pool ONE PPG across its entities (stores) — entity = ``grain_unit``,
time = ``week_start`` — so the agent passes the full multi-PPG frame and the
plugin selects + reshapes its target PPG. They report the own-price elasticity
(the log_price coefficient net of entity heterogeneity) but are not
price-sweepable, so they stay out of ``predictor.PREDICTABLE_MODELS``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from core.models.result import Capability, ModelResult, ProblemType

LOG_PRICE = "log_price"
TARGET = "log_units"
ENTITY = "grain_unit"
TIME = "week_start"


def _usable_controls(frame: pd.DataFrame, controls: list[str]) -> list[str]:
    cols = [c for c in controls if c in frame.columns and c not in (LOG_PRICE, TARGET)]
    return [c for c in cols if frame[c].nunique(dropna=True) > 1]


def fit_panel(
    ppg_id: str,
    frame: pd.DataFrame,
    controls: list[str],
    *,
    model_name: str,
    fitter: Callable[[pd.Series, pd.DataFrame], Any],
) -> ModelResult:
    """Reshape the target PPG to a (entity, time) panel and fit ``fitter``,
    which returns a linearmodels results object exposing ``params`` /
    ``std_errors`` indexed by regressor name."""
    if ENTITY not in frame.columns:
        raise ValueError("panel models require a store-grain frame (grain_unit column)")
    sub = frame[frame["ppg_id"] == ppg_id]
    if sub[ENTITY].nunique() < 2:
        raise ValueError(f"{ppg_id}: need >= 2 entities for a panel fit")

    usable = _usable_controls(sub, controls)
    cols = [LOG_PRICE, *usable]
    panel = sub[[ENTITY, TIME, TARGET, *cols]].dropna().copy()
    # linearmodels requires a date-like or numeric time index; CSV round-trips
    # turn week_start into a string, so coerce (falling back to an ordinal).
    time = pd.to_datetime(panel[TIME], errors="coerce")
    if time.isna().any():
        time = panel[TIME].rank(method="dense").astype(int)
    panel[TIME] = time
    panel = panel.set_index([ENTITY, TIME]).sort_index()
    if len(panel) < 12:
        raise ValueError(f"{ppg_id}: too few panel observations ({len(panel)})")

    y = panel[TARGET].astype(float)
    X = panel[cols].astype(float)
    res = fitter(y, X)

    own = float(res.params[LOG_PRICE])
    try:
        std_err = float(res.std_errors[LOG_PRICE])
    except Exception:  # noqa: BLE001
        std_err = None

    diagnostics: dict[str, Any] = {
        "n_entities": int(panel.index.get_level_values(ENTITY).nunique()),
        "rsquared": float(getattr(res, "rsquared", float("nan"))),
    }
    return ModelResult(
        ppg_id=ppg_id,
        model=model_name,
        problem_type=ProblemType.PANEL,
        own_elasticity=own,
        std_err=std_err,
        p_value=None,
        r_squared=diagnostics["rsquared"],
        n_obs=int(len(panel)),
        controls=usable,
        coefficients={},
        diagnostics=diagnostics,
        capabilities=Capability.SCALAR_ELASTICITY | Capability.NEEDS_PANEL,
    )
