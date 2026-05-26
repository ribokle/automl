"""Escalation loop: fit candidates in order, stop at first acceptable fit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.models.library import registry  # noqa: F401 — ensure registration
from core.models.library.base import FitContext
from core.models.router.escalation import run_escalation


def _demand_frame(n: int = 160, beta: float = -1.5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_price = rng.normal(0.0, 0.25, n)
    ctrl = rng.normal(0.0, 1.0, n)
    log_units = 5.0 + beta * log_price + 0.3 * ctrl + rng.normal(0.0, 0.05, n)
    return pd.DataFrame(
        {
            "log_price": log_price,
            "log_units": log_units,
            "ctrl": ctrl,
            "week_start": pd.date_range("2021-01-01", periods=n, freq="W"),
        }
    )


def _ctx(test: pd.DataFrame | None) -> FitContext:
    return FitContext(ppg_id="P1", controls=["ctrl"], test=test)


def test_stops_at_first_acceptable_candidate() -> None:
    frame = _demand_frame()
    train, test = frame.iloc[:120], frame.iloc[120:]
    res = run_escalation(
        ["ridge", "loglog_ols", "lightgbm"],
        train,
        _ctx(test),
        max_candidates=4,
        magnitude_ceiling=8.0,
        wape_floor=0.30,
    )
    assert res.winner is not None
    assert res.winner.sign_ok
    # ridge fits the clean synthetic well -> we stop before trying lightgbm.
    assert len(res.attempts) == 1
    assert res.winner.model == "ridge"


def test_unknown_candidate_keys_are_skipped() -> None:
    frame = _demand_frame()
    res = run_escalation(
        ["does_not_exist", "loglog_ols"],
        frame,
        _ctx(None),
        max_candidates=4,
        magnitude_ceiling=8.0,
        wape_floor=0.30,
    )
    assert res.winner is not None
    assert res.winner.model == "loglog_ols"


def test_max_candidates_caps_attempts() -> None:
    frame = _demand_frame()
    # wape_floor=0 forces every candidate to be "unacceptable" so the loop runs
    # to the candidate cap instead of stopping early.
    res = run_escalation(
        ["loglog_ols", "ridge", "lasso", "elasticnet", "lightgbm"],
        frame,
        _ctx(None),
        max_candidates=2,
        magnitude_ceiling=8.0,
        wape_floor=0.0,
    )
    assert len(res.attempts) == 2
