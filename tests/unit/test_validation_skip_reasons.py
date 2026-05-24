"""Validation: empty-fold ``skipped`` verdict + WAPE cap.

When ``build_folds`` can't produce any CV splits (too few rows),
``evaluate_ppg`` now returns ``verdict="skipped"`` instead of "fail" so
the headline pass / warn / fail counts don't conflate "we never asked"
with "we asked and it broke". A run-away fold WAPE is capped at the
sanity ceiling and reported via a separate diagnostic.
"""
from __future__ import annotations

from core.validation.checks import WAPE_FOLD_CAP, evaluate_ppg


def test_no_folds_returns_skipped_verdict() -> None:
    v = evaluate_ppg("PPG_X", folds=[])
    assert v.verdict == "skipped"
    assert v.n_folds == 0
    assert any(c["name"] == "rolling_cv" and c["status"] == "skipped" for c in v.checks)


def test_runaway_wape_is_capped_and_flagged() -> None:
    # One catastrophic fold (WAPE 510) should be capped to WAPE_FOLD_CAP
    # rather than dragging the mean to thousands of percent.
    folds = [
        {"own_elasticity": -1.2, "test_wape": 0.25},
        {"own_elasticity": -1.3, "test_wape": 0.30},
        {"own_elasticity": -1.1, "test_wape": 0.20},
        {"own_elasticity": -1.4, "test_wape": 510.0},
    ]
    v = evaluate_ppg("PPG_Y", folds)
    assert v.n_wape_capped == 1
    # Cap forces the mean to stay well below the outlier value.
    assert v.wape_mean < WAPE_FOLD_CAP
    # Verdict picks up the wape_unstable diagnostic and reflects it.
    assert any(c["name"] == "wape_unstable" for c in v.checks)
    assert v.verdict in ("warn", "fail")
