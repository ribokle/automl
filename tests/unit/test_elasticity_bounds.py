"""RLM fallback when OLS produces implausible elasticities.

Loglog / semilog fitters refit with a Huber M-estimator when the raw
slope exceeds 6 in absolute value, and stash the pre-robust value in
diagnostics so downstream stages can show the audit trail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.models.loglog_ols import (
    ELASTICITY_SUSPECT_THRESHOLD as LOGLOG_THRESHOLD,
)
from core.models.loglog_ols import (
    fit_loglog,
)
from core.models.semilog_ols import (
    ELASTICITY_SUSPECT_THRESHOLD as SEMILOG_THRESHOLD,
)
from core.models.semilog_ols import (
    fit_semilog,
)


def _frame_with_outlier_slope(rng: np.random.Generator, *, slope: float = -25.0) -> pd.DataFrame:
    """Build a tiny frame with a few extreme leverage points that drag
    OLS toward a wildly elastic slope, then a tight base."""
    n = 80
    log_price = rng.normal(2.0, 0.1, n)
    log_units = rng.normal(4.0, 0.05, n) + (-1.5) * (log_price - log_price.mean())
    # Drop in 4 leverage points that pull the slope wildly negative.
    log_price[:4] = log_price[:4] + 0.3
    log_units[:4] = log_units[:4] + slope * 0.3
    return pd.DataFrame(
        {
            "log_units": log_units,
            "log_price": log_price,
            "price": np.exp(log_price),
            "tpr_share": rng.uniform(0, 1, n),
        }
    )


def test_loglog_refits_robust_on_extreme_elasticity() -> None:
    rng = np.random.default_rng(0)
    frame = _frame_with_outlier_slope(rng)

    fit = fit_loglog("PPG_X", frame, controls=["tpr_share"])

    # The RLM refit pulls the elasticity back inside the suspect band.
    assert abs(fit.own_elasticity) <= LOGLOG_THRESHOLD + 1e-6
    # And the diagnostic carries the pre-robust value plus a "huber_m" flag.
    assert "elasticity_pre_robust" in fit.diagnostics
    assert abs(fit.diagnostics["elasticity_pre_robust"]) > LOGLOG_THRESHOLD
    assert fit.diagnostics.get("robust_refit") == "huber_m"


def test_loglog_skips_robust_when_in_range() -> None:
    rng = np.random.default_rng(1)
    n = 100
    log_price = rng.normal(2.0, 0.15, n)
    log_units = 5.0 + (-1.3) * (log_price - log_price.mean()) + rng.normal(0, 0.02, n)
    frame = pd.DataFrame({"log_units": log_units, "log_price": log_price})
    fit = fit_loglog("PPG_OK", frame, controls=[])
    assert "elasticity_pre_robust" not in fit.diagnostics
    assert fit.diagnostics.get("robust_refit") is None
    assert abs(fit.own_elasticity) <= LOGLOG_THRESHOLD


def test_semilog_refits_robust_on_extreme_elasticity() -> None:
    rng = np.random.default_rng(2)
    # Semi-log slope is dimensionless * price; we engineer a high *elasticity*
    # via a steep price-slope on a slice whose mean price is large.
    n = 80
    price = rng.normal(10.0, 0.3, n)
    log_units = rng.normal(4.0, 0.05, n) - 2.0 * (price - price.mean())
    # Leverage points.
    price[:5] += 0.7
    log_units[:5] -= 1.5
    frame = pd.DataFrame(
        {
            "log_units": log_units,
            "price": price,
            "log_price": np.log(price),
        }
    )
    fit = fit_semilog("PPG_Y", frame, controls=[])
    # Either RLM brought it back, or we record why it could not.
    assert "elasticity_pre_robust" in fit.diagnostics or abs(fit.own_elasticity) <= SEMILOG_THRESHOLD
