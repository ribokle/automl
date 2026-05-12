from __future__ import annotations

import numpy as np

from synthetic.elasticity_spec import PPGS
from synthetic.generator import generate_panel


def test_panel_shape_and_columns():
    df, truth = generate_panel(seed=7, n_weeks=20, skus_per_ppg=3)
    expected_rows = len(PPGS) * 3 * 5 * 20  # ppgs * skus_per_ppg * stores * weeks
    assert len(df) == expected_rows
    required = {
        "sku", "week_start", "store_id", "region", "units", "price", "base_price",
        "tpr_flag", "display_flag", "feature_flag", "distribution_acv",
        "competitor_price", "holiday", "category", "brand", "pack_size", "segment", "ppg_id",
    }
    assert required.issubset(df.columns)
    assert (df["price"] > 0).all()
    assert (df["base_price"] > 0).all()
    assert (df["units"] >= 0).all()
    assert df["tpr_flag"].isin([0, 1]).all()


def test_elasticity_signal_recoverable_via_loglog():
    """Sanity: a per-PPG log-log regression should recover negative elasticities within 50%
    of truth on a clean synthetic panel."""
    df, truth = generate_panel(seed=11, n_weeks=104, skus_per_ppg=6)
    recoveries = []
    for ppg_id, spec in truth["ppgs"].items():
        sub = df[df["ppg_id"] == ppg_id]
        sub = sub[sub["units"] > 0]
        x = np.log(sub["price"].to_numpy())
        y = np.log(sub["units"].to_numpy())
        # OLS via numpy
        A = np.vstack([x, np.ones_like(x)]).T
        coef, _ = np.linalg.lstsq(A, y, rcond=None)[:2]
        elasticity = float(coef[0])
        true_e = spec["own_elasticity"]
        recoveries.append((ppg_id, elasticity, true_e))
        # Must be negative.
        assert elasticity < 0, f"{ppg_id} elasticity not negative: {elasticity}"
    # At least half within 50% of truth — a loose smoke check; tightened in Phase 3.
    within = sum(abs(e - t) / abs(t) <= 0.5 for _, e, t in recoveries)
    assert within >= len(recoveries) // 2
