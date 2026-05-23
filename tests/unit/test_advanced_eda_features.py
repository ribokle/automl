"""Unit tests for the pure stats helpers behind the advanced_eda agent.

Each helper is tested in isolation with hand-crafted inputs whose answer we
know. The integration test exercises the agent end-to-end against the
synthetic panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.advanced_eda import (
    acf_pacf,
    cardinality_report,
    cross_ppg_correlation,
    detect_change_points,
    detect_forward_buy,
    detect_pantry_loading,
    detect_stockouts,
    distribution_report,
    holiday_lift_table,
    isolation_forest_anomalies,
    pareto_abc,
    price_ladder,
    promo_calendar,
    promo_lift_sketches,
    stationarity_tests,
    stl_decomposition,
    store_variability,
    time_series_diagnostics,
    top_k_ppgs_by_revenue,
)


def _weekly_index(n: int, start: str = "2022-01-03") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="W")


def _synthetic_ppg(
    ppg_id: str,
    n_weeks: int = 104,
    base: float = 100.0,
    seasonal_amp: float = 30.0,
    trend: float = 0.5,
    noise: float = 5.0,
    seed: int = 0,
    promo_share: float = 0.2,
    base_price: float = 5.0,
    discount_at_promo: float = 0.8,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = _weekly_index(n_weeks)
    season = seasonal_amp * np.sin(np.arange(n_weeks) * 2 * np.pi / 52)
    drift = trend * np.arange(n_weeks)
    units = (base + season + drift + rng.normal(0, noise, n_weeks)).clip(min=1).round().astype(int)
    tpr_flag = (rng.random(n_weeks) < promo_share).astype(float)
    return pd.DataFrame(
        {
            "ppg_id": ppg_id,
            "week_start": weeks,
            "units": units,
            "price": np.where(tpr_flag == 1, base_price * discount_at_promo, base_price),
            "base_price": base_price,
            "discount_depth": np.where(tpr_flag == 1, 1 - discount_at_promo, 0.0),
            "tpr_share": tpr_flag,
            "display_share": tpr_flag * 0.4,
            "feature_share": tpr_flag * 0.3,
            "distribution_acv": 90.0,
            "competitor_price": base_price,
            "is_holiday_week": 0,
        }
    )


def test_top_k_ppgs_by_revenue() -> None:
    df = pd.concat(
        [
            _synthetic_ppg("A", base=200, base_price=10),
            _synthetic_ppg("B", base=50, base_price=2),
            _synthetic_ppg("C", base=100, base_price=5),
        ],
        ignore_index=True,
    )
    top = top_k_ppgs_by_revenue(df, k=2)
    assert top[0] == "A"
    assert "B" not in top


def test_stl_decomposition_recovers_seasonality() -> None:
    df = _synthetic_ppg("A", seasonal_amp=40.0, noise=2.0)
    s = pd.Series(df["units"].values.astype(float), index=df["week_start"])
    out = stl_decomposition(s, period=52)
    assert out["available"] is True
    assert len(out["observed"]) == len(s)
    assert out["seasonal_amplitude"] > 30.0  # injected 40σ amplitude


def test_stl_too_short_returns_unavailable() -> None:
    s = pd.Series(np.arange(40, dtype=float), index=_weekly_index(40))
    out = stl_decomposition(s, period=52)
    assert out["available"] is False


def test_acf_pacf_returns_expected_shape() -> None:
    df = _synthetic_ppg("A")
    s = pd.Series(df["units"].values.astype(float))
    out = acf_pacf(s, nlags=12)
    assert out["available"] is True
    assert len(out["acf"]) == 13
    assert len(out["pacf"]) == 13
    assert 0.0 < out["ci"] < 0.5


def test_stationarity_distinguishes_random_walk_from_white_noise() -> None:
    rng = np.random.default_rng(0)
    white = pd.Series(rng.normal(0, 1, 200))
    rw = pd.Series(np.cumsum(rng.normal(0, 1, 200)))
    out_white = stationarity_tests(white)
    out_rw = stationarity_tests(rw)
    assert out_white["verdict"] in ("stationary", "inconclusive")
    assert out_rw["verdict"] in ("non_stationary", "inconclusive")
    # Stronger: ADF p-value should be much lower for white noise
    assert out_white["adf_p"] < out_rw["adf_p"]


def test_time_series_diagnostics_per_ppg() -> None:
    df = pd.concat(
        [_synthetic_ppg("A", seasonal_amp=40.0, noise=2.0), _synthetic_ppg("B", seasonal_amp=10.0, seed=1)],
        ignore_index=True,
    )
    out = time_series_diagnostics(df, ["A", "B"])
    assert len(out) == 2
    assert {r["ppg_id"] for r in out} == {"A", "B"}
    assert out[0]["n_weeks"] == 104
    assert out[0]["stl"]["available"]


def test_detect_stockouts_flags_zero_units_at_unchanged_price() -> None:
    df = _synthetic_ppg("A", n_weeks=20, promo_share=0.0)
    # Inject one stockout at week 10
    df.loc[10, "units"] = 0
    out = detect_stockouts(df)
    assert any(a["anomaly_type"] == "stockout" for a in out)
    # The flagged week should match
    assert any(a["ppg_id"] == "A" and "2022" in a["week_start"] for a in out)


def test_detect_stockouts_skips_zero_with_low_acv() -> None:
    df = _synthetic_ppg("A", n_weeks=20, promo_share=0.0)
    df.loc[10, "units"] = 0
    df.loc[10, "distribution_acv"] = 0  # not a stockout, just not distributed
    out = detect_stockouts(df)
    assert all(a["week_start"].split("T")[0] != df.loc[10, "week_start"].date().isoformat() for a in out)


def test_detect_pantry_loading_finds_spike_then_dip() -> None:
    df = _synthetic_ppg("A", n_weeks=60, noise=2.0, promo_share=0.0)
    # Inject a large spike at week 30, then a dip at week 31
    df.loc[30, "units"] = int(df["units"].mean() + 4 * df["units"].std())
    df.loc[31, "units"] = int(df["units"].mean() - 1.5 * df["units"].std())
    out = detect_pantry_loading(df)
    assert any(a["anomaly_type"] == "pantry_loading" and a["ppg_id"] == "A" for a in out)


def test_detect_forward_buy_finds_pre_increase_spike() -> None:
    df = _synthetic_ppg("A", n_weeks=40, noise=2.0, promo_share=0.0)
    # Spike at week 19, base_price increase at week 20
    df.loc[19, "units"] = int(df["units"].mean() + 3 * df["units"].std())
    df.loc[20:, "base_price"] = 5.5  # +10%
    out = detect_forward_buy(df)
    assert any(a["anomaly_type"] == "forward_buy" and a["ppg_id"] == "A" for a in out)


def test_isolation_forest_skips_tiny_series() -> None:
    df = _synthetic_ppg("A", n_weeks=10)
    out = isolation_forest_anomalies(df, ["A"])
    assert out == []


def test_isolation_forest_flags_extreme_combos() -> None:
    df = _synthetic_ppg("A", n_weeks=60, promo_share=0.0)
    # Inject a clearly anomalous combination
    df.loc[30, "price"] = 0.01
    df.loc[30, "discount_depth"] = 0.99
    df.loc[30, "units"] = int(df["units"].max() * 5)
    out = isolation_forest_anomalies(df, ["A"])
    assert any(a["ppg_id"] == "A" for a in out)


def test_detect_change_points_recovers_known_jump() -> None:
    df = _synthetic_ppg("A", n_weeks=80, noise=2.0, base_price=5.0)
    df.loc[40:, "base_price"] = 6.5  # ~30% jump at week 40
    out = detect_change_points(df, ["A"], min_size=10)
    assert any(r["ppg_id"] == "A" and abs(r["delta_pct"] - 30) < 10 for r in out)


def test_distribution_report_emits_stats() -> None:
    df = _synthetic_ppg("A")
    out = distribution_report(df, ["units", "price"])
    keys = {r["column"] for r in out}
    assert keys == {"units", "price"}
    for row in out:
        assert "skew" in row and "kurtosis" in row and "shapiro_p" in row


def test_promo_lift_sketches_positive_when_units_spike_on_promo() -> None:
    df = _synthetic_ppg("A", n_weeks=60, noise=2.0, promo_share=0.3)
    # Inflate units on promo weeks
    promo_mask = df["tpr_share"].values > 0.5
    df.loc[promo_mask, "units"] = (df.loc[promo_mask, "units"].astype(float) * 1.5).astype(int)
    out = promo_lift_sketches(df)
    assert len(out) == 1
    tpr_lift = out[0]["lifts"]["tpr"]["lift"]
    assert tpr_lift > 1.2


def test_cross_ppg_correlation_groups_by_category() -> None:
    df = pd.concat(
        [_synthetic_ppg("A", seed=0), _synthetic_ppg("B", seed=1), _synthetic_ppg("C", seed=2)],
        ignore_index=True,
    )
    cats = {"A": "soda", "B": "soda", "C": "chips"}
    out = cross_ppg_correlation(df, cats)
    cat_names = [c["category"] for c in out["categories"]]
    # Only soda has >= 2 PPGs, so chips should be skipped
    assert "soda" in cat_names
    assert "chips" not in cat_names


def test_pareto_abc_classifies_top_revenue_as_A() -> None:
    df = pd.concat(
        [
            _synthetic_ppg("A", base=300, base_price=10),
            _synthetic_ppg("B", base=50, base_price=2),
            _synthetic_ppg("C", base=10, base_price=1),
        ],
        ignore_index=True,
    )
    # Add sku and brand columns for the dimensional curves
    raw = df.copy()
    raw["sku"] = raw["ppg_id"]
    raw["brand"] = raw["ppg_id"]
    raw["store_id"] = "S01"
    out = pareto_abc(raw, df)
    by_ppg = {r["ppg_id"]: r["abc_class"] for r in out["ppg_abc"]}
    assert by_ppg["A"] == "A"
    assert by_ppg["C"] == "C"


def test_price_ladder_recovers_negative_slope() -> None:
    df = _synthetic_ppg("A", n_weeks=80, promo_share=0.0, base_price=5.0)
    # Inject inverse relationship: lower price => higher units
    df["price"] = np.linspace(7.0, 3.0, len(df))
    df["units"] = (1000 / df["price"]).astype(int)
    out = price_ladder(df)
    assert len(out) == 1
    assert out[0]["price_volume_slope"] is not None
    assert out[0]["price_volume_slope"] < -0.5  # negative slope expected
    assert "caveat" in out[0]


def test_promo_calendar_emits_cells() -> None:
    df = _synthetic_ppg("A", n_weeks=20)
    out = promo_calendar(df)
    assert len(out["weeks"]) == 20
    assert out["ppgs"] == ["A"]
    assert len(out["matrix"]) == 1
    assert len(out["matrix"][0]) == 20
    # Should include at least some promo cells
    assert any(cell != "none" for cell in out["matrix"][0])


def test_holiday_lift_table_handles_missing_holiday() -> None:
    df = _synthetic_ppg("A")
    raw = df.rename(columns={}).copy()
    raw["holiday"] = ""
    raw.loc[26, "holiday"] = "Memorial Day"
    out = holiday_lift_table(raw)
    assert isinstance(out, list)
    if out:
        assert out[0]["holiday"] == "Memorial Day"


def test_cardinality_report_flags_rare_values() -> None:
    df = pd.DataFrame({"category": ["soda"] * 100 + ["beer"] * 3})
    out = cardinality_report(df, ["category"])
    assert len(out) == 1
    rows = {r["value"]: r for r in out[0]["values"]}
    assert rows["beer"]["rare"] is True
    assert rows["soda"]["rare"] is False


def test_store_variability_skips_when_no_store_column() -> None:
    df = _synthetic_ppg("A")
    out = store_variability(df)
    assert out == []


def test_store_variability_emits_per_ppg() -> None:
    rows = []
    for ppg in ("A", "B"):
        df = _synthetic_ppg(ppg)
        for store in ("S01", "S02", "S03"):
            grp = df.copy()
            grp["store_id"] = store
            rows.append(grp)
    raw = pd.concat(rows, ignore_index=True)
    out = store_variability(raw)
    assert len(out) == 2
    assert {r["ppg_id"] for r in out} == {"A", "B"}
    assert all(r["n_stores"] == 3 for r in out)
