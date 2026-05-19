"""Empirical-Bayes hierarchical shrinkage.

Phase 3c verification anchor: the DerSimonian-Laird τ² estimator must (1)
collapse to zero when the per-PPG point estimates are mutually consistent
under their sampling SEs, (2) recover non-zero τ² when between-PPG
spread exceeds sampling noise, and (3) leave high-precision PPGs nearly
unshrunk while pulling noisy PPGs strongly toward the population mean.
"""
from __future__ import annotations

import math

import pytest

from core.models.bayes_hier import shrink, to_payload


def test_zero_heterogeneity_collapses_to_pooled_mean() -> None:
    """If every PPG estimates the same β within its SE, τ² hits 0 and
    every posterior mean equals the inverse-variance-weighted mean."""
    estimates = [("PPG01", -2.0, 0.1), ("PPG02", -2.0, 0.1), ("PPG03", -2.0, 0.1)]
    post = shrink(estimates)
    assert post.tau_squared == 0.0
    expected_mean = -2.0
    for p in post.per_ppg:
        assert p.shrunk_mean == pytest.approx(expected_mean, abs=1e-9)
        assert p.shrinkage_weight == pytest.approx(1.0, abs=1e-9)


def test_real_heterogeneity_produces_positive_tau2() -> None:
    """Spread of β̂ around 1.0 SE apart, with SEs of 0.1, must register as
    real heterogeneity — τ² > 0 and shrinkage < 1 per PPG."""
    estimates = [
        ("PPG01", -1.0, 0.1),
        ("PPG02", -2.0, 0.1),
        ("PPG03", -3.0, 0.1),
        ("PPG04", -4.0, 0.1),
    ]
    post = shrink(estimates)
    assert post.tau_squared > 0
    for p in post.per_ppg:
        assert 0.0 < p.shrinkage_weight < 1.0


def test_noisy_ppg_shrinks_more_than_precise_one() -> None:
    """Two PPGs with very different SEs: the noisy one gets pulled toward
    the population mean far more than the tight one."""
    estimates = [
        ("PPG_TIGHT", -2.0, 0.05),
        ("PPG_TIGHT2", -2.2, 0.05),
        ("PPG_LOOSE", -0.5, 5.0),  # very imprecise
    ]
    post = shrink(estimates)
    tight = next(p for p in post.per_ppg if p.ppg_id == "PPG_TIGHT")
    loose = next(p for p in post.per_ppg if p.ppg_id == "PPG_LOOSE")
    # Both shrunk means lie between the point estimate and the population mean.
    assert tight.shrinkage_weight < loose.shrinkage_weight
    # Loose PPG basically returns the population mean.
    assert abs(loose.shrunk_mean - post.population_mean) < 0.1


def test_posterior_mean_lies_between_point_and_population() -> None:
    """β̃ᵢ is a convex combination of β̂ᵢ and the population μ̂."""
    estimates = [
        ("PPG01", -1.0, 0.3),
        ("PPG02", -2.5, 0.4),
        ("PPG03", -1.8, 0.2),
    ]
    post = shrink(estimates)
    mu = post.population_mean
    for p in post.per_ppg:
        lo, hi = sorted([p.point, mu])
        assert lo - 1e-9 <= p.shrunk_mean <= hi + 1e-9


def test_ci_is_symmetric_and_uses_two_sided_z() -> None:
    estimates = [
        ("PPG01", -1.5, 0.2),
        ("PPG02", -2.5, 0.2),
        ("PPG03", -3.0, 0.2),
    ]
    post = shrink(estimates)
    for p in post.per_ppg:
        midpoint = 0.5 * (p.ci_low + p.ci_high)
        half_width = 0.5 * (p.ci_high - p.ci_low)
        assert midpoint == pytest.approx(p.shrunk_mean, abs=1e-9)
        assert half_width == pytest.approx(1.96 * p.shrunk_std, abs=1e-3)


def test_drops_non_finite_and_zero_se_entries() -> None:
    estimates = [
        ("PPG_OK", -2.0, 0.3),
        ("PPG_NAN", float("nan"), 0.3),
        ("PPG_ZERO", -1.5, 0.0),
        ("PPG_NEG_SE", -1.5, -0.1),
    ]
    post = shrink(estimates)
    assert post.n_studies == 1
    assert [p.ppg_id for p in post.per_ppg] == ["PPG_OK"]


def test_empty_input_returns_nan_population() -> None:
    post = shrink([])
    assert post.n_studies == 0
    assert math.isnan(post.population_mean)
    assert post.per_ppg == []


def test_payload_round_trip_has_required_keys() -> None:
    estimates = [("PPG01", -2.0, 0.2), ("PPG02", -2.5, 0.3)]
    payload = to_payload(shrink(estimates))
    assert set(payload.keys()) >= {
        "population_mean",
        "tau_squared",
        "n_studies",
        "q_statistic",
        "per_ppg",
    }
    row = payload["per_ppg"][0]
    assert set(row.keys()) >= {
        "ppg_id",
        "point",
        "std_err",
        "shrunk_mean",
        "shrunk_std",
        "ci_low",
        "ci_high",
        "shrinkage_weight",
    }
