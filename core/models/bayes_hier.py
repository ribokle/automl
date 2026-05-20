"""Empirical-Bayes hierarchical shrinkage across PPGs.

The per-PPG OLS / semi-log fits give point estimates ``β̂ᵢ`` with standard
errors ``sᵢ``. We treat the cross-PPG distribution of true elasticities as
``βᵢ ~ Normal(μ, τ²)`` with ``β̂ᵢ | βᵢ ~ Normal(βᵢ, sᵢ²)`` (the usual
random-effects meta-analysis model). With ``μ`` and ``τ²`` estimated from
the data, each PPG's posterior is

    β̃ᵢ = (β̂ᵢ/sᵢ² + μ̂/τ²) / (1/sᵢ² + 1/τ²)
    Var(βᵢ | β̂ᵢ) = 1 / (1/sᵢ² + 1/τ²)

so noisy PPGs get pulled toward the population mean and tight PPGs barely
move — the empirical-Bayes / Stein shrinkage form. ``μ̂`` is the
inverse-variance-weighted mean (the fixed-effect estimator), ``τ²`` comes
from the DerSimonian-Laird method-of-moments. No MCMC, fully deterministic.

When ``τ²`` collapses to zero (no detectable heterogeneity), every PPG is
shrunk all the way to ``μ̂`` (complete pooling). When ``τ²`` is large
relative to sampling variance, shrinkage vanishes and each PPG retains its
own estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


Z_95 = 1.959963984540054  # two-sided 95% under Normal


@dataclass
class PosteriorEstimate:
    ppg_id: str
    point: float
    std_err: float
    shrunk_mean: float
    shrunk_std: float
    ci_low: float
    ci_high: float
    shrinkage_weight: float  # posterior weight on the prior (μ̂); 0 = no shrinkage, 1 = full shrinkage


@dataclass
class HierarchicalPosterior:
    population_mean: float
    tau_squared: float
    q_statistic: float
    n_studies: int
    per_ppg: list[PosteriorEstimate]


def _fixed_effect_mean(betas: np.ndarray, ses: np.ndarray) -> tuple[float, np.ndarray]:
    """Inverse-variance weighted mean and the weights."""
    w = 1.0 / np.square(ses)
    mu_fe = float(np.sum(w * betas) / np.sum(w))
    return mu_fe, w


def _dersimonian_laird_tau2(
    betas: np.ndarray, ses: np.ndarray, mu_fe: float, w_fe: np.ndarray
) -> tuple[float, float]:
    """DerSimonian-Laird method-of-moments τ² estimator.

    Returns ``(tau2, Q)`` where ``Q`` is the heterogeneity statistic.
    Clamped at 0 when sampling variance alone explains the spread.
    """
    k = len(betas)
    if k < 2:
        return 0.0, 0.0
    q = float(np.sum(w_fe * np.square(betas - mu_fe)))
    sum_w = float(np.sum(w_fe))
    sum_w2 = float(np.sum(np.square(w_fe)))
    denom = sum_w - sum_w2 / sum_w
    if denom <= 0:
        return 0.0, q
    tau2 = max(0.0, (q - (k - 1)) / denom)
    return tau2, q


def shrink(
    estimates: list[tuple[str, float, float]],
) -> HierarchicalPosterior:
    """Empirical-Bayes shrinkage of per-PPG ``(ppg_id, β̂, sᵢ)`` triples.

    PPGs with ``sᵢ <= 0`` or non-finite estimates are dropped from the
    pooled fit — they carry no information for τ² estimation.
    """
    cleaned = [
        (ppg, float(b), float(s))
        for ppg, b, s in estimates
        if math.isfinite(b) and math.isfinite(s) and s > 0
    ]
    if not cleaned:
        return HierarchicalPosterior(
            population_mean=float("nan"),
            tau_squared=0.0,
            q_statistic=0.0,
            n_studies=0,
            per_ppg=[],
        )

    ppgs = [c[0] for c in cleaned]
    betas = np.array([c[1] for c in cleaned], dtype=float)
    ses = np.array([c[2] for c in cleaned], dtype=float)

    mu_fe, w_fe = _fixed_effect_mean(betas, ses)
    tau2, q = _dersimonian_laird_tau2(betas, ses, mu_fe, w_fe)

    # Random-effects population mean: recompute with τ² inflating the weights.
    w_re = 1.0 / (np.square(ses) + tau2)
    mu_re = float(np.sum(w_re * betas) / np.sum(w_re))

    posts: list[PosteriorEstimate] = []
    for ppg, beta, se in zip(ppgs, betas, ses):
        s2 = se * se
        if tau2 > 0:
            # Inverse-variance combination of likelihood (β̂, s²) and prior (μ_RE, τ²).
            post_prec = 1.0 / s2 + 1.0 / tau2
            post_var = 1.0 / post_prec
            post_mean = (beta / s2 + mu_re / tau2) * post_var
            shrink_w = (1.0 / tau2) / post_prec  # weight on the prior
        else:
            # Complete pooling: every PPG collapses onto μ̂_RE.
            post_var = 1.0 / np.sum(1.0 / np.square(ses))
            post_mean = mu_re
            shrink_w = 1.0
        post_sd = math.sqrt(post_var)
        posts.append(
            PosteriorEstimate(
                ppg_id=ppg,
                point=float(beta),
                std_err=float(se),
                shrunk_mean=float(post_mean),
                shrunk_std=float(post_sd),
                ci_low=float(post_mean - Z_95 * post_sd),
                ci_high=float(post_mean + Z_95 * post_sd),
                shrinkage_weight=float(shrink_w),
            )
        )

    return HierarchicalPosterior(
        population_mean=float(mu_re),
        tau_squared=float(tau2),
        q_statistic=float(q),
        n_studies=len(cleaned),
        per_ppg=posts,
    )


def to_payload(post: HierarchicalPosterior) -> dict:
    """JSON-serialisable payload for the artifact + UI."""
    return {
        "population_mean": post.population_mean,
        "tau_squared": post.tau_squared,
        "q_statistic": post.q_statistic,
        "n_studies": post.n_studies,
        "per_ppg": [
            {
                "ppg_id": p.ppg_id,
                "point": p.point,
                "std_err": p.std_err,
                "shrunk_mean": p.shrunk_mean,
                "shrunk_std": p.shrunk_std,
                "ci_low": p.ci_low,
                "ci_high": p.ci_high,
                "shrinkage_weight": p.shrinkage_weight,
            }
            for p in post.per_ppg
        ],
    }
