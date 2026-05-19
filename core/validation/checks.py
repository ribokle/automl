"""Per-PPG validation verdict from rolling-origin CV results.

Each verdict aggregates a PPG's per-fold metrics into pass / warn /
fail status against four rules:

- **sign_stability** — fraction of folds with the correct elasticity
  sign. Pass if ≥ 0.75; warn if ≥ 0.50; otherwise fail.
- **wape_mean** — mean hold-out WAPE across folds. Pass if ≤ 0.20; warn
  if ≤ 0.30; otherwise fail.
- **elasticity_cv** — coefficient of variation
  ``std(elasticity) / |mean(elasticity)|`` across folds. Pass if ≤ 0.4;
  warn if ≤ 0.7; otherwise fail. Skipped when only one fold is
  available.
- **magnitude_band** — |mean(elasticity)| inside the plausibility band
  ``[0.3, 6.0]``. Outside the band downgrades to warn (or keeps fail if
  another rule already failed).

Verdict aggregation follows the worst-status precedence: fail > warn >
pass. ``severity`` is set per check so the UI can colour-code the row.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, stdev


SIGN_PASS = 0.75
SIGN_WARN = 0.50
WAPE_PASS = 0.20
WAPE_WARN = 0.30
CV_PASS = 0.4
CV_WARN = 0.7
ELASTICITY_LOW = 0.3
ELASTICITY_HIGH = 6.0


@dataclass
class Verdict:
    ppg_id: str
    verdict: str  # pass | warn | fail
    sign_stability: float
    wape_mean: float
    wape_std: float
    elasticity_mean: float
    elasticity_std: float
    elasticity_cv: float
    n_folds: int
    checks: list[dict] = field(default_factory=list)


def _worst(current: str, candidate: str) -> str:
    order = {"pass": 0, "warn": 1, "fail": 2}
    return current if order[current] >= order[candidate] else candidate


def evaluate_ppg(
    ppg_id: str, folds: list[dict], expected_sign: int = -1
) -> Verdict:
    """Aggregate fold metrics into a per-PPG verdict.

    ``expected_sign`` defaults to -1 (negative own-price elasticity). A
    fold counts as sign-correct if ``own_elasticity`` matches the
    expected sign — we don't trust the fitter's ``sign_ok`` flag alone
    because a wrong-sign retry could have flipped the model family
    mid-fold.
    """
    if not folds:
        return Verdict(
            ppg_id=ppg_id,
            verdict="fail",
            sign_stability=float("nan"),
            wape_mean=float("nan"),
            wape_std=float("nan"),
            elasticity_mean=float("nan"),
            elasticity_std=float("nan"),
            elasticity_cv=float("nan"),
            n_folds=0,
            checks=[{"name": "rolling_cv", "status": "fail", "detail": "no folds produced"}],
        )

    elasticities = [float(f["own_elasticity"]) for f in folds]
    wapes = [float(f["test_wape"]) for f in folds if math.isfinite(float(f["test_wape"]))]
    sign_hits = sum(
        1 for e in elasticities if (e * expected_sign) > 0
    )

    sign_stability = sign_hits / len(folds)
    e_mean = mean(elasticities)
    e_std = stdev(elasticities) if len(elasticities) > 1 else 0.0
    e_cv = (e_std / abs(e_mean)) if e_mean != 0 else float("inf")
    w_mean = mean(wapes) if wapes else float("nan")
    w_std = stdev(wapes) if len(wapes) > 1 else 0.0

    checks: list[dict] = []
    verdict = "pass"

    if sign_stability >= SIGN_PASS:
        checks.append({"name": "sign_stability", "status": "pass", "detail": f"{sign_stability:.0%}"})
    elif sign_stability >= SIGN_WARN:
        checks.append({"name": "sign_stability", "status": "warn", "detail": f"{sign_stability:.0%}"})
        verdict = _worst(verdict, "warn")
    else:
        checks.append({"name": "sign_stability", "status": "fail", "detail": f"{sign_stability:.0%}"})
        verdict = _worst(verdict, "fail")

    if math.isnan(w_mean):
        checks.append({"name": "wape_mean", "status": "info", "detail": "no hold-out WAPE recorded"})
    elif w_mean <= WAPE_PASS:
        checks.append({"name": "wape_mean", "status": "pass", "detail": f"{w_mean:.3f}"})
    elif w_mean <= WAPE_WARN:
        checks.append({"name": "wape_mean", "status": "warn", "detail": f"{w_mean:.3f}"})
        verdict = _worst(verdict, "warn")
    else:
        checks.append({"name": "wape_mean", "status": "fail", "detail": f"{w_mean:.3f}"})
        verdict = _worst(verdict, "fail")

    if len(folds) >= 2:
        if e_cv <= CV_PASS:
            checks.append({"name": "elasticity_cv", "status": "pass", "detail": f"CV={e_cv:.2f}"})
        elif e_cv <= CV_WARN:
            checks.append({"name": "elasticity_cv", "status": "warn", "detail": f"CV={e_cv:.2f}"})
            verdict = _worst(verdict, "warn")
        else:
            checks.append({"name": "elasticity_cv", "status": "fail", "detail": f"CV={e_cv:.2f}"})
            verdict = _worst(verdict, "fail")

    abs_e_mean = abs(e_mean)
    if ELASTICITY_LOW <= abs_e_mean <= ELASTICITY_HIGH:
        checks.append({"name": "magnitude_band", "status": "pass", "detail": f"|ε|={abs_e_mean:.2f}"})
    else:
        checks.append(
            {
                "name": "magnitude_band",
                "status": "warn",
                "detail": f"|ε|={abs_e_mean:.2f} outside [{ELASTICITY_LOW},{ELASTICITY_HIGH}]",
            }
        )
        verdict = _worst(verdict, "warn")

    return Verdict(
        ppg_id=ppg_id,
        verdict=verdict,
        sign_stability=sign_stability,
        wape_mean=w_mean,
        wape_std=w_std,
        elasticity_mean=e_mean,
        elasticity_std=e_std,
        elasticity_cv=e_cv,
        n_folds=len(folds),
        checks=checks,
    )
