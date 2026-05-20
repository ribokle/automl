"""Shared types for elasticity model fits.

Each fitter (log-log, semi-log, ...) returns the same ``ElasticityFit`` so
the modelling agent can compare them on equal footing and pick a winner per
PPG. The own-price elasticity is always reported in elastic units (i.e.
%Δunits per %Δprice), regardless of model form, so the sign-retry loop can
make a uniform decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ElasticityFit:
    ppg_id: str
    model: str
    own_elasticity: float
    std_err: float
    p_value: float
    r_squared: float
    n_obs: int
    controls: list[str] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def sign_ok(self) -> bool:
        return self.own_elasticity < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ppg_id": self.ppg_id,
            "model": self.model,
            "own_elasticity": float(self.own_elasticity),
            "std_err": float(self.std_err),
            "p_value": float(self.p_value),
            "r_squared": float(self.r_squared),
            "n_obs": int(self.n_obs),
            "controls": list(self.controls),
            "coefficients": {k: float(v) for k, v in self.coefficients.items()},
            "diagnostics": self.diagnostics,
            "sign_ok": self.sign_ok,
        }
