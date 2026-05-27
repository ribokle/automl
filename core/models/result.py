"""Generalized model-result contract for the model library.

``ElasticityFit`` (``core.models.base``) stays the narrow contract every
downstream stage already understands. ``ModelResult`` is the superset every
library plugin returns: it can carry a scalar own-price elasticity, a
cross-price matrix, and/or a forecast. ``to_elasticity_fit`` narrows a
``ModelResult`` back to an ``ElasticityFit`` so the existing modelling agent +
downstream (decomposition / simulation / optimisation / validation) keep
working unchanged, returning ``None`` when no scalar elasticity is available
(pure forecasters, demand systems without a usable diagonal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, StrEnum, auto
from typing import Any

from core.models.base import ElasticityFit


class ProblemType(StrEnum):
    """The business question a model is being asked to answer."""

    OWN_ELASTICITY = "own_elasticity"
    CROSS_PRICE = "cross_price"
    FORECAST = "forecast"
    PROMO_UPLIFT = "promo_uplift"
    DEMAND_SYSTEM = "demand_system"
    PANEL = "panel"


class Capability(Flag):
    """What a plugin can produce / requires. Used by the router to filter."""

    NONE = 0
    SCALAR_ELASTICITY = auto()
    CROSS_PRICE_MATRIX = auto()
    FORECAST = auto()
    PREDICT_LOG_UNITS = auto()
    NEEDS_PANEL = auto()
    NEEDS_TIME_INDEX = auto()
    HEAVY_DEP = auto()


@dataclass
class ForecastBlock:
    horizon: int
    index: list[str] = field(default_factory=list)
    mean: list[float] = field(default_factory=list)
    lower: list[float] | None = None
    upper: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "index": list(self.index),
            "mean": [float(v) for v in self.mean],
            "lower": None if self.lower is None else [float(v) for v in self.lower],
            "upper": None if self.upper is None else [float(v) for v in self.upper],
        }


@dataclass
class ModelResult:
    ppg_id: str
    model: str
    problem_type: ProblemType = ProblemType.OWN_ELASTICITY
    own_elasticity: float | None = None
    std_err: float | None = None
    p_value: float | None = None
    cross_price: dict[str, dict[str, float]] | None = None
    forecast: ForecastBlock | None = None
    r_squared: float | None = None
    n_obs: int = 0
    controls: list[str] = field(default_factory=list)
    coefficients: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    capabilities: Capability = Capability.NONE

    @property
    def sign_ok(self) -> bool:
        return self.own_elasticity is not None and self.own_elasticity < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ppg_id": self.ppg_id,
            "model": self.model,
            "problem_type": self.problem_type.value,
            "own_elasticity": None if self.own_elasticity is None else float(self.own_elasticity),
            "std_err": None if self.std_err is None else float(self.std_err),
            "p_value": None if self.p_value is None else float(self.p_value),
            "cross_price": self.cross_price,
            "forecast": None if self.forecast is None else self.forecast.to_dict(),
            "r_squared": None if self.r_squared is None else float(self.r_squared),
            "n_obs": int(self.n_obs),
            "controls": list(self.controls),
            "coefficients": {k: float(v) for k, v in self.coefficients.items()},
            "diagnostics": self.diagnostics,
            "capabilities": int(self.capabilities.value),
        }


def from_elasticity_fit(
    fit: ElasticityFit,
    *,
    problem_type: ProblemType = ProblemType.OWN_ELASTICITY,
    capabilities: Capability = Capability.SCALAR_ELASTICITY | Capability.PREDICT_LOG_UNITS,
) -> ModelResult:
    """Lift a legacy ``ElasticityFit`` into a ``ModelResult`` (for plugin wrappers)."""
    return ModelResult(
        ppg_id=fit.ppg_id,
        model=fit.model,
        problem_type=problem_type,
        own_elasticity=fit.own_elasticity,
        std_err=fit.std_err,
        p_value=fit.p_value,
        r_squared=fit.r_squared,
        n_obs=fit.n_obs,
        controls=list(fit.controls),
        coefficients=dict(fit.coefficients),
        diagnostics=dict(fit.diagnostics),
        capabilities=capabilities,
    )


def to_elasticity_fit(result: ModelResult) -> ElasticityFit | None:
    """Narrow a ``ModelResult`` back to an ``ElasticityFit``.

    Returns ``None`` when no scalar own-price elasticity is available, so the
    caller can record a ``skip_reason`` row with the same shape the modelling
    agent already writes for skipped cells. For cross-price models the matrix
    diagonal becomes ``own_elasticity`` and the full matrix is preserved in
    ``diagnostics['cross_price_matrix']``.
    """
    own = result.own_elasticity
    diagnostics = dict(result.diagnostics)
    if own is None and result.cross_price:
        own = result.cross_price.get(result.ppg_id, {}).get(result.ppg_id)
        diagnostics["cross_price_matrix"] = result.cross_price
    if own is None:
        return None

    def _f(v: float | None) -> float:
        return float(v) if v is not None else float("nan")

    return ElasticityFit(
        ppg_id=result.ppg_id,
        model=result.model,
        own_elasticity=float(own),
        std_err=_f(result.std_err),
        p_value=_f(result.p_value),
        r_squared=_f(result.r_squared),
        n_obs=int(result.n_obs),
        controls=list(result.controls),
        coefficients=dict(result.coefficients),
        diagnostics=diagnostics,
    )
