"""Plugin contract for the model library.

Every model is an independent plugin: a small class that declares its
``key``, ``family``, the ``ProblemType``s it answers, its ``Capability``
flags, and any third-party ``required_packages``. Plugins implement
``fit(frame, ctx) -> ModelResult``. Heavy dependencies MUST be imported inside
``fit`` / probed in ``is_available`` — never at module import time — so that
importing the library never drags in torch/pymc/pyblp.

A plugin module may import only this module, ``core.models.result``, and shared
non-model utilities (``core.models.metrics``, ``core.models.shap_attribution``,
``core.models.library._*`` helpers). It must NOT import a sibling model module;
``tests/unit/test_library_no_cross_import.py`` enforces this.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from core.models.result import Capability, ModelResult, ProblemType


@dataclass(frozen=True)
class FitContext:
    """Everything a plugin needs beyond the data frame itself."""

    ppg_id: str
    controls: list[str] = field(default_factory=list)
    test: pd.DataFrame | None = None
    grain: str = "ppg_week"
    problem_type: ProblemType = ProblemType.OWN_ELASTICITY
    hparams: dict[str, Any] = field(default_factory=dict)
    rng_seed: int = 0


@runtime_checkable
class ModelPlugin(Protocol):
    key: str
    family: str
    problem_types: frozenset[ProblemType]
    capabilities: Capability
    required_packages: tuple[str, ...]

    def is_available(self) -> bool: ...
    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult: ...
    def default_hparams(self) -> dict[str, Any]: ...


class BaseModelPlugin:
    """Convenience base: dep probing + empty defaults. Subclasses set the
    class attributes and implement ``fit``."""

    key: str = ""
    family: str = ""
    problem_types: frozenset[ProblemType] = frozenset({ProblemType.OWN_ELASTICITY})
    capabilities: Capability = Capability.SCALAR_ELASTICITY
    required_packages: tuple[str, ...] = ()

    def is_available(self) -> bool:
        return all(
            importlib.util.find_spec(pkg) is not None for pkg in self.required_packages
        )

    def default_hparams(self) -> dict[str, Any]:
        return {}

    def resolve_hparams(self, ctx: FitContext) -> dict[str, Any]:
        merged = dict(self.default_hparams())
        merged.update(ctx.hparams or {})
        return merged

    def fit(self, frame: pd.DataFrame, ctx: FitContext) -> ModelResult:  # noqa: D401
        raise NotImplementedError
