"""Router protocol: pick an ordered candidate model set."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.models.library.diagnostics import DataProfile
from core.models.result import ProblemType

# The legacy three-candidate ensemble, always appended as a guaranteed tail so
# the candidate list is never empty even when every preferred dep is missing.
LEGACY_TAIL: tuple[str, ...] = ("loglog_ols", "semilog_ols", "lightgbm")


@runtime_checkable
class Router(Protocol):
    def select(
        self,
        problem: ProblemType,
        profile: DataProfile,
        enabled: set[str] | None = None,
    ) -> list[str]:
        """Return registry keys, best-first, filtered to available models."""
        ...
