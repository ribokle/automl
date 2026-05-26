"""Model library: a registry of independent, configurable model plugins.

Importing this package registers every available model via the per-family
subpackages' import side-effects. Light families (no heavy third-party deps)
are imported eagerly. Heavier families are imported defensively so a missing
optional dependency never breaks ``import core.models.library`` — the registry
simply won't contain those keys, and the router's availability filter handles
the rest.
"""
from __future__ import annotations

from core.models.library import registry

# Light families — always importable (deps are in the base install). Optional
# third-party deps inside these modules are imported lazily in ``fit``, so
# registering them never triggers a heavy import.
from core.models.library import (  # noqa: F401,E402
    classical,
    ml_nonparam,
    regularized,
    robust_quantile,
    trees,
)

# Heavier families register defensively; absence of an optional dep must not
# break importing the library as a whole.
for _family in (
    "panel",
    "causal",
    "discrete_choice",
    "bayesian",
    "timeseries",
    "deep",
    "promo",
):
    try:  # noqa: SIM105
        __import__(f"core.models.library.{_family}")
    except ImportError:
        pass

__all__ = ["registry"]
