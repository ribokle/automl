"""Decorator-based plugin registry.

``@register`` populates a process-global map at import time. The library's
``__init__`` imports every family subpackage so registration side-effects run.
``available`` filters by an optional allow/deny list AND by ``is_available()``
so a model whose third-party dependency is missing is dropped silently — this
is what keeps the pipeline runnable when heavy extras aren't installed.
"""
from __future__ import annotations

from core.models.library.base import BaseModelPlugin

_REGISTRY: dict[str, BaseModelPlugin] = {}


def register(cls: type[BaseModelPlugin]) -> type[BaseModelPlugin]:
    inst = cls()
    if not inst.key:
        raise ValueError(f"{cls.__name__} must set a non-empty `key`")
    if inst.key in _REGISTRY:
        raise ValueError(f"duplicate model key: {inst.key!r}")
    _REGISTRY[inst.key] = inst
    return cls


def get(key: str) -> BaseModelPlugin:
    return _REGISTRY[key]


def has(key: str) -> bool:
    return key in _REGISTRY


def all_keys() -> list[str]:
    return sorted(_REGISTRY)


def all_plugins() -> list[BaseModelPlugin]:
    return list(_REGISTRY.values())


def available(
    keys: list[str] | None = None,
    *,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
) -> list[BaseModelPlugin]:
    selected = keys if keys is not None else all_keys()
    out: list[BaseModelPlugin] = []
    for key in selected:
        plugin = _REGISTRY.get(key)
        if plugin is None:
            continue
        if enabled and key not in enabled:
            continue
        if disabled and key in disabled:
            continue
        if not plugin.is_available():
            continue
        out.append(plugin)
    return out


def available_keys(
    *, enabled: set[str] | None = None, disabled: set[str] | None = None
) -> set[str]:
    return {p.key for p in available(enabled=enabled, disabled=disabled)}
