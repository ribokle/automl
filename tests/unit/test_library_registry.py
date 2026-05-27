"""Registry registration + availability filtering."""
from __future__ import annotations

import core.models.library as lib
from core.models.library import registry
from core.models.library.base import BaseModelPlugin


def test_light_families_register() -> None:
    keys = set(registry.all_keys())
    for expected in {"loglog_ols", "semilog_ols", "lightgbm", "ridge", "lasso", "elasticnet"}:
        assert expected in keys, f"{expected} not registered"


def test_get_returns_plugin_instance() -> None:
    plugin = registry.get("ridge")
    assert isinstance(plugin, BaseModelPlugin)
    assert plugin.key == "ridge"
    assert plugin.family == "regularized"


def test_available_keys_nonempty_with_light_deps() -> None:
    assert registry.available_keys()


def test_available_respects_enabled_and_disabled() -> None:
    only_ridge = registry.available_keys(enabled={"ridge"})
    assert only_ridge == {"ridge"}
    without_ridge = registry.available_keys(disabled={"ridge"})
    assert "ridge" not in without_ridge
    assert "loglog_ols" in without_ridge


def test_missing_dependency_drops_plugin(monkeypatch) -> None:
    plugin = registry.get("ridge")
    monkeypatch.setattr(plugin, "is_available", lambda: False)
    assert "ridge" not in registry.available_keys()


def test_library_module_exposes_registry() -> None:
    assert lib.registry is registry
