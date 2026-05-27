"""Invariant: no library model module imports a sibling model module.

Model modules may import only the library scaffolding (base/registry/
diagnostics), shared underscore-prefixed helpers, the result contract, and
third-party / legacy ``core.models`` fitters — never another
``core.models.library.<family>.<module>``.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

LIBRARY_ROOT = Path(__file__).resolve().parents[2] / "core" / "models" / "library"
_SIBLING = re.compile(r"^core\.models\.library\.(?!_)([a-z_]+)\.([a-z_]+)$")
_ALLOWED_SUBMODULES = {"base", "registry", "diagnostics"}


def _model_modules() -> list[Path]:
    out: list[Path] = []
    for path in LIBRARY_ROOT.rglob("*.py"):
        name = path.name
        if name == "__init__.py" or name.startswith("_"):
            continue
        if path.parent == LIBRARY_ROOT and path.stem in _ALLOWED_SUBMODULES:
            continue
        out.append(path)
    return out


def _imported_modules(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    return mods


def test_no_model_imports_a_sibling_model() -> None:
    offenders: list[str] = []
    for path in _model_modules():
        tree = ast.parse(path.read_text())
        for module in _imported_modules(tree):
            m = _SIBLING.match(module)
            if m and m.group(2) not in _ALLOWED_SUBMODULES:
                offenders.append(f"{path.relative_to(LIBRARY_ROOT)} imports {module}")
    assert not offenders, "model modules must not import siblings:\n" + "\n".join(offenders)
