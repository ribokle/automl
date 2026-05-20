"""Dependency providers for FastAPI routes."""
from __future__ import annotations

from pathlib import Path

from core.config import get_settings


def get_run_dir() -> Path:
    base = get_settings().run_dir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base
