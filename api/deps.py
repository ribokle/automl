"""Dependency providers for FastAPI routes."""
from __future__ import annotations

import os
from pathlib import Path


def get_run_dir() -> Path:
    base = Path(os.environ.get("RUN_DIR", "./runs")).resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base
