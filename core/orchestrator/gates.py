"""Approval-gate definitions.

Gates default on `ppg_mapping`, `modeling`, and `optimization`. They can be
disabled per-run via the CLI `--no-gates` flag, in which case the runner
auto-approves.
"""
from __future__ import annotations

DEFAULT_GATES: dict[str, bool] = {
    "ppg_mapping": True,
    "modeling": True,
    "optimization": True,
}
