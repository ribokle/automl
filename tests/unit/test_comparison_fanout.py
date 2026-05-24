"""Unit tests for the comparison-grain fan-out helpers.

Covers:
- ``_agents_for_depth``: maps the operator's depth picker to a concrete
  agent list, respecting data-dependency ordering.
- ``_snapshot_mtimes``: captures the file baseline used by the
  comparison loop to detect which files each grain's pass produced.
"""
from __future__ import annotations

import time
from pathlib import Path

from core.orchestrator.runner import (
    _COMPARISON_DOWNSTREAM_AGENTS,
    _agents_for_depth,
    _snapshot_mtimes,
)


def test_agents_for_depth_none_returns_full_tail() -> None:
    assert _agents_for_depth(None) == _COMPARISON_DOWNSTREAM_AGENTS


def test_agents_for_depth_empty_returns_empty() -> None:
    assert _agents_for_depth([]) == ()


def test_agents_for_depth_modeling_only() -> None:
    out = _agents_for_depth(["modeling"])
    assert out == ("feature_engineering", "feature_refine", "modeling")


def test_agents_for_depth_through_validation() -> None:
    out = _agents_for_depth(["modeling", "validation"])
    # Must include every prereq up to and including validation;
    # insights is below validation in the tail and stays out.
    assert "validation" in out
    assert "insights" not in out
    assert "feature_engineering" in out
    assert "decomposition" in out


def test_agents_for_depth_picks_deepest_stage() -> None:
    # Operator picks the LAST stage explicitly; resolver should include
    # every prereq even if they weren't ticked.
    out = _agents_for_depth(["insights"])
    assert out == _COMPARISON_DOWNSTREAM_AGENTS


def test_snapshot_mtimes_captures_json_artifacts_only(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.csv").write_text("x")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "c.json").write_text("{}")

    snap = _snapshot_mtimes(tmp_path)
    # Only top-level JSON artifacts; non-JSON files (b.csv) and
    # subdirectories are skipped.
    assert set(snap.keys()) == {"a.json"}
    for name, mtime in snap.items():
        assert isinstance(mtime, float)


def test_snapshot_mtimes_skips_bookkeeping_and_warehouse(tmp_path: Path) -> None:
    # These live in run_dir but must never be snapshotted / renamed:
    # state.json is run bookkeeping; the warehouse is locked on Windows.
    (tmp_path / "state.json").write_text("{}")
    (tmp_path / "events.jsonl").write_text("")
    (tmp_path / "warehouse.duckdb").write_text("")
    (tmp_path / "modeling_results.json").write_text("{}")

    snap = _snapshot_mtimes(tmp_path)
    assert set(snap.keys()) == {"modeling_results.json"}


def test_snapshot_mtimes_reflects_writes(tmp_path: Path) -> None:
    (tmp_path / "stable.json").write_text("v1")
    snap1 = _snapshot_mtimes(tmp_path)

    # Sleep just past mtime resolution, then write a new file.
    time.sleep(0.01)
    (tmp_path / "new.json").write_text("{}")

    snap2 = _snapshot_mtimes(tmp_path)
    assert "new.json" in snap2 and "new.json" not in snap1
    # Stable file's mtime unchanged.
    assert snap2["stable.json"] == snap1["stable.json"]
