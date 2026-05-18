"""Smoke tests for the artifact endpoint.

The path-traversal check used to do `str(target).startswith(str(base) + "/")`,
which only worked on POSIX. On Windows, `str(target)` uses backslashes while
the appended literal slash made every legitimate path look like a traversal
attempt and return 400. The fixed version uses `Path.relative_to`.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


def _make_client(tmp_path: Path) -> TestClient:
    os.environ["RUN_DIR"] = str(tmp_path)
    return TestClient(app)


def test_happy_path_returns_200(tmp_path: Path):
    run = tmp_path / "abc"
    run.mkdir()
    (run / "coverage_grid.json").write_text('{"ok": true}')
    client = _make_client(tmp_path)

    res = client.get("/artifacts/abc/coverage_grid.json")
    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test_missing_file_returns_404(tmp_path: Path):
    (tmp_path / "abc").mkdir()
    client = _make_client(tmp_path)

    res = client.get("/artifacts/abc/missing.json")
    assert res.status_code == 404


def test_path_traversal_returns_400(tmp_path: Path):
    (tmp_path / "abc").mkdir()
    (tmp_path.parent / "secret.txt").write_text("nope")
    client = _make_client(tmp_path)

    res = client.get("/artifacts/abc/..%2F..%2Fsecret.txt")
    assert res.status_code == 400
