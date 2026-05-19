"""CSV upload validation, sanitization, and size cap."""
from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app


def _client(tmp_path: Path) -> TestClient:
    os.environ["RUN_DIR"] = str(tmp_path / "runs")
    os.environ.pop("API_AUTH_TOKEN", None)
    return TestClient(app)


def test_csv_upload_writes_to_data_dir(tmp_path: Path) -> None:
    client = _client(tmp_path)
    body = b"sku,price\nA,1.99\n"
    res = client.post(
        "/uploads",
        files={"file": ("panel.csv", io.BytesIO(body), "text/csv")},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["bytes"] == len(body)
    assert payload["filename"] == "panel.csv"
    written = Path(payload["path"])
    assert written.exists()
    assert written.read_bytes() == body


def test_filename_is_sanitized(tmp_path: Path) -> None:
    client = _client(tmp_path)
    res = client.post(
        "/uploads",
        files={"file": ("../../etc/passwd.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
    )
    assert res.status_code == 200
    written = Path(res.json()["path"])
    assert written.name == "passwd.csv"
    assert ".." not in written.parts


def test_non_csv_extension_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path)
    res = client.post(
        "/uploads",
        files={"file": ("evil.sh", io.BytesIO(b"#!/bin/sh\nrm -rf /\n"), "text/x-shellscript")},
    )
    assert res.status_code == 415


def test_size_cap_enforced(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    client = _client(tmp_path)
    big = b"x" * (2 * 1024 * 1024)
    res = client.post(
        "/uploads",
        files={"file": ("big.csv", io.BytesIO(big), "text/csv")},
    )
    assert res.status_code == 413
    upload_dir = tmp_path / "data"
    assert not any(upload_dir.glob("big*.csv")) if upload_dir.exists() else True


def test_empty_upload_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path)
    res = client.post(
        "/uploads",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert res.status_code == 400
