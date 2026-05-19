"""CORS allow-list is configurable via ALLOWED_ORIGINS env."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient


def _reload_app() -> object:
    import importlib

    import api.main as main

    return importlib.reload(main).app


def test_default_origin_is_localhost_3000(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    os.environ["RUN_DIR"] = str(tmp_path)

    app = _reload_app()
    client = TestClient(app)
    res = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_disallowed_origin_gets_no_cors_header(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com")
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    os.environ["RUN_DIR"] = str(tmp_path)

    app = _reload_app()
    client = TestClient(app)
    res = client.get("/health", headers={"Origin": "http://evil.example.com"})
    assert res.status_code == 200
    assert "access-control-allow-origin" not in res.headers


def test_multiple_origins_supported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "ALLOWED_ORIGINS", "https://app.example.com, https://staging.example.com"
    )
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    os.environ["RUN_DIR"] = str(tmp_path)

    app = _reload_app()
    client = TestClient(app)
    res = client.get("/health", headers={"Origin": "https://staging.example.com"})
    assert res.headers.get("access-control-allow-origin") == "https://staging.example.com"
