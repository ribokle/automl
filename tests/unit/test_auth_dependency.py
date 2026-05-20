"""Bearer-token auth applied to API routes when API_AUTH_TOKEN is set."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def env(tmp_path: Path):
    os.environ["RUN_DIR"] = str(tmp_path)
    yield
    os.environ.pop("API_AUTH_TOKEN", None)


def test_health_is_always_unauthenticated(env, monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_TOKEN", "secret123")
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200


def test_missing_token_returns_401(env, monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_TOKEN", "secret123")
    client = TestClient(app)
    res = client.get("/runs")
    assert res.status_code == 401
    assert res.headers.get("www-authenticate") == "Bearer"


def test_wrong_token_returns_401(env, monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_TOKEN", "secret123")
    client = TestClient(app)
    res = client.get("/runs", headers={"Authorization": "Bearer wrong"})
    assert res.status_code == 401


def test_correct_token_passes(env, monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_TOKEN", "secret123")
    client = TestClient(app)
    res = client.get("/runs", headers={"Authorization": "Bearer secret123"})
    assert res.status_code == 200


def test_no_token_configured_means_open(env, monkeypatch) -> None:
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)
    client = TestClient(app)
    res = client.get("/runs")
    assert res.status_code == 200


def test_uploads_route_also_enforces(env, monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_TOKEN", "secret123")
    client = TestClient(app)
    res = client.post("/uploads")
    assert res.status_code == 401
