"""Settings layer regression tests.

Covers the contract:
- empty env -> defaults match the literals each module used pre-refactor;
- ``MODEL_<AGENT>`` env vars override ``model_for(agent)``;
- ``ANTHROPIC_MODEL_OPUS`` / ``ANTHROPIC_MODEL_SONNET`` propagate;
- ``ALLOWED_ORIGINS`` parses CSV strings into ``list[str]``;
- ``BASELINE_DIR`` propagates into the ingestion agent's baseline lookup.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.config import Settings, get_settings


@pytest.fixture
def clean_env(monkeypatch):
    """Strip every config-relevant env var so tests see the defaults."""
    for key in list(os.environ):
        if key.startswith(
            (
                "MODEL_",
                "ANTHROPIC_",
                "LLM_",
                "API_",
                "ALLOWED_",
                "MAX_UPLOAD_",
                "RUN_DIR",
                "BASELINE_DIR",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_match_current_behaviour(clean_env):
    s = Settings()
    assert s.anthropic_model_opus == "claude-opus-4-7"
    assert s.anthropic_model_sonnet == "claude-sonnet-4-6"
    assert s.max_upload_mb == 200
    assert s.llm_trace is True
    assert s.llm_cli_timeout_seconds == 180
    assert s.allowed_origins == ["http://localhost:3000"]
    assert s.baseline_dir == Path("core/data/baselines")
    assert s.drift_slack_pct == pytest.approx(0.4)
    v = s.validation
    assert (v.sign_pass, v.sign_warn) == (0.75, 0.50)
    assert (v.wape_pass, v.wape_warn) == (0.20, 0.30)
    assert (v.cv_pass, v.cv_warn) == (0.4, 0.7)
    assert (v.elasticity_low, v.elasticity_high) == (0.3, 6.0)


def test_per_agent_model_override(clean_env, monkeypatch):
    monkeypatch.setenv("MODEL_PPG_MAPPING", "claude-sonnet-x")
    from core.llm.routing import model_for

    assert model_for("ppg_mapping") == "claude-sonnet-x"
    # An unrelated agent still uses the role default.
    assert model_for("ingestion") == "claude-sonnet-4-6"


def test_global_opus_override(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL_OPUS", "claude-opus-future")
    from core.llm.routing import model_for

    assert model_for("modeling") == "claude-opus-future"
    assert model_for("ingestion") == "claude-sonnet-4-6"


def test_allowed_origins_parses_csv(clean_env, monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.com, https://b.com")
    s = Settings()
    assert s.allowed_origins == ["https://a.com", "https://b.com"]


def test_baseline_dir_override_propagates(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("BASELINE_DIR", str(tmp_path))
    s = Settings()
    assert s.baseline_dir == tmp_path
