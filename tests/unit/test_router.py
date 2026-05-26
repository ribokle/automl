"""Deterministic + LLM router selection."""
from __future__ import annotations

import json

import core.models.library  # noqa: F401 — ensure plugins are registered
from core.config import RouterSettings
from core.models.library.diagnostics import DataProfile
from core.models.router.llm_router import LLMRouter
from core.models.router.rules import DeterministicRouter
from core.models.result import ProblemType


def _profile(**overrides) -> DataProfile:
    base = dict(
        n_obs=200,
        n_train=160,
        n_test=40,
        log_price_std=0.2,
        price_cv=0.2,
        is_panel=False,
        n_entities=1,
        has_cross_price_cols=False,
        time_length=200,
        has_regular_time_index=True,
        seasonality_detected=False,
        n_seasons=0,
        target_type="continuous_log_units",
        n_controls=3,
    )
    base.update(overrides)
    return DataProfile(**base)


def test_small_n_prefers_ridge_over_loglog() -> None:
    router = DeterministicRouter(RouterSettings(small_n_threshold=60))
    out = router.select(ProblemType.OWN_ELASTICITY, _profile(n_obs=30))
    assert out[0] == "ridge"
    assert "loglog_ols" in out
    assert out.index("ridge") < out.index("loglog_ols")
    assert "huber" in out  # robust fitter is registered and available


def test_large_n_prefers_loglog_first() -> None:
    router = DeterministicRouter(RouterSettings(small_n_threshold=60))
    out = router.select(ProblemType.OWN_ELASTICITY, _profile(n_obs=500))
    assert out[0] == "loglog_ols"


def test_forecast_prefers_time_series_and_keeps_tail() -> None:
    router = DeterministicRouter()
    out = router.select(ProblemType.FORECAST, _profile(seasonality_detected=True))
    assert out[0] == "sarimax"  # statsmodels TS models are available
    assert "ets" in out
    assert "loglog_ols" in out  # legacy tail always present


def test_selection_is_deterministic() -> None:
    router = DeterministicRouter()
    p = _profile(n_obs=42)
    assert router.select(ProblemType.OWN_ELASTICITY, p) == router.select(
        ProblemType.OWN_ELASTICITY, p
    )


def test_enabled_filter_restricts_candidates() -> None:
    router = DeterministicRouter()
    out = router.select(ProblemType.OWN_ELASTICITY, _profile(), enabled={"loglog_ols"})
    assert out == ["loglog_ols"]


def test_llm_router_dry_run_falls_back_to_rules() -> None:
    router = LLMRouter()
    p = _profile(n_obs=30)
    rules_out = DeterministicRouter().select(ProblemType.OWN_ELASTICITY, p)
    assert router.select(ProblemType.OWN_ELASTICITY, p, llm_text=None) == rules_out


def test_llm_router_parses_valid_json_and_appends_tail() -> None:
    router = LLMRouter()
    text = json.dumps({"candidates": ["elasticnet", "ridge"], "rationale": "x"})
    out = router.select(ProblemType.OWN_ELASTICITY, _profile(), llm_text=text)
    assert out[0] == "elasticnet"
    assert out[1] == "ridge"
    assert "loglog_ols" in out  # tail appended


def test_llm_router_invalid_json_falls_back() -> None:
    router = LLMRouter()
    p = _profile()
    out = router.select(ProblemType.OWN_ELASTICITY, p, llm_text="not json")
    assert out == DeterministicRouter().select(ProblemType.OWN_ELASTICITY, p)


def test_llm_router_unknown_keys_fall_back() -> None:
    router = LLMRouter()
    p = _profile()
    text = json.dumps({"candidates": ["nonexistent_model"]})
    out = router.select(ProblemType.OWN_ELASTICITY, p, llm_text=text)
    assert out == DeterministicRouter().select(ProblemType.OWN_ELASTICITY, p)
