"""Rolling-origin CV + per-PPG verdict logic + the validation agent."""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.agents.validation import ValidationAgent
from core.orchestrator.state import AgentResult, AgentStatus, RunState
from core.validation.checks import evaluate_ppg
from core.validation.rolling import build_folds, fit_one_fold


def _toy_frame(n: int = 80, slope: float = -2.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_base_price = np.log(3.0) * np.ones(n)
    log_price = log_base_price + np.log(0.8 + 0.4 * rng.random(n))
    return pd.DataFrame(
        {
            "ppg_id": "PPG_V",
            "week_start": pd.date_range("2024-01-01", periods=n, freq="W").astype(str),
            "log_units": 6.5 + slope * log_price + rng.normal(0, 0.05, size=n),
            "log_price": log_price,
            "log_base_price": log_base_price,
            "tpr_share": rng.binomial(1, 0.3, size=n).astype(float),
            "log_distribution_acv": np.log(70 + 20 * rng.random(n)),
        }
    )


def test_build_folds_returns_n_folds_when_frame_long_enough() -> None:
    frame = _toy_frame(n=80)
    folds = build_folds(frame, n_folds=4, min_train_size=20)
    assert len(folds) == 4
    for f in folds[:-1]:
        assert len(f.train) >= 20
        assert len(f.test) >= 1
    assert folds[-1].train.index[-1] + 1 + len(folds[-1].test) == len(frame)


def test_build_folds_empty_when_frame_too_short() -> None:
    frame = _toy_frame(n=10)
    folds = build_folds(frame, n_folds=4, min_train_size=20)
    assert folds == []


def test_build_folds_train_strictly_precedes_test() -> None:
    frame = _toy_frame(n=80).sort_values("week_start").reset_index(drop=True)
    folds = build_folds(frame, n_folds=3, min_train_size=20)
    for f in folds:
        train_max = f.train["week_start"].max()
        test_min = f.test["week_start"].min()
        assert train_max < test_min


def test_fit_one_fold_recovers_correct_sign_on_clean_dgp() -> None:
    frame = _toy_frame(n=80, slope=-2.0)
    folds = build_folds(frame, n_folds=3, min_train_size=30)
    for f in folds:
        out = fit_one_fold(
            "PPG_V", f, controls=["tpr_share", "log_distribution_acv"], model_kind="loglog_ols"
        )
        assert out["own_elasticity"] < 0
        assert out["sign_ok"] is True


def test_evaluate_pass_when_stable_and_low_wape() -> None:
    folds = [
        {"own_elasticity": -2.0, "test_wape": 0.10, "sign_ok": True},
        {"own_elasticity": -2.1, "test_wape": 0.11, "sign_ok": True},
        {"own_elasticity": -1.95, "test_wape": 0.09, "sign_ok": True},
        {"own_elasticity": -2.05, "test_wape": 0.12, "sign_ok": True},
    ]
    v = evaluate_ppg("PPG_X", folds)
    assert v.verdict == "pass"
    assert v.sign_stability == 1.0
    assert all(c["status"] in ("pass", "info") for c in v.checks)


def test_evaluate_warns_when_wape_high() -> None:
    folds = [
        {"own_elasticity": -2.0, "test_wape": 0.25, "sign_ok": True},
        {"own_elasticity": -2.1, "test_wape": 0.27, "sign_ok": True},
        {"own_elasticity": -1.95, "test_wape": 0.26, "sign_ok": True},
        {"own_elasticity": -2.05, "test_wape": 0.28, "sign_ok": True},
    ]
    v = evaluate_ppg("PPG_X", folds)
    assert v.verdict == "warn"
    assert any(c["name"] == "wape_mean" and c["status"] == "warn" for c in v.checks)


def test_evaluate_fails_when_sign_flips_majority() -> None:
    folds = [
        {"own_elasticity": -1.0, "test_wape": 0.15, "sign_ok": True},
        {"own_elasticity": 0.5, "test_wape": 0.18, "sign_ok": False},
        {"own_elasticity": 0.7, "test_wape": 0.20, "sign_ok": False},
        {"own_elasticity": 0.4, "test_wape": 0.17, "sign_ok": False},
    ]
    v = evaluate_ppg("PPG_X", folds)
    assert v.verdict == "fail"
    sign_check = next(c for c in v.checks if c["name"] == "sign_stability")
    assert sign_check["status"] == "fail"


def test_evaluate_warns_when_cv_high() -> None:
    folds = [
        {"own_elasticity": -1.0, "test_wape": 0.10, "sign_ok": True},
        {"own_elasticity": -3.5, "test_wape": 0.12, "sign_ok": True},
        {"own_elasticity": -1.2, "test_wape": 0.11, "sign_ok": True},
        {"own_elasticity": -3.0, "test_wape": 0.13, "sign_ok": True},
    ]
    v = evaluate_ppg("PPG_X", folds)
    assert v.elasticity_cv > 0.4
    assert v.verdict in ("warn", "fail")
    cv_check = next(c for c in v.checks if c["name"] == "elasticity_cv")
    assert cv_check["status"] in ("warn", "fail")


def test_evaluate_no_folds_returns_fail() -> None:
    v = evaluate_ppg("PPG_X", [])
    assert v.verdict == "fail"
    assert v.n_folds == 0


def _seed_run(tmp_path: Path, frame: pd.DataFrame, modeling: dict, options: dict | None = None) -> RunState:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir, options=options or {})
    state.run_dir = str(run_dir.resolve())
    state.agents["validation"] = AgentResult(agent="validation", status=AgentStatus.pending)
    frame.to_csv(run_dir / "features.csv", index=False)
    (run_dir / "modeling_results.json").write_text(json.dumps(modeling))
    return state


def test_validation_agent_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frame = _toy_frame(n=80)
    modeling = {
        "controls_used": ["tpr_share", "log_distribution_acv"],
        "per_ppg": [
            {
                "ppg_id": "PPG_V",
                "winner_model": "loglog_ols",
                "winner": {"model": "loglog_ols"},
                "attempts": [],
                "sign_retry_fired": False,
            }
        ],
    }
    state = _seed_run(tmp_path, frame, modeling)
    asyncio.run(ValidationAgent().run(state))

    run_dir = Path(state.run_dir)
    for name in ("validation_report.json", "validation_table.json"):
        assert (run_dir / name).exists()

    table = json.loads((run_dir / "validation_table.json").read_text())
    assert len(table) == 1
    assert table[0]["ppg_id"] == "PPG_V"
    assert table[0]["verdict"] in ("pass", "warn", "fail")

    report = json.loads((run_dir / "validation_report.json").read_text())
    assert "thresholds" in report
    assert report["per_ppg"][0]["n_folds"] >= 2


def test_validation_agent_skips_lightgbm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frame = _toy_frame(n=80)
    modeling = {
        "controls_used": [],
        "per_ppg": [
            {
                "ppg_id": "PPG_V",
                "winner_model": "lightgbm",
                "winner": {"model": "lightgbm"},
                "attempts": [],
                "sign_retry_fired": False,
            }
        ],
    }
    state = _seed_run(tmp_path, frame, modeling)
    asyncio.run(ValidationAgent().run(state))
    out = state.agents["validation"].outputs
    assert out["n_validated"] == 0
    assert out["n_skipped"] == 1


def test_validation_agent_honours_n_folds_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    frame = _toy_frame(n=80)
    modeling = {
        "controls_used": ["tpr_share"],
        "per_ppg": [
            {
                "ppg_id": "PPG_V",
                "winner_model": "loglog_ols",
                "winner": {"model": "loglog_ols"},
                "attempts": [],
                "sign_retry_fired": False,
            }
        ],
    }
    state = _seed_run(
        tmp_path, frame, modeling, options={"validation": {"n_folds": 6}}
    )
    asyncio.run(ValidationAgent().run(state))
    report = json.loads((Path(state.run_dir) / "validation_report.json").read_text())
    assert report["n_folds"] == 6
    assert report["per_ppg"][0]["n_folds"] == 6
