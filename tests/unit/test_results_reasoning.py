"""Results-reasoning agent.

Reads the modelling agent's output, runs deterministic verdict checks
(sign / magnitude band / R² floor / hold-out WAPE), and writes a
table-shaped summary the UI renders. LLM narration is optional and
gracefully degrades to deterministic text under dry-run.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.agents.results_reasoning import ResultsReasoningAgent
from core.orchestrator.state import AgentResult, AgentStatus, RunState


def _modelling_blob(ppgs: list[dict]) -> dict:
    return {
        "controls_used": ["log_distribution_acv", "tpr_share"],
        "per_ppg": ppgs,
        "n_correct_sign": sum(1 for p in ppgs if p["winner"] and p["winner"]["sign_ok"]),
        "n_retries": sum(1 for p in ppgs if p["sign_retry_fired"]),
        "n_skipped": sum(1 for p in ppgs if p["winner_model"] == "skipped"),
        "n_total": len(ppgs),
        "model_pool": ["loglog_ols", "semilog_ols", "lightgbm"],
    }


def _winner(
    *,
    elasticity: float,
    r2: float = 0.85,
    wape: float = 0.10,
    model: str = "loglog_ols",
) -> dict:
    return {
        "ppg_id": "X",
        "model": model,
        "own_elasticity": elasticity,
        "std_err": 0.1,
        "p_value": 0.001,
        "r_squared": r2,
        "n_obs": 80,
        "controls": ["log_distribution_acv", "tpr_share"],
        "coefficients": {},
        "diagnostics": {"train_wape": 0.08, "test_wape": wape, "n_test": 20},
        "sign_ok": elasticity < 0,
    }


def _seed_run(tmp_path: Path, modelling: dict) -> RunState:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    state.run_dir = str(run_dir.resolve())
    state.agents["results_reasoning"] = AgentResult(
        agent="results_reasoning", status=AgentStatus.pending
    )
    (run_dir / "modeling_results.json").write_text(json.dumps(modelling))
    return state


def test_pass_verdict_for_clean_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    win = {**_winner(elasticity=-1.8, r2=0.9, wape=0.08), "ppg_id": "PPG_A"}
    modelling = _modelling_blob(
        [
            {
                "ppg_id": "PPG_A",
                "winner_model": "loglog_ols",
                "sign_retry_fired": False,
                "attempts": [win],
                "winner": win,
                "n_train": 80,
                "n_test": 20,
            }
        ]
    )
    state = _seed_run(tmp_path, modelling)
    asyncio.run(ResultsReasoningAgent().run(state))
    verdict = json.loads((Path(state.run_dir) / "results_reasoning.json").read_text())
    assert verdict["n_pass"] == 1 and verdict["n_warn"] == 0 and verdict["n_fail"] == 0
    [row] = verdict["per_ppg"]
    assert row["verdict"] == "pass"
    statuses = {c["name"]: c["status"] for c in row["checks"]}
    assert statuses == {
        "sign_correct": "pass",
        "magnitude_band": "pass",
        "r_squared_floor": "pass",
        "holdout_wape": "pass",
    }


def test_warn_when_elasticity_out_of_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # |ε|=10 is well outside the [0.3, 6.0] CPG plausibility band.
    win = {**_winner(elasticity=-10.0), "ppg_id": "PPG_B"}
    modelling = _modelling_blob(
        [
            {
                "ppg_id": "PPG_B",
                "winner_model": "loglog_ols",
                "sign_retry_fired": False,
                "attempts": [win],
                "winner": win,
                "n_train": 80,
                "n_test": 20,
            }
        ]
    )
    state = _seed_run(tmp_path, modelling)
    asyncio.run(ResultsReasoningAgent().run(state))
    verdict = json.loads((Path(state.run_dir) / "results_reasoning.json").read_text())
    [row] = verdict["per_ppg"]
    assert row["verdict"] == "warn"
    statuses = {c["name"]: c["status"] for c in row["checks"]}
    assert statuses["magnitude_band"] == "warn"


def test_fail_when_sign_wrong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    win = {**_winner(elasticity=1.5), "ppg_id": "PPG_C"}
    modelling = _modelling_blob(
        [
            {
                "ppg_id": "PPG_C",
                "winner_model": "loglog_ols",
                "sign_retry_fired": False,
                "attempts": [win],
                "winner": win,
                "n_train": 80,
                "n_test": 20,
            }
        ]
    )
    state = _seed_run(tmp_path, modelling)
    asyncio.run(ResultsReasoningAgent().run(state))
    verdict = json.loads((Path(state.run_dir) / "results_reasoning.json").read_text())
    [row] = verdict["per_ppg"]
    assert row["verdict"] == "fail"


def test_summary_table_is_one_row_per_ppg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ppgs = [
        {
            "ppg_id": f"PPG_{i}",
            "winner_model": "loglog_ols",
            "sign_retry_fired": False,
            "attempts": [
                {**_winner(elasticity=-1.5 - 0.1 * i, wape=0.1 + 0.01 * i), "ppg_id": f"PPG_{i}"}
            ],
            "winner": {
                **_winner(elasticity=-1.5 - 0.1 * i, wape=0.1 + 0.01 * i),
                "ppg_id": f"PPG_{i}",
            },
            "n_train": 80,
            "n_test": 20,
        }
        for i in range(3)
    ]
    state = _seed_run(tmp_path, _modelling_blob(ppgs))
    asyncio.run(ResultsReasoningAgent().run(state))
    summary = json.loads((Path(state.run_dir) / "model_choice_summary.json").read_text())
    assert len(summary) == 3
    assert {row["ppg_id"] for row in summary} == {"PPG_0", "PPG_1", "PPG_2"}
    for row in summary:
        assert {"ppg_id", "winner", "own_elasticity", "test_wape", "verdict", "rationale"} <= set(
            row.keys()
        )
