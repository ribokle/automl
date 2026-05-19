"""Modeling agent + log-log / semi-log fitters.

Phase 3 verification anchor: log-log OLS must recover the correct
elasticity sign for >=7/8 PPGs on the synthetic panel. Also covers the
semi-log sign-retry path with a synthetic mini-frame where log-log
deliberately produces the wrong sign.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from core.agents.modeling import ModelingAgent
from core.features.eda import ppg_week_aggregate
from core.features.engineering import build_features
from core.models.loglog_ols import fit_loglog
from core.models.semilog_ols import fit_semilog
from core.orchestrator.state import AgentResult, AgentStatus, RunState
from synthetic.elasticity_spec import PPGS
from synthetic.generator import generate_panel


@pytest.fixture(scope="module")
def synthetic_features(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Build the engineered PPG×week feature frame the modelling agent expects.

    Uses the real pipeline: synthetic panel -> DuckDB -> ppg_week_aggregate
    -> build_features, so any drift in the upstream contract surfaces here.
    """
    seed_dir = tmp_path_factory.mktemp("modeling_fixture")
    panel, _ = generate_panel(seed=42)
    # Mirror the derived column the dbt panel mart materialises so the
    # in-DuckDB aggregator doesn't fail with BinderError on a fresh raw load.
    panel["discount_depth"] = np.where(
        panel["base_price"] > 0,
        1.0 - panel["price"] / panel["base_price"],
        0.0,
    )
    db = seed_dir / "warehouse.duckdb"
    con = duckdb.connect(str(db))
    try:
        con.register("panel_df", panel)
        con.execute("CREATE SCHEMA IF NOT EXISTS main")
        con.execute("CREATE TABLE main.panel AS SELECT * FROM panel_df")
    finally:
        con.close()
    return build_features(ppg_week_aggregate(db))


def test_loglog_recovers_sign_on_synthetic(synthetic_features: pd.DataFrame) -> None:
    """P3 acceptance metric: log-log alone must get the sign right for >=7/8 PPGs."""
    truths = {p.ppg_id: p.own_elasticity for p in PPGS}
    correct = 0
    for ppg_id in truths:
        slice_ = synthetic_features[synthetic_features["ppg_id"] == ppg_id]
        fit = fit_loglog(ppg_id, slice_, controls=["log_distribution_acv", "tpr_share"])
        if fit.sign_ok:
            correct += 1
    assert correct >= 7, f"log-log recovered sign for only {correct}/8 PPGs on synthetic"


def test_loglog_elasticity_within_loose_band(synthetic_features: pd.DataFrame) -> None:
    """Magnitudes are not the acceptance metric for this slice, but they
    should at least be in plausible CPG range (|ε| between 0.3 and 6) on
    >=5/8 PPGs so the modelling agent's output isn't pathologically off."""
    in_band = 0
    for p in PPGS:
        slice_ = synthetic_features[synthetic_features["ppg_id"] == p.ppg_id]
        fit = fit_loglog(p.ppg_id, slice_, controls=["log_distribution_acv", "tpr_share"])
        if fit.sign_ok and 0.3 <= abs(fit.own_elasticity) <= 6.0:
            in_band += 1
    assert in_band >= 5, f"only {in_band}/8 PPGs produced a plausible elasticity magnitude"


def test_semilog_independently_recovers_sign(synthetic_features: pd.DataFrame) -> None:
    """Semi-log is the retry fallback. As a smoke check, semi-log on its own
    should also recover the elasticity sign for the bulk of PPGs on a clean
    DGP — otherwise the retry is useless."""
    correct = 0
    for p in PPGS:
        slice_ = synthetic_features[synthetic_features["ppg_id"] == p.ppg_id]
        fit = fit_semilog(p.ppg_id, slice_, controls=["log_distribution_acv", "tpr_share"])
        if fit.sign_ok:
            correct += 1
    assert correct >= 6, f"semi-log only recovered sign for {correct}/8 PPGs"


def test_sign_retry_and_wape_selection(
    tmp_path: Path,
    synthetic_features: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the modeling agent's selection logic in isolation: when
    log-log returns the wrong sign, semi-log gets refitted, and the winner
    is the lowest-test-WAPE sign-correct candidate among all three
    families. All three fitters are mocked so the assertion is about the
    routing, not the underlying numerics."""
    import core.agents.modeling as modeling_mod
    from core.models.base import ElasticityFit

    def _fit(model: str, elasticity: float, wape: float) -> ElasticityFit:
        return ElasticityFit(
            ppg_id="PPG01",
            model=model,
            own_elasticity=elasticity,
            std_err=0.1,
            p_value=0.001,
            r_squared=0.9,
            n_obs=80,
            controls=["tpr_share"],
            coefficients={},
            diagnostics={"test_wape": wape, "n_test": 20},
        )

    monkeypatch.setattr(modeling_mod, "fit_loglog", lambda *a, **kw: _fit("loglog_ols", 0.5, 0.20))
    monkeypatch.setattr(modeling_mod, "fit_semilog", lambda *a, **kw: _fit("semilog_ols", -1.4, 0.14))
    monkeypatch.setattr(modeling_mod, "fit_lightgbm", lambda *a, **kw: _fit("lightgbm", -1.1, 0.10))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    state = _seed_run(tmp_path, synthetic_features, ["PPG01"])
    asyncio.run(ModelingAgent().run(state))
    results = json.loads((Path(state.run_dir) / "modeling_results.json").read_text())
    [row] = results["per_ppg"]
    assert row["sign_retry_fired"] is True
    assert len(row["attempts"]) == 3
    # LightGBM has lowest WAPE among sign-correct -> winner.
    assert row["winner_model"] == "lightgbm"
    assert row["winner"]["sign_ok"] is True


def test_skips_lightgbm_only_path_when_loglog_sign_ok(
    tmp_path: Path,
    synthetic_features: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When log-log already has the right sign, semi-log does NOT get
    fitted (sign_retry_fired stays False) and the comparison is between
    log-log and LightGBM only."""
    import core.agents.modeling as modeling_mod
    from core.models.base import ElasticityFit

    def _fit(model: str, elasticity: float, wape: float) -> ElasticityFit:
        return ElasticityFit(
            ppg_id="PPG01",
            model=model,
            own_elasticity=elasticity,
            std_err=0.1,
            p_value=0.001,
            r_squared=0.85,
            n_obs=80,
            controls=[],
            coefficients={},
            diagnostics={"test_wape": wape, "n_test": 20},
        )

    monkeypatch.setattr(modeling_mod, "fit_loglog", lambda *a, **kw: _fit("loglog_ols", -2.0, 0.08))
    monkeypatch.setattr(modeling_mod, "fit_lightgbm", lambda *a, **kw: _fit("lightgbm", -1.6, 0.15))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    state = _seed_run(tmp_path, synthetic_features, ["PPG01"])
    asyncio.run(ModelingAgent().run(state))
    results = json.loads((Path(state.run_dir) / "modeling_results.json").read_text())
    [row] = results["per_ppg"]
    assert row["sign_retry_fired"] is False
    assert len(row["attempts"]) == 2
    assert row["winner_model"] == "loglog_ols"


def _seed_run(tmp_path: Path, features: pd.DataFrame, eligible_ppgs: list[str]) -> RunState:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.new(data_path=str(tmp_path / "x.csv"), run_dir=run_dir)
    state.run_dir = str(run_dir.resolve())
    state.agents["modeling"] = AgentResult(agent="modeling", status=AgentStatus.pending)
    features.to_csv(run_dir / "features.csv", index=False)
    (run_dir / "ppg_selection.json").write_text(
        json.dumps([{"ppg_id": p, "eligible": True} for p in eligible_ppgs])
    )
    (run_dir / "feature_refine.json").write_text(
        json.dumps({"kept": ["log_price", "log_distribution_acv", "tpr_share"]})
    )
    return state


def test_modeling_agent_writes_per_ppg_artifacts(
    tmp_path: Path, synthetic_features: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    eligible = [p.ppg_id for p in PPGS]
    state = _seed_run(tmp_path, synthetic_features, eligible)
    asyncio.run(ModelingAgent().run(state))

    run_dir = Path(state.run_dir)
    results = json.loads((run_dir / "modeling_results.json").read_text())
    compact = json.loads((run_dir / "elasticity_per_ppg.json").read_text())

    assert results["n_total"] == 8
    assert results["n_correct_sign"] >= 7
    assert results["model_pool"] == ["loglog_ols", "semilog_ols", "lightgbm"]
    assert len(compact) == 8
    for row in compact:
        assert row["model"] in {"loglog_ols", "semilog_ols", "lightgbm", "skipped"}
        if row["model"] != "skipped":
            assert row["n_obs"] > 0
            assert "own_elasticity" in row
            assert row["test_wape"] is not None and row["test_wape"] >= 0
