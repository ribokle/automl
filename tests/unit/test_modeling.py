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


def test_sign_retry_invokes_semilog_when_loglog_wrong(
    tmp_path: Path,
    synthetic_features: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When log-log returns a positive sign, the agent must refit with
    semi-log and record both attempts. We force the failure by patching
    ``fit_loglog`` to flip the elasticity sign on one PPG, then assert the
    retry fired and semi-log won."""
    import core.agents.modeling as modeling_mod
    from core.models.base import ElasticityFit

    original_loglog = modeling_mod.fit_loglog
    target_ppg = "PPG01"

    def flipped(ppg_id: str, frame: pd.DataFrame, controls: list[str]) -> ElasticityFit:
        real = original_loglog(ppg_id, frame, controls)
        if ppg_id == target_ppg:
            real.own_elasticity = abs(real.own_elasticity)  # force wrong sign
        return real

    monkeypatch.setattr(modeling_mod, "fit_loglog", flipped)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    state = _seed_run(tmp_path, synthetic_features, [target_ppg])
    asyncio.run(ModelingAgent().run(state))
    results = json.loads((Path(state.run_dir) / "modeling_results.json").read_text())
    [row] = results["per_ppg"]
    assert row["sign_retry_fired"] is True
    assert row["winner_model"] == "semilog_ols"
    assert len(row["attempts"]) == 2


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
    assert len(compact) == 8
    for row in compact:
        assert row["model"] in {"loglog_ols", "semilog_ols", "skipped"}
        if row["model"] != "skipped":
            assert row["n_obs"] > 0
            assert "own_elasticity" in row
