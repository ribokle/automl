"""End-to-end shape test for the Hoch-style modelling grain.

Runs the feature_engineering -> modeling slice of the pipeline against
a tiny synthetic warehouse with ``MODELLING_GRAIN=store_ppg_week`` and
asserts the new artefacts carry per-cell rows plus a pooled-to-PPG
view downstream agents consume.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from core.agents.feature_engineering import FeatureEngineeringAgent
from core.agents.modeling import ModelingAgent
from core.config import ModellingGrain, get_settings
from core.orchestrator.state import AgentResult, AgentStatus, RunState


def _seed_warehouse(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    rows: list[dict] = []
    weeks = pd.date_range("2024-01-01", periods=40, freq="W-MON")
    # 3 stores × 2 PPGs × 40 weeks — every store has enough rows to fit per cell.
    for store in ("store_1", "store_2", "store_3"):
        for ppg in ("PPG_A", "PPG_B"):
            # Per-store baseline so cells aren't identical.
            store_intercept = 4.0 + 0.4 * "_123".find(store[-1])
            true_e = -1.4 if ppg == "PPG_A" else -1.1
            for w in weeks:
                price = 5.0 + rng.normal(0, 0.4)
                log_units = (
                    store_intercept + true_e * (np.log(price) - np.log(5.0)) + rng.normal(0, 0.05)
                )
                rows.append(
                    {
                        "sku": f"{ppg}_{store}",
                        "store_id": store,
                        "ppg_id": ppg,
                        "category": "cat",
                        "week_start": w.date(),
                        "units": int(np.exp(log_units)),
                        "price": price,
                        "base_price": 5.5,
                        "discount_depth": max(0.0, (5.5 - price) / 5.5),
                        "tpr_flag": int(price < 4.7),
                        "display_flag": 0,
                        "feature_flag": 0,
                        "distribution_acv": 100.0,
                        "competitor_price": price + rng.normal(0, 0.2),
                        "holiday": None,
                    }
                )
    df = pd.DataFrame(rows)
    db = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(db))
    try:
        con.register("panel_df", df)
        con.execute("CREATE SCHEMA IF NOT EXISTS main")
        con.execute("CREATE TABLE main.panel AS SELECT * FROM panel_df")
    finally:
        con.close()
    return db


def _seed_run(tmp_path: Path, db: Path) -> RunState:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.new(
        data_path=str(tmp_path / "x.csv"),
        run_dir=run_dir,
        options={"modelling_grain": ModellingGrain.STORE_PPG_WEEK},
    )
    state.run_dir = str(run_dir.resolve())
    state.duckdb_path = str(db.resolve())
    state.agents["feature_engineering"] = AgentResult(
        agent="feature_engineering", status=AgentStatus.pending
    )
    state.agents["modeling"] = AgentResult(agent="modeling", status=AgentStatus.pending)
    (run_dir / "ppg_selection.json").write_text(
        json.dumps([{"ppg_id": "PPG_A", "eligible": True}, {"ppg_id": "PPG_B", "eligible": True}])
    )
    return state


def test_store_grain_emits_per_cell_and_pooled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    db = _seed_warehouse(tmp_path)
    state = _seed_run(tmp_path, db)

    asyncio.run(FeatureEngineeringAgent().run(state))
    run_dir = Path(state.run_dir)
    feats_path = run_dir / "features.parquet"
    if not feats_path.exists():
        feats_path = run_dir / "features.csv"
    feats = pd.read_parquet(feats_path) if feats_path.suffix == ".parquet" else pd.read_csv(feats_path)
    assert "grain_unit" in feats.columns
    # 3 stores x 2 PPGs distinct cells.
    assert feats.groupby(["grain_unit", "ppg_id"]).ngroups == 6

    # feature_refine isn't run here, so let modeling use the default
    # control set (engineered minus log_price/log_units).
    asyncio.run(ModelingAgent().run(state))

    modeling = json.loads((run_dir / "modeling_results.json").read_text())
    # One row per (grain_unit, ppg) cell.
    cell_keys = {(r["ppg_id"], r.get("grain_unit")) for r in modeling["per_ppg"]}
    assert cell_keys == {
        ("PPG_A", "store_1"), ("PPG_A", "store_2"), ("PPG_A", "store_3"),
        ("PPG_B", "store_1"), ("PPG_B", "store_2"), ("PPG_B", "store_3"),
    }

    # Every winner has the right sign and an in-band magnitude.
    for r in modeling["per_ppg"]:
        w = r.get("winner")
        if not w:
            continue
        assert w["sign_ok"], f"{r['ppg_id']}@{r['grain_unit']} winner has wrong sign"
        assert abs(w["own_elasticity"]) <= 8.0

    # Pooled view collapses store cells back to one row per PPG.
    pooled = json.loads((run_dir / "elasticity_per_ppg_pooled.json").read_text())
    pooled_ppgs = {r["ppg_id"] for r in pooled}
    assert pooled_ppgs == {"PPG_A", "PPG_B"}
    for row in pooled:
        assert row["grain_unit"] == "pooled"
        assert row["model"] == "store_inverse_variance_pool"
        assert row["n_stores_pooled"] >= 1
        assert row["sign_ok"]
