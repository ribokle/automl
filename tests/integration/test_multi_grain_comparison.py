"""Comparison-grain fan-out emits per-grain modelling artifacts.

Builds a small synthetic warehouse, hand-runs ingestion +
ppg_mapping + ppg_selection + feature_engineering + modeling at the
primary grain (chain × PPG), then exercises ``_run_comparison_grains``
to fan out a brand-grain comparison pass. The primary modelling outputs
must survive at their canonical filenames, and the comparison outputs
must land at the grain-suffixed filenames the UI loads.
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
from core.config import ModellingGrain
from core.orchestrator.runner import _run_comparison_grains
from core.orchestrator.state import AgentResult, AgentStatus, RunState


def _seed(tmp_path: Path) -> RunState:
    rng = np.random.default_rng(11)
    rows: list[dict] = []
    weeks = pd.date_range("2024-01-01", periods=60, freq="W-MON")
    # 4 brands × 8 PPGs (2 per brand) × 60 weeks → 480 PPG-rows.
    brands = ["acme", "globex", "initech", "soylent"]
    for brand in brands:
        for sub in ("v1", "v2"):
            ppg_id = f"{brand}_{sub}"
            true_e = -1.3 if sub == "v1" else -1.0
            for w in weeks:
                price = 5.0 + rng.normal(0, 0.4)
                log_units = 4.5 + true_e * (np.log(price) - np.log(5.0)) + rng.normal(0, 0.05)
                rows.append(
                    {
                        "store_id": "s1",
                        "ppg_id": ppg_id,
                        "category": "cat",
                        "brand": brand,
                        "sku": f"{ppg_id}_a",
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

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState.new(
        data_path=str(tmp_path / "x.csv"),
        run_dir=run_dir,
        options={
            "modelling_grain": ModellingGrain.PPG_WEEK,
            "comparison_grains": [ModellingGrain.BRAND_WEEK.value],
            # Keep the fan-out tight for this synthetic warehouse: only
            # the modelling tail, no downstream agents (this test
            # doesn't seed the artifacts they need).
            "comparison_agents": ["modeling"],
        },
    )
    state.run_dir = str(run_dir.resolve())
    state.duckdb_path = str(db.resolve())
    for name in ("feature_engineering", "modeling"):
        state.agents[name] = AgentResult(agent=name, status=AgentStatus.pending)
    # PPG-grain eligibility: all 8 PPGs.
    ppg_ids = sorted({r["ppg_id"] for r in rows})
    (run_dir / "ppg_selection.json").write_text(
        json.dumps([{"ppg_id": pid, "eligible": True} for pid in ppg_ids])
    )
    return state


def test_comparison_fanout_writes_grain_suffixed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    state = _seed(tmp_path)
    run_dir = Path(state.run_dir)

    # Hand-run the primary FE + modelling pass so the canonical
    # artifacts exist before the comparison loop kicks in.
    asyncio.run(FeatureEngineeringAgent().run(state))
    asyncio.run(ModelingAgent().run(state))

    primary_mr = json.loads((run_dir / "modeling_results.json").read_text())
    primary_n_total = primary_mr["n_total"]
    assert primary_n_total > 0
    primary_ppg_ids = {r["ppg_id"] for r in primary_mr["per_ppg"]}
    # PPG-grain rows are PPG ids (acme_v1, globex_v1, ...).
    assert any("_v1" in p for p in primary_ppg_ids)

    asyncio.run(_run_comparison_grains(state, agent_mode=False))

    # Primary artifacts still at the canonical filenames.
    restored = json.loads((run_dir / "modeling_results.json").read_text())
    assert restored == primary_mr

    # Brand-grain comparison artifacts land at the suffixed names.
    comp_path = run_dir / "modeling_results__brand_week.json"
    assert comp_path.exists(), "comparison-grain modelling artifact missing"
    comp_mr = json.loads(comp_path.read_text())
    assert comp_mr["n_total"] > 0
    # Brand-grain ppg_ids are the brand labels, not the PPG ids.
    comp_ppg_ids = {r["ppg_id"] for r in comp_mr["per_ppg"]}
    assert "acme" in comp_ppg_ids or "globex" in comp_ppg_ids

    # No leftover backup files.
    leftovers = list(run_dir.glob("*__primary_backup__*"))
    assert leftovers == []

    # Run state restored: modelling_grain back to the primary.
    assert state.options["modelling_grain"] == ModellingGrain.PPG_WEEK
