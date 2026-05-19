"""Simulation agent.

Sweeps a price × promo scenario grid per PPG using the winning OLS model
from the modelling stage. The output is the warm-start dataset the
optimisation agent consumes, and a what-if surface the UI can render as
a heatmap (price multiplier × promo state -> revenue / margin / units).

LightGBM-winning PPGs are skipped for now — the same follow-up that adds
ablation decomposition will add the LightGBM simulator. Default
modelling on the synthetic panel produces OLS winners for ~all PPGs so
this is a minor gap.

Outputs:
- ``simulation_grid.json`` per-cell results per PPG.
- ``simulation_summary.json`` best-revenue + best-margin per PPG.
- ``simulation_table.json`` flat (PPG × objective) row table for the UI.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from core.agents.base import Agent
from core.orchestrator.state import AgentResult, ArtifactRef, RunState
from core.simulation.grid import (
    DEFAULT_PRICE_MULTIPLIERS,
    DEFAULT_PROMO_STATES,
    ScenarioGridSpec,
    grid_summary,
    simulate_ols_grid,
)


SYSTEM_PROMPT = """You are the pricing scenario analyst. You receive a
compact per-PPG table with revenue-optimal and margin-optimal price
multipliers from a vectorised what-if sweep. Return STRICT JSON:
{"headline": "<=240 chars cross-PPG summary noting which PPGs gain most
from a price cut vs increase",
 "per_ppg": [{"ppg_id": "...", "rationale": "<=140 chars on the chosen
multipliers"}]}
JSON only. Cite only PPGs in the input."""


SUPPORTED_OLS_MODELS = {"loglog_ols", "semilog_ols"}


def _load_modeling(run_dir: Path) -> dict:
    path = run_dir / "modeling_results.json"
    if not path.exists():
        raise RuntimeError("modeling_results.json missing — modeling must run first")
    return json.loads(path.read_text())


def _load_features(run_dir: Path) -> pd.DataFrame:
    pq = run_dir / "features.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except (ImportError, ValueError):
            pass
    csv = run_dir / "features.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise RuntimeError("features artifact missing — feature_engineering must run first")


def _context_for_ppg(slice_: pd.DataFrame, coefficients: dict[str, float]) -> dict[str, float]:
    """Mean values for every model column that isn't swept by the grid.

    Promo columns + price columns are excluded; the simulator overwrites
    those per cell.
    """
    excluded = {
        "tpr_share",
        "display_share",
        "feature_share",
        "log_price",
        "log_price_gap",
        "log_base_price",
        "price",
    }
    context: dict[str, float] = {}
    for col in coefficients:
        if col in ("const",) or col in excluded:
            continue
        if col in slice_.columns:
            context[col] = float(slice_[col].mean())
    if "log_competitor_price" in slice_.columns:
        context["log_competitor_price"] = float(slice_["log_competitor_price"].mean())
    return context


def _base_price_for(ppg_id: str, slice_: pd.DataFrame, model_kind: str) -> float:
    if model_kind == "semilog_ols" and "price" in slice_.columns:
        return float(slice_["price"].mean())
    if "log_base_price" in slice_.columns:
        return float(pd.Series(slice_["log_base_price"]).pipe(lambda s: s.mean()))
    raise ValueError(f"no base price column available for {ppg_id}")


def _simulate_one(
    ppg_id: str,
    slice_: pd.DataFrame,
    coefficients: dict[str, float],
    model_kind: str,
) -> tuple[pd.DataFrame, dict]:
    import numpy as np

    if model_kind == "loglog_ols":
        log_base_price = (
            float(slice_["log_base_price"].mean()) if "log_base_price" in slice_.columns else 0.0
        )
        base_price = float(np.exp(log_base_price))
    else:
        base_price = float(slice_["price"].mean()) if "price" in slice_.columns else 1.0

    spec = ScenarioGridSpec(
        price_multipliers=DEFAULT_PRICE_MULTIPLIERS,
        promo_states=DEFAULT_PROMO_STATES,
        context=_context_for_ppg(slice_, coefficients),
    )
    grid = simulate_ols_grid(coefficients, base_price, spec, model_kind=model_kind)
    return grid, grid_summary(grid)


class SimulationAgent(Agent):
    name = "simulation"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        run_dir = Path(run.run_dir)
        feats = await asyncio.to_thread(_load_features, run_dir)
        modeling = await asyncio.to_thread(_load_modeling, run_dir)

        # Use the same coefficients the modelling agent recorded so the
        # simulation is consistent with the chosen winner without refitting.
        grids: list[dict] = []
        summaries: list[dict] = []
        table: list[dict] = []
        skipped: list[dict] = []

        for row in modeling.get("per_ppg", []):
            ppg_id = row["ppg_id"]
            winner = row.get("winner_model")
            slice_ = feats[feats["ppg_id"] == ppg_id]
            if winner not in SUPPORTED_OLS_MODELS or slice_.empty:
                skipped.append(
                    {
                        "ppg_id": ppg_id,
                        "winner_model": winner,
                        "reason": (
                            "no rows for PPG"
                            if slice_.empty
                            else f"closed-form simulation not supported for {winner}"
                        ),
                    }
                )
                continue
            coefficients = row.get("winner", {}).get("coefficients", {})
            if not coefficients:
                skipped.append(
                    {"ppg_id": ppg_id, "winner_model": winner, "reason": "no coefficients"}
                )
                continue

            grid, summary = await asyncio.to_thread(
                _simulate_one, ppg_id, slice_, coefficients, winner
            )
            await self.emit(
                run,
                "tool_called",
                {
                    "tool": "simulate_grid",
                    "ppg_id": ppg_id,
                    "model_kind": winner,
                    "n_cells": int(len(grid)),
                },
            )

            grids.append(
                {
                    "ppg_id": ppg_id,
                    "model_kind": winner,
                    "cells": grid.to_dict(orient="records"),
                }
            )
            summary["ppg_id"] = ppg_id
            summary["model_kind"] = winner
            summaries.append(summary)

            table.append(
                {
                    "ppg_id": ppg_id,
                    "objective": "revenue",
                    "price_multiplier": summary["best_revenue"]["price_multiplier"],
                    "promo": summary["best_revenue"]["promo"],
                    "value": summary["best_revenue"]["revenue"],
                    "units": summary["best_revenue"]["units"],
                }
            )
            table.append(
                {
                    "ppg_id": ppg_id,
                    "objective": "margin",
                    "price_multiplier": summary["best_margin"]["price_multiplier"],
                    "promo": summary["best_margin"]["promo"],
                    "value": summary["best_margin"]["margin"],
                    "units": summary["best_margin"]["units"],
                }
            )

        grids_path = run_dir / "simulation_grid.json"
        grids_path.write_text(json.dumps(grids, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(grids_path),
                mime="application/json",
                agent=self.name,
                name=grids_path.name,
            )
        )

        summary_path = run_dir / "simulation_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(summary_path),
                mime="application/json",
                agent=self.name,
                name=summary_path.name,
            )
        )

        table_path = run_dir / "simulation_table.json"
        table_path.write_text(json.dumps(table, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(table_path),
                mime="application/json",
                agent=self.name,
                name=table_path.name,
            )
        )

        headline, rationales = self._narrate(result, summaries)
        rationale_by_id = {r["ppg_id"]: r["rationale"] for r in rationales}
        for s in summaries:
            s["rationale"] = rationale_by_id.get(s["ppg_id"], "")

        result.outputs = {
            "n_simulated": len(summaries),
            "n_skipped": len(skipped),
            "n_cells_per_ppg": len(DEFAULT_PRICE_MULTIPLIERS) * len(DEFAULT_PROMO_STATES),
            "price_grid": list(DEFAULT_PRICE_MULTIPLIERS),
        }
        if skipped:
            result.outputs["skipped"] = skipped
        result.reasoning = headline or (
            f"Simulated {len(summaries)} PPGs across "
            f"{len(DEFAULT_PRICE_MULTIPLIERS)}×{len(DEFAULT_PROMO_STATES)}-cell grids."
        )
        result.confidence = 0.9 if summaries else 0.0

    def _narrate(
        self, result: AgentResult, summaries: list[dict]
    ) -> tuple[str, list[dict]]:
        compact = [
            {
                "ppg_id": s["ppg_id"],
                "best_revenue": s["best_revenue"],
                "best_margin": s["best_margin"],
            }
            for s in summaries
        ]
        try:
            resp = self.call_llm(
                result,
                system=SYSTEM_PROMPT,
                user=json.dumps(compact, default=str),
                max_tokens=700,
                label="simulation-rationale",
            )
            if resp.raw.get("dry_run"):
                return "", []
            blob = json.loads(resp.text)
            return str(blob.get("headline", "")), list(blob.get("per_ppg", []))
        except (json.JSONDecodeError, ValueError):
            return "", []
