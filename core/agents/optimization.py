"""Optimization agent.

For each PPG with an OLS-based winner from the modelling stage, solves
the discrete price-ladder + promo selection problem under the configured
constraints. The pipeline runs scipy continuous optimisation first as a
warm-start anchor; the PuLP MILP then rounds onto the nearest feasible
ladder rung and reports the chosen cell. If the strict MILP is
infeasible, the agent falls back to a soft-constraint relaxation and
surfaces which constraints were violated and by how much.

LightGBM-winning PPGs are skipped for now with a structured note — the
same follow-up that adds an ablation decomposer will add the LightGBM
optimiser. Default modelling on the synthetic panel produces OLS winners
for ~all PPGs so this remains a minor gap.

Constraint configuration:

- Defaults come from ``OptimizationConstraints`` (see
  ``core/optimization/constraints.py``).
- The run's ``options["optimization"]`` dict may override any field —
  this is how the UI constraint editor + the constraint-elicitation
  approval gate feed the agent.

Outputs:

- ``optimization_results.json`` — per-PPG continuous warm-start + MILP
  recommendation + binding violations + cell-level metrics.
- ``optimization_table.json`` — flat ``(ppg_id, multiplier, promo,
  units, revenue, margin, ...)`` rows for the UI's shared
  ``<ResultsTable>``.
- ``optimization_constraints.json`` — the (possibly overridden)
  constraint set that produced the recommendation, persisted for the UI
  + audit trail.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from core.agents.base import Agent
from core.models.predictor import (
    LINEAR_COEFF_MODELS,
    PREDICTABLE_MODELS,
    TREE_MODELS,
    build_predictor,
)
from core.optimization.constraints import OptimizationConstraints, PPGOptInputs
from core.optimization.continuous import solve_continuous
from core.optimization.milp import solve_milp
from core.orchestrator.state import AgentResult, ArtifactRef, RunState


SYSTEM_PROMPT = """You are the pricing-optimisation analyst. You receive
per-PPG recommended price/promo cells from a constrained MILP (price
ladder + margin floor + competitive gap), the continuous warm-start
multiplier, and a list of binding constraints for any PPG where the
problem was relaxed. Return STRICT JSON:
{"headline": "<=240 chars across-PPG summary noting which PPGs got
biggest revenue lifts and which needed constraint relaxation",
 "per_ppg": [{"ppg_id": "...", "rationale": "<=160 chars on the chosen
multiplier and any trade-off"}]}
JSON only. Cite only PPGs in the input."""


SUPPORTED_OLS_MODELS = LINEAR_COEFF_MODELS
SUPPORTED_MODELS = PREDICTABLE_MODELS


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


def _constraints_from_options(options: dict) -> OptimizationConstraints:
    blob = options.get("optimization") or {}
    return OptimizationConstraints(
        price_ladder=tuple(blob.get("price_ladder", OptimizationConstraints().price_ladder)),
        promo_states=tuple(blob.get("promo_states", OptimizationConstraints().promo_states)),
        cog_pct=float(blob.get("cog_pct", OptimizationConstraints().cog_pct)),
        margin_floor_pct=float(blob.get("margin_floor_pct", OptimizationConstraints().margin_floor_pct)),
        comp_gap_pct=float(blob.get("comp_gap_pct", OptimizationConstraints().comp_gap_pct)),
        max_decrease=float(blob.get("max_decrease", OptimizationConstraints().max_decrease)),
        max_increase=float(blob.get("max_increase", OptimizationConstraints().max_increase)),
        objective=str(blob.get("objective", OptimizationConstraints().objective)),
    )


def _build_inputs(
    ppg_id: str,
    slice_: pd.DataFrame,
    modeling_row: dict,
    controls: list[str],
    model_kind: str,
) -> PPGOptInputs:
    import numpy as np

    if model_kind == "semilog_ols":
        base_price = float(slice_["price"].mean()) if "price" in slice_.columns else 1.0
    else:
        log_base_price = (
            float(slice_["log_base_price"].mean()) if "log_base_price" in slice_.columns else 0.0
        )
        base_price = float(np.exp(log_base_price))

    excluded = {
        "tpr_share",
        "display_share",
        "feature_share",
        "log_price",
        "log_price_gap",
        "log_base_price",
        "price",
    }

    if model_kind in SUPPORTED_OLS_MODELS:
        coefficients = dict(modeling_row.get("winner", {}).get("coefficients", {}))
        context: dict[str, float] = {}
        for col in coefficients:
            if col == "const" or col in excluded:
                continue
            if col in slice_.columns:
                context[col] = float(slice_[col].mean())
        predictor = None
    else:  # lightgbm
        coefficients = {}
        predictor = build_predictor(modeling_row, slice_, controls)
        context = {}
        for col in predictor.feature_cols:
            if col in excluded:
                continue
            if col in slice_.columns:
                context[col] = float(slice_[col].mean())

    if "log_competitor_price" in slice_.columns:
        context["log_competitor_price"] = float(slice_["log_competitor_price"].mean())

    competitor_price: float | None = None
    if "competitor_price" in slice_.columns:
        cp = float(slice_["competitor_price"].mean())
        if cp > 0:
            competitor_price = cp
    elif "log_competitor_price" in slice_.columns:
        competitor_price = float(np.exp(slice_["log_competitor_price"].mean()))

    return PPGOptInputs(
        ppg_id=ppg_id,
        model_kind=model_kind,
        coefficients=coefficients,
        base_price=base_price,
        context=context,
        competitor_price=competitor_price,
        predictor=predictor,
    )


def _training_price_envelope(slice_: pd.DataFrame) -> tuple[float, float] | None:
    """Min / max observed price for one PPG's training slice.

    Returns ``None`` when neither ``price`` nor ``log_price`` is available.
    Used by :func:`_clip_ladder_to_envelope` to constrain LightGBM
    recommendations to the price range the booster actually saw — past
    that range the tree ensemble extrapolates flat (even with monotone
    constraints) and the optimiser's chosen multiplier is no longer
    grounded in data.
    """
    import numpy as np

    if "price" in slice_.columns and slice_["price"].notna().any():
        prices = slice_["price"].astype(float).to_numpy()
    elif "log_price" in slice_.columns and slice_["log_price"].notna().any():
        prices = np.exp(slice_["log_price"].astype(float).to_numpy())
    else:
        return None
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if prices.size == 0:
        return None
    return float(prices.min()), float(prices.max())


def _clip_ladder_to_envelope(
    constraints: OptimizationConstraints,
    inp: PPGOptInputs,
    envelope: tuple[float, float] | None,
) -> tuple[OptimizationConstraints, dict | None]:
    """For LightGBM winners, keep only ladder rungs whose absolute price
    lies inside the training-price envelope.

    Returns the (possibly-clipped) constraints and a small audit dict the
    optimisation result blob records so the UI can show why a rung was
    dropped. If clipping would leave zero rungs the original ladder is
    kept verbatim (better to surface a relaxed solution than to silently
    drop the PPG).
    """
    if inp.model_kind not in TREE_MODELS or envelope is None or inp.base_price <= 0:
        return constraints, None
    lo, hi = envelope
    kept: list[float] = []
    dropped: list[float] = []
    for m in constraints.price_ladder:
        price = inp.base_price * m
        if lo - 1e-9 <= price <= hi + 1e-9:
            kept.append(m)
        else:
            dropped.append(m)
    if not kept:
        # Refuse to clip ourselves into a corner — the relaxed MILP will
        # surface a binding-violations note instead.
        return constraints, {
            "ppg_id": inp.ppg_id,
            "envelope": [lo, hi],
            "kept_multipliers": list(constraints.price_ladder),
            "dropped_multipliers": [],
            "ladder_clipped": False,
            "reason": "all ladder rungs outside training envelope; clip skipped",
        }
    if not dropped:
        return constraints, None
    clipped = OptimizationConstraints(
        price_ladder=tuple(kept),
        promo_states=constraints.promo_states,
        cog_pct=constraints.cog_pct,
        margin_floor_pct=constraints.margin_floor_pct,
        comp_gap_pct=constraints.comp_gap_pct,
        max_decrease=constraints.max_decrease,
        max_increase=constraints.max_increase,
        objective=constraints.objective,
    )
    return clipped, {
        "ppg_id": inp.ppg_id,
        "envelope": [lo, hi],
        "kept_multipliers": kept,
        "dropped_multipliers": dropped,
        "ladder_clipped": True,
    }


def _optimise_one(
    inp: PPGOptInputs,
    constraints: OptimizationConstraints,
    envelope: tuple[float, float] | None,
) -> dict:
    effective, envelope_note = _clip_ladder_to_envelope(constraints, inp, envelope)
    continuous = solve_continuous(inp, effective)
    milp = solve_milp(inp, effective)
    return {
        "ppg_id": inp.ppg_id,
        "model_kind": inp.model_kind,
        "base_price": inp.base_price,
        "competitor_price": inp.competitor_price,
        "objective": constraints.objective,
        "envelope_clip": envelope_note,
        "continuous": {
            "price_multiplier": continuous.price_multiplier,
            "price": continuous.price,
            "promo": continuous.promo,
            "units": continuous.units,
            "revenue": continuous.revenue,
            "margin": continuous.margin,
            "objective_value": continuous.objective_value,
            "bounds_used": list(continuous.bounds_used),
            "feasible": continuous.feasible,
        },
        "milp": {
            "price_multiplier": milp.price_multiplier,
            "price": milp.price,
            "promo": milp.promo,
            "units": milp.units,
            "revenue": milp.revenue,
            "margin": milp.margin,
            "objective_value": milp.objective_value,
            "feasible_strict": milp.feasible_strict,
            "relaxed": milp.relaxed,
            "binding_violations": milp.binding_violations,
            "n_cells_considered": milp.n_cells_considered,
            "n_cells_feasible": milp.n_cells_feasible,
            "chosen_slacks": milp.chosen_slacks,
        },
    }


class OptimizationAgent(Agent):
    name = "optimization"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        run_dir = Path(run.run_dir)
        feats = await asyncio.to_thread(_load_features, run_dir)
        modeling = await asyncio.to_thread(_load_modeling, run_dir)
        constraints = _constraints_from_options(run.options)

        constraints_path = run_dir / "optimization_constraints.json"
        constraints_path.write_text(json.dumps(constraints.to_dict(), indent=2))
        result.artifacts.append(
            ArtifactRef(
                path=str(constraints_path),
                mime="application/json",
                agent=self.name,
                name=constraints_path.name,
            )
        )

        per_ppg: list[dict] = []
        flat_table: list[dict] = []
        skipped: list[dict] = []

        controls_for_opt = modeling.get("controls_used", [])

        for row in modeling.get("per_ppg", []):
            ppg_id = row["ppg_id"]
            winner = row.get("winner_model")
            slice_ = feats[feats["ppg_id"] == ppg_id]
            if winner not in SUPPORTED_MODELS or slice_.empty:
                skipped.append(
                    {
                        "ppg_id": ppg_id,
                        "winner_model": winner,
                        "reason": (
                            "no rows for PPG"
                            if slice_.empty
                            else f"optimisation not supported for {winner}"
                        ),
                    }
                )
                continue
            if winner in SUPPORTED_OLS_MODELS:
                coefficients = row.get("winner", {}).get("coefficients", {})
                if not coefficients:
                    skipped.append(
                        {"ppg_id": ppg_id, "winner_model": winner, "reason": "no coefficients"}
                    )
                    continue

            inp = _build_inputs(ppg_id, slice_, row, controls_for_opt, winner)
            envelope = _training_price_envelope(slice_) if winner in TREE_MODELS else None
            payload = await asyncio.to_thread(_optimise_one, inp, constraints, envelope)
            per_ppg.append(payload)
            envelope_clipped = bool(
                payload.get("envelope_clip") and payload["envelope_clip"].get("ladder_clipped")
            )
            flat_table.append(
                {
                    "ppg_id": ppg_id,
                    "objective": constraints.objective,
                    "price_multiplier": payload["milp"]["price_multiplier"],
                    "price": payload["milp"]["price"],
                    "base_price": inp.base_price,
                    "promo": payload["milp"]["promo"],
                    "units": payload["milp"]["units"],
                    "revenue": payload["milp"]["revenue"],
                    "margin": payload["milp"]["margin"],
                    "feasible_strict": payload["milp"]["feasible_strict"],
                    "relaxed": payload["milp"]["relaxed"],
                    "envelope_clipped": envelope_clipped,
                    "model_kind": winner,
                }
            )
            await self.emit(
                run,
                "tool_called",
                {
                    "tool": "optimise_ppg",
                    "ppg_id": ppg_id,
                    "model_kind": winner,
                    "milp_multiplier": payload["milp"]["price_multiplier"],
                    "relaxed": payload["milp"]["relaxed"],
                    "envelope_clipped": envelope_clipped,
                    "n_feasible_cells": payload["milp"]["n_cells_feasible"],
                },
            )

        results_path = run_dir / "optimization_results.json"
        results_path.write_text(json.dumps(per_ppg, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(results_path),
                mime="application/json",
                agent=self.name,
                name=results_path.name,
            )
        )

        table_path = run_dir / "optimization_table.json"
        table_path.write_text(json.dumps(flat_table, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(table_path),
                mime="application/json",
                agent=self.name,
                name=table_path.name,
            )
        )

        headline, rationales = self._narrate(result, per_ppg)
        rationale_by_id = {r["ppg_id"]: r["rationale"] for r in rationales}
        for p in per_ppg:
            p["rationale"] = rationale_by_id.get(p["ppg_id"], "")

        n_relaxed = sum(1 for p in per_ppg if p["milp"]["relaxed"])
        n_envelope_clipped = sum(
            1
            for p in per_ppg
            if p.get("envelope_clip") and p["envelope_clip"].get("ladder_clipped")
        )
        result.outputs = {
            "n_optimised": len(per_ppg),
            "n_skipped": len(skipped),
            "n_relaxed": n_relaxed,
            "n_envelope_clipped": n_envelope_clipped,
            "objective": constraints.objective,
            "ladder_size": len(constraints.price_ladder),
        }
        if skipped:
            result.outputs["skipped"] = skipped
        result.reasoning = headline or (
            f"Optimised {len(per_ppg)} PPGs against a {len(constraints.price_ladder)}-rung "
            f"ladder; {n_relaxed} required constraint relaxation."
        )
        result.confidence = 0.85 if per_ppg and n_relaxed == 0 else (
            0.6 if per_ppg else 0.0
        )

    def _narrate(
        self, result: AgentResult, per_ppg: list[dict]
    ) -> tuple[str, list[dict]]:
        compact = [
            {
                "ppg_id": p["ppg_id"],
                "milp_multiplier": p["milp"]["price_multiplier"],
                "milp_promo": p["milp"]["promo"],
                "milp_revenue": p["milp"]["revenue"],
                "milp_margin": p["milp"]["margin"],
                "continuous_multiplier": p["continuous"]["price_multiplier"],
                "relaxed": p["milp"]["relaxed"],
                "binding_violations": p["milp"]["binding_violations"],
            }
            for p in per_ppg
        ]
        try:
            resp = self.call_llm(
                result,
                system=SYSTEM_PROMPT,
                user=json.dumps(compact, default=str),
                max_tokens=700,
                label="optimisation-rationale",
            )
            if resp.raw.get("dry_run"):
                return "", []
            blob = json.loads(resp.text)
            return str(blob.get("headline", "")), list(blob.get("per_ppg", []))
        except (json.JSONDecodeError, ValueError):
            return "", []
