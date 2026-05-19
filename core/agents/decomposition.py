"""Decomposition agent.

For each PPG whose winning model in the modelling stage is OLS-based,
refits the winner on the full feature frame (no train/test split — we
want all observed weeks attributed) and decomposes every observed week
into ``base + due-to-driver_i + residual``. Aggregates per-feature
contributions into business-friendly groups (price / promo /
distribution / seasonality / competitor / lags / other) for the UI
table.

LightGBM-winning PPGs are skipped with a structured note: closed-form
decomposition isn't applicable and full-frame ablation will land in a
follow-up. The pipeline keeps moving — simulation + optimisation can
still operate on the modelling output directly.

Outputs (one file each, every one a JSON):
- ``decomposition_per_ppg_week.json`` — weekly grid per PPG, every
  driver's unit contribution + residual + reconciliation flag.
- ``decomposition_summary.json`` — per-PPG totals + per-feature +
  per-group contributions + reconciliation diagnostic.
- ``decomposition_table.json`` — flat ``(ppg_id, group, units,
  share_of_lift)`` rows; the canonical shape the UI's shared
  ``<ResultsTable>`` renders.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from core.agents.base import Agent
from core.decomp.due_to import (
    aggregate_to_groups,
    decompose_ols_frame,
    summarise_ppg,
)
from core.decomp.groups import FEATURE_TO_GROUP, GROUP_ORDER, group_for
from core.models.loglog_ols import fit_loglog
from core.models.semilog_ols import fit_semilog
from core.orchestrator.state import AgentResult, ArtifactRef, RunState


SYSTEM_PROMPT = """You are the demand-decomposition analyst. You receive
per-PPG totals split into base + due-to-{price, promo, distribution,
seasonality, competitor, lags, other} unit contributions, plus a
reconciliation_pct_error per PPG. Return STRICT JSON of the form:
{"headline": "<=240 chars across-PPG summary citing which drivers
explain most of the lift",
 "per_ppg": [{"ppg_id": "...", "rationale": "<=160 chars on the dominant
driver(s) and why"}]}
JSON only. Cite only PPGs that appear in the input."""


SUPPORTED_OLS_MODELS = {"loglog_ols", "semilog_ols"}


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


def _load_modeling(run_dir: Path) -> dict:
    path = run_dir / "modeling_results.json"
    if not path.exists():
        raise RuntimeError("modeling_results.json missing — modeling must run first")
    return json.loads(path.read_text())


def _refit_for_decomp(model_kind: str, ppg_id: str, frame: pd.DataFrame, controls: list[str]):
    """Refit the winning OLS family on the FULL frame (no holdout).

    Returns the coefficients dict the closed-form decomposition needs.
    """
    if model_kind == "loglog_ols":
        return fit_loglog(ppg_id, frame, controls).coefficients
    if model_kind == "semilog_ols":
        return fit_semilog(ppg_id, frame, controls).coefficients
    raise ValueError(f"closed-form decomposition not implemented for {model_kind}")


def _decompose_one_ppg(
    ppg_id: str, frame: pd.DataFrame, controls: list[str], model_kind: str
) -> tuple[pd.DataFrame, dict]:
    coefs = _refit_for_decomp(model_kind, ppg_id, frame, controls)
    weekly = decompose_ols_frame(frame, coefs)
    features = [c for c in coefs if c != "const"]
    weekly = aggregate_to_groups(weekly, features, FEATURE_TO_GROUP)
    summary = summarise_ppg(weekly, features, FEATURE_TO_GROUP)
    summary["model_kind"] = model_kind
    summary["coefficients"] = {k: float(v) for k, v in coefs.items()}
    return weekly, summary


class DecompositionAgent(Agent):
    name = "decomposition"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        run_dir = Path(run.run_dir)
        feats = await asyncio.to_thread(_load_features, run_dir)
        modeling = await asyncio.to_thread(_load_modeling, run_dir)
        controls_for_decomp = modeling.get("controls_used", [])

        per_ppg_weekly: list[dict] = []
        per_ppg_summary: list[dict] = []
        flat_table: list[dict] = []
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
                            else f"closed-form decomposition not supported for {winner}"
                        ),
                    }
                )
                continue

            weekly, summary = await asyncio.to_thread(
                _decompose_one_ppg,
                ppg_id,
                slice_,
                controls_for_decomp,
                winner,
            )
            await self.emit(
                run,
                "tool_called",
                {
                    "tool": "decompose_ppg",
                    "ppg_id": ppg_id,
                    "model_kind": winner,
                    "reconciliation_pct_error": round(summary["reconciliation_pct_error"], 6),
                },
            )

            week_col = "week_start" if "week_start" in slice_.columns else None
            weekly_rows: list[dict] = []
            for idx, frame_row in weekly.reset_index(drop=True).iterrows():
                wk = (
                    str(slice_[week_col].iloc[int(idx)])
                    if week_col
                    else f"row_{int(idx)}"
                )
                weekly_rows.append(
                    {
                        "week_start": wk,
                        "observed": float(frame_row.get("observed", float("nan"))),
                        "predicted": float(frame_row["predicted"]),
                        "base": float(frame_row["base"]),
                        "residual": float(frame_row.get("residual", float("nan"))),
                        "due_by_group": {
                            grp: float(frame_row.get(f"due_group_{grp}", 0.0))
                            for grp in GROUP_ORDER
                        },
                    }
                )
            per_ppg_weekly.append({"ppg_id": ppg_id, "weekly": weekly_rows})

            summary["ppg_id"] = ppg_id
            per_ppg_summary.append(summary)

            for group in GROUP_ORDER:
                units = float(summary["per_group_units"].get(group, 0.0))
                share = float(summary["per_group_share"].get(group, 0.0))
                if units == 0.0 and share == 0.0 and group != "price":
                    continue
                flat_table.append(
                    {
                        "ppg_id": ppg_id,
                        "group": group,
                        "due_units": units,
                        "share_of_lift": share,
                        "total_lift": float(summary["total_lift"]),
                        "model_kind": winner,
                    }
                )

        weekly_path = run_dir / "decomposition_per_ppg_week.json"
        weekly_path.write_text(json.dumps(per_ppg_weekly, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(weekly_path),
                mime="application/json",
                agent=self.name,
                name=weekly_path.name,
            )
        )

        summary_path = run_dir / "decomposition_summary.json"
        summary_path.write_text(json.dumps(per_ppg_summary, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(summary_path),
                mime="application/json",
                agent=self.name,
                name=summary_path.name,
            )
        )

        table_path = run_dir / "decomposition_table.json"
        table_path.write_text(json.dumps(flat_table, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(table_path),
                mime="application/json",
                agent=self.name,
                name=table_path.name,
            )
        )

        headline, rationales = self._narrate(result, per_ppg_summary)
        rationale_by_id = {r["ppg_id"]: r["rationale"] for r in rationales}
        for s in per_ppg_summary:
            s["rationale"] = rationale_by_id.get(s["ppg_id"], "")

        max_recon = max(
            (abs(s["reconciliation_pct_error"]) for s in per_ppg_summary),
            default=0.0,
        )
        result.outputs = {
            "n_decomposed": len(per_ppg_summary),
            "n_skipped": len(skipped),
            "max_reconciliation_pct_error": round(max_recon, 6),
            "groups_covered": list(GROUP_ORDER),
        }
        if skipped:
            result.outputs["skipped"] = skipped
        result.reasoning = headline or (
            f"Decomposed {len(per_ppg_summary)} PPGs into base + driver lifts; "
            f"max reconciliation error {max_recon * 100:.3f}%."
        )
        result.confidence = 1.0 if max_recon < 0.01 else max(0.0, 1.0 - max_recon)

    def _narrate(
        self, result: AgentResult, summaries: list[dict]
    ) -> tuple[str, list[dict]]:
        # Compact payload — we don't need to send weekly rows to the LLM.
        compact = [
            {
                "ppg_id": s["ppg_id"],
                "total_predicted": s["total_predicted"],
                "total_base": s["total_base"],
                "per_group_units": s["per_group_units"],
                "per_group_share": s["per_group_share"],
                "reconciliation_pct_error": s["reconciliation_pct_error"],
            }
            for s in summaries
        ]
        try:
            resp = self.call_llm(
                result,
                system=SYSTEM_PROMPT,
                user=json.dumps(compact, default=str),
                max_tokens=700,
                label="decomp-rationale",
            )
            if resp.raw.get("dry_run"):
                return "", []
            blob = json.loads(resp.text)
            return str(blob.get("headline", "")), list(blob.get("per_ppg", []))
        except (json.JSONDecodeError, ValueError):
            return "", []
