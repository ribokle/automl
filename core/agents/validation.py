"""Validation agent.

Runs rolling-origin (expanding-window) cross-validation on the winning
model family from the modelling stage (log-log OLS, semi-log OLS, or
LightGBM), then produces a per-PPG verdict (pass / warn / fail) over
four checks: sign stability across folds, mean hold-out WAPE,
elasticity coefficient-of-variation, and plausibility-band magnitude.

PPGs with no usable winner are skipped with a structured note.

Outputs:

- ``validation_report.json`` — per-PPG verdicts + per-fold details +
  thresholds.
- ``validation_table.json`` — flat one-row-per-PPG table the UI's
  shared ``<ResultsTable>`` renders.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd

from core.agents.base import Agent
from core.benchmarks.elasticity import (
    classify as classify_benchmark,
    load_elasticity_benchmarks,
)
from core.config import get_settings
from core.features.engineering import ENGINEERED_COLUMNS, TARGET
from core.orchestrator.state import AgentResult, ArtifactRef, RunState
from core.validation.checks import Verdict, evaluate_ppg
from core.validation.rolling import build_folds, fit_one_fold

SYSTEM_PROMPT = """You are the model-validation reviewer. You receive a
per-PPG table of rolling-origin cross-validation results: mean hold-out
WAPE across k folds, elasticity coefficient-of-variation, sign
stability, and a deterministic verdict (pass / warn / fail). For each
PPG, return one sentence (<=160 chars) explaining WHY the verdict came
out the way it did. Return STRICT JSON:
{"per_ppg": [{"ppg_id": "...", "rationale": "..."}, ...],
 "headline": "<=240 chars exec summary"}
JSON only, no prose. Cite only PPGs present in the input."""


SUPPORTED_MODELS = {"loglog_ols", "semilog_ols", "lightgbm"}
DEFAULT_N_FOLDS = 4


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


def _load_controls(run_dir: Path) -> list[str]:
    refine_path = run_dir / "feature_refine.json"
    if refine_path.exists():
        blob = json.loads(refine_path.read_text())
        kept = blob.get("kept") or []
        return [c for c in kept if c not in (TARGET, "log_price")]
    return [c for c in ENGINEERED_COLUMNS if c not in (TARGET, "log_price")]


def _load_ppg_categories(run_dir: Path) -> dict[str, str]:
    """Read ppg_mapping_table.json -> {ppg_id: category}.

    Returns an empty dict if the mapping artefact is missing; the
    validation agent then records ``benchmark_status="no_benchmark"`` for
    every PPG, which is the safe default.
    """
    path = run_dir / "ppg_mapping_table.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    out: dict[str, str] = {}
    for row in rows:
        ppg_id = row.get("ppg_id")
        cat = row.get("category")
        if ppg_id and cat and ppg_id not in out:
            out[ppg_id] = str(cat)
    return out


def _collect_residuals(per_ppg: list[dict]) -> list[dict]:
    """Flatten per-fold hold-out residuals into one per-PPG bag.

    The histogram is built browser-side from the concatenated test
    residuals across every fold; per-fold tagging is preserved so the
    UI can colour the bars by fold if desired.
    """
    rows: list[dict] = []
    for p in per_ppg:
        residuals: list[float] = []
        folds: list[dict] = []
        for f in p.get("folds") or []:
            r = f.get("test_residuals_log") or []
            residuals.extend(float(v) for v in r)
            folds.append({"fold": f.get("fold"), "n": len(r)})
        if not residuals:
            continue
        rows.append(
            {
                "ppg_id": p["ppg_id"],
                "winner_model": p.get("winner_model"),
                "verdict": p.get("verdict"),
                "residuals_log": residuals,
                "folds": folds,
            }
        )
    return rows


def _validate_one(
    ppg_id: str,
    slice_: pd.DataFrame,
    controls: list[str],
    model_kind: str,
    n_folds: int,
) -> tuple[Verdict, list[dict]]:
    folds = build_folds(slice_, n_folds=n_folds)
    fold_results = [
        fit_one_fold(ppg_id, f, controls, model_kind=model_kind) for f in folds
    ]
    verdict = evaluate_ppg(ppg_id, fold_results)
    return verdict, fold_results


class ValidationAgent(Agent):
    name = "validation"

    async def _execute(self, run: RunState, result: AgentResult) -> None:
        run_dir = Path(run.run_dir)
        feats = await asyncio.to_thread(_load_features, run_dir)
        modeling = await asyncio.to_thread(_load_modeling, run_dir)
        controls = _load_controls(run_dir)
        ppg_categories = _load_ppg_categories(run_dir)
        benchmarks = load_elasticity_benchmarks()

        n_folds = int(
            (run.options.get("validation") or {}).get("n_folds", DEFAULT_N_FOLDS)
        )

        per_ppg: list[dict] = []
        flat_table: list[dict] = []
        skipped: list[dict] = []

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
                            else f"validation not supported for {winner}"
                        ),
                    }
                )
                continue

            verdict, fold_results = await asyncio.to_thread(
                _validate_one, ppg_id, slice_, controls, winner, n_folds
            )
            per_ppg.append(
                {
                    "ppg_id": ppg_id,
                    "winner_model": winner,
                    "verdict": verdict.verdict,
                    "sign_stability": verdict.sign_stability,
                    "wape_mean": verdict.wape_mean,
                    "wape_std": verdict.wape_std,
                    "elasticity_mean": verdict.elasticity_mean,
                    "elasticity_std": verdict.elasticity_std,
                    "elasticity_cv": verdict.elasticity_cv,
                    "n_folds": verdict.n_folds,
                    "checks": verdict.checks,
                    "folds": fold_results,
                }
            )
            category = ppg_categories.get(ppg_id)
            bench = benchmarks.lookup(category)
            bench_status = classify_benchmark(verdict.elasticity_mean, bench)
            flat_table.append(
                {
                    "ppg_id": ppg_id,
                    "category": category,
                    "winner": winner,
                    "verdict": verdict.verdict,
                    "sign_stability": verdict.sign_stability,
                    "wape_mean": verdict.wape_mean,
                    "elasticity_mean": verdict.elasticity_mean,
                    "elasticity_cv": verdict.elasticity_cv,
                    "n_folds": verdict.n_folds,
                    "benchmark_status": bench_status,
                    "benchmark_mean": bench.mean if bench else None,
                    "benchmark_low": bench.lo if bench else None,
                    "benchmark_high": bench.hi if bench else None,
                    "benchmark_source": bench.source if bench else None,
                    "benchmark_category": bench.display_name if bench else None,
                }
            )
            await self.emit(
                run,
                "tool_called",
                {
                    "tool": "rolling_cv",
                    "ppg_id": ppg_id,
                    "n_folds": verdict.n_folds,
                    "wape_mean": (
                        round(verdict.wape_mean, 4)
                        if verdict.wape_mean == verdict.wape_mean  # not NaN
                        else None
                    ),
                    "verdict": verdict.verdict,
                },
            )

        n_pass = sum(1 for p in per_ppg if p["verdict"] == "pass")
        n_warn = sum(1 for p in per_ppg if p["verdict"] == "warn")
        n_fail = sum(1 for p in per_ppg if p["verdict"] == "fail")

        n_in_benchmark = sum(1 for r in flat_table if r["benchmark_status"] == "in_band")
        n_out_benchmark = sum(
            1 for r in flat_table if r["benchmark_status"] in {"out_band_low", "out_band_high"}
        )
        n_no_benchmark = sum(1 for r in flat_table if r["benchmark_status"] == "no_benchmark")
        n_with_benchmark = n_in_benchmark + n_out_benchmark
        benchmark_pass_rate = (
            round(n_in_benchmark / n_with_benchmark, 3) if n_with_benchmark else None
        )

        headline, rationales = self._narrate(result, per_ppg)
        rationale_by_id = {r["ppg_id"]: r["rationale"] for r in rationales}
        for p in per_ppg:
            p["rationale"] = rationale_by_id.get(p["ppg_id"], "")
        for row in flat_table:
            row["rationale"] = rationale_by_id.get(row["ppg_id"], "")

        _t = get_settings().validation
        report_blob = {
            "thresholds": {
                "sign_pass": _t.sign_pass,
                "sign_warn": _t.sign_warn,
                "wape_pass": _t.wape_pass,
                "wape_warn": _t.wape_warn,
                "cv_pass": _t.cv_pass,
                "cv_warn": _t.cv_warn,
                "elasticity_low": _t.elasticity_low,
                "elasticity_high": _t.elasticity_high,
            },
            "n_folds": n_folds,
            "per_ppg": per_ppg,
            "headline": headline,
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
            "benchmark": {
                "n_in_band": n_in_benchmark,
                "n_out_band": n_out_benchmark,
                "n_no_benchmark": n_no_benchmark,
                "pass_rate": benchmark_pass_rate,
                "grand_mean": benchmarks.grand_mean,
                "sources": [s.get("key") for s in benchmarks.sources],
            },
        }
        if skipped:
            report_blob["skipped"] = skipped

        report_path = run_dir / "validation_report.json"
        report_path.write_text(json.dumps(report_blob, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(report_path),
                mime="application/json",
                agent=self.name,
                name=report_path.name,
            )
        )

        table_path = run_dir / "validation_table.json"
        table_path.write_text(json.dumps(flat_table, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(table_path),
                mime="application/json",
                agent=self.name,
                name=table_path.name,
            )
        )

        residuals_blob = _collect_residuals(per_ppg)
        residuals_path = run_dir / "validation_residuals.json"
        residuals_path.write_text(json.dumps(residuals_blob, indent=2, default=str))
        result.artifacts.append(
            ArtifactRef(
                path=str(residuals_path),
                mime="application/json",
                agent=self.name,
                name=residuals_path.name,
            )
        )

        result.outputs = {
            "n_validated": len(per_ppg),
            "n_skipped": len(skipped),
            "n_pass": n_pass,
            "n_warn": n_warn,
            "n_fail": n_fail,
            "n_folds": n_folds,
            "n_in_benchmark": n_in_benchmark,
            "n_out_benchmark": n_out_benchmark,
            "n_no_benchmark": n_no_benchmark,
            "benchmark_pass_rate": benchmark_pass_rate,
        }
        if skipped:
            result.outputs["skipped"] = skipped
        result.reasoning = headline or (
            f"Rolling-origin CV ({n_folds} folds): {n_pass} pass, {n_warn} warn, "
            f"{n_fail} fail across {len(per_ppg)} PPGs."
        )
        result.confidence = (n_pass / len(per_ppg)) if per_ppg else 0.0

    def _narrate(
        self, result: AgentResult, per_ppg: list[dict]
    ) -> tuple[str, list[dict]]:
        compact = [
            {
                "ppg_id": p["ppg_id"],
                "verdict": p["verdict"],
                "sign_stability": p["sign_stability"],
                "wape_mean": p["wape_mean"],
                "elasticity_mean": p["elasticity_mean"],
                "elasticity_cv": p["elasticity_cv"],
                "n_folds": p["n_folds"],
            }
            for p in per_ppg
        ]
        try:
            resp = self.call_llm(
                result,
                system=SYSTEM_PROMPT,
                user=json.dumps(compact, default=str),
                max_tokens=700,
                label="validation-narrative",
            )
            if resp.raw.get("dry_run"):
                return "", []
            blob = json.loads(resp.text)
            return str(blob.get("headline", "")), list(blob.get("per_ppg", []))
        except (json.JSONDecodeError, ValueError):
            return "", []
