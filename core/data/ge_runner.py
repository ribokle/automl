"""Great Expectations runner using an Ephemeral context.

Connects to the dbt-built `panel` mart in DuckDB by loading into a pandas
DataFrame, then validates against the suites defined in `core.data.expectations`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.data.expectations import all_expectations
from core.data.ingestion_report import CheckResult, Severity


def _load_panel(duckdb_path: Path, table: str) -> pd.DataFrame:
    import duckdb

    # Don't pass `read_only` — dbt's adapter may hold a writable connection on
    # the same file, and DuckDB rejects mixed-mode connections.
    con = duckdb.connect(str(duckdb_path))
    try:
        return con.execute(f"select * from main.{table}").df()
    finally:
        con.close()


def _run_with_great_expectations(df: pd.DataFrame) -> list[CheckResult]:
    try:
        import great_expectations as gx
        from great_expectations.core.expectation_suite import ExpectationSuite
    except Exception as exc:  # pragma: no cover
        return [
            CheckResult(
                source="ge",
                name="ge_import",
                status="fail",
                severity=Severity.error,
                message=f"failed to import great_expectations: {exc}",
            )
        ]

    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(name="panel_ds")
    asset = datasource.add_dataframe_asset(name="panel_asset")
    batch_def = asset.add_batch_definition_whole_dataframe(name="all")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    results: list[CheckResult] = []
    for suite_name, expectations in all_expectations().items():
        suite = ExpectationSuite(name=suite_name)
        for exp in expectations:
            suite.add_expectation(exp)
        validation_result = batch.validate(suite)
        for r in validation_result["results"]:
            exp_type = r["expectation_config"]["type"]
            success = bool(r["success"])
            results.append(
                CheckResult(
                    source="ge",
                    name=f"{suite_name}.{exp_type}",
                    status="pass" if success else "fail",
                    severity=Severity.warn if not success else Severity.info,
                    message=str(r.get("result", {}).get("observed_value", ""))[:200],
                )
            )
    return results


def run_ge_checks(duckdb_path: Path, table: str = "panel") -> list[CheckResult]:
    df = _load_panel(duckdb_path, table)
    return _run_with_great_expectations(df)


def capture_baseline(duckdb_path: Path, out_json: Path, table: str = "panel") -> dict[str, Any]:
    """Write a minimal baseline distribution snapshot for future drift checks."""
    import json

    df = _load_panel(duckdb_path, table)
    baseline: dict[str, Any] = {"row_count": int(len(df)), "columns": {}}
    for col in ("price", "units", "distribution_acv"):
        if col in df.columns:
            s = df[col].dropna()
            baseline["columns"][col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "q25": float(s.quantile(0.25)),
                "q50": float(s.quantile(0.50)),
                "q75": float(s.quantile(0.75)),
                "q95": float(s.quantile(0.95)),
            }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(baseline, indent=2))
    return baseline
