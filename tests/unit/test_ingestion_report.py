from __future__ import annotations

from core.data.ingestion_report import CheckResult, IngestionReport, Severity


def test_unified_report_categorizes():
    r = IngestionReport(
        duckdb_path="/tmp/x.duckdb",
        table="panel",
        row_count=100,
        dbt=[
            CheckResult(source="dbt", name="not_null_sku", status="pass", severity=Severity.info),
            CheckResult(source="dbt", name="weekly_continuity", status="fail", severity=Severity.warn,
                        message="gaps"),
        ],
        ge=[
            CheckResult(source="ge", name="price_mean", status="fail", severity=Severity.warn),
        ],
    )
    assert r.ok is True
    assert len(r.warnings) == 2
    assert len(r.errors) == 0
