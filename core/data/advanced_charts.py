"""Chart-spec builders behind the advanced-EDA operator dashboard.

These functions transform the raw artefacts produced by ``core.features.advanced_eda``
into shapes the ECharts components can render directly — no further reshape
in TypeScript.

Builders are pure functions that follow the same graceful-degradation pattern
as ``core.data.charts``: missing inputs return an empty / placeholder shape
rather than raising.
"""
from __future__ import annotations

from typing import Any


def stl_chart(diag: dict[str, Any]) -> dict[str, Any]:
    """Pack STL decomposition into 4 series the ECharts component expects."""
    stl = diag.get("stl") or {}
    if not stl.get("available"):
        return {"available": False, "reason": stl.get("reason", "unavailable")}
    return {
        "available": True,
        "ppg_id": diag["ppg_id"],
        "weeks": stl["weeks"],
        "observed": stl["observed"],
        "trend": stl["trend"],
        "seasonal": stl["seasonal"],
        "resid": stl["resid"],
        "seasonal_amplitude": stl.get("seasonal_amplitude"),
        "trend_range": stl.get("trend_range"),
    }


def acf_chart(diag: dict[str, Any]) -> dict[str, Any]:
    """Pack ACF/PACF into a single chart spec with lags + confidence band."""
    ac = diag.get("acf_pacf") or {}
    if not ac.get("available"):
        return {"available": False, "reason": ac.get("reason", "unavailable")}
    return {
        "available": True,
        "ppg_id": diag["ppg_id"],
        "lags": ac["lags"],
        "acf": ac["acf"],
        "pacf": ac["pacf"],
        "ci": ac["ci"],
    }


def anomaly_timeline(panel_weeks: list[str], anomalies: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-PPG timeline scatter: each anomaly placed at its week with type +
    severity. Front-end overlays this on the existing weekly trend chart."""
    by_ppg: dict[str, list[dict[str, Any]]] = {}
    for a in anomalies:
        by_ppg.setdefault(a["ppg_id"], []).append(
            {
                "week_start": a["week_start"],
                "anomaly_type": a["anomaly_type"],
                "severity": a.get("severity", 0.0),
                "note": a.get("note", ""),
            }
        )
    return {"weeks": panel_weeks, "by_ppg": by_ppg}


def cross_ppg_heatmap(cross: dict[str, Any]) -> dict[str, Any]:
    """Cross-PPG correlation as a per-category list of heatmap blobs."""
    return {"categories": cross.get("categories", [])}


def lorenz_curve_spec(pareto: dict[str, Any], dimension: str) -> dict[str, Any]:
    """Lorenz curve for a given dimension (sku / brand / store), with both
    volume and revenue series."""
    dim = pareto.get(dimension) or {}
    return {
        "dimension": dimension,
        "volume": dim.get("volume", {"x": [], "cum_share": [], "labels": []}),
        "revenue": dim.get("revenue", {"x": [], "cum_share": [], "labels": []}),
    }


def price_ladder_chart(ladder_blob: dict[str, Any]) -> dict[str, Any]:
    """One PPG's ladder, prepped for a scatter + bar-frequency chart."""
    return {
        "ppg_id": ladder_blob["ppg_id"],
        "ladder": ladder_blob.get("ladder", []),
        "price_volume_slope": ladder_blob.get("price_volume_slope"),
        "log_intercept": ladder_blob.get("log_intercept"),
        "caveat": ladder_blob.get("caveat", ""),
    }


def promo_calendar_chart(calendar: dict[str, Any]) -> dict[str, Any]:
    return {
        "weeks": calendar.get("weeks", []),
        "ppgs": calendar.get("ppgs", []),
        "matrix": calendar.get("matrix", []),
    }
