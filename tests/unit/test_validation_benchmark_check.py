"""Test that the validation agent's benchmark augmentation is wired through.

The full ValidationAgent run requires rolling-CV fixtures we don't want to
maintain in unit tests; we validate the new behaviour by exercising the
two helpers the agent composes — ``_load_ppg_categories`` and
``classify(lookup_category(...))`` — on a realistic
``ppg_mapping_table.json`` shape.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.agents.validation import _load_ppg_categories
from core.benchmarks.elasticity import classify, lookup_category


def _write_mapping(run_dir: Path) -> None:
    rows = [
        {"ppg_id": "ppg_01", "sku": "sku_a", "category": "Soft Drinks", "brand": "cola"},
        {"ppg_id": "ppg_02", "sku": "sku_b", "category": "Yogurt",      "brand": "dannon"},
        {"ppg_id": "ppg_03", "sku": "sku_c", "category": "Beer",        "brand": "budweiser"},
        {"ppg_id": "ppg_04", "sku": "sku_d", "category": "Antimatter",  "brand": "acme"},
    ]
    (run_dir / "ppg_mapping_table.json").write_text(json.dumps(rows))


def test_load_categories_from_mapping(tmp_path: Path) -> None:
    _write_mapping(tmp_path)
    cats = _load_ppg_categories(tmp_path)
    assert cats == {
        "ppg_01": "Soft Drinks",
        "ppg_02": "Yogurt",
        "ppg_03": "Beer",
        "ppg_04": "Antimatter",
    }


def test_load_categories_missing_artefact(tmp_path: Path) -> None:
    # No mapping file — agent must degrade gracefully to no-benchmark for everyone.
    assert _load_ppg_categories(tmp_path) == {}


def test_benchmark_classification_per_ppg(tmp_path: Path) -> None:
    _write_mapping(tmp_path)
    cats = _load_ppg_categories(tmp_path)

    # ppg_01 -> Soft Drinks, mean -3.18, band [-4.13, -2.23]
    bench_01 = lookup_category(cats["ppg_01"])
    assert bench_01 is not None
    assert classify(-3.0, bench_01) == "in_band"
    assert classify(-5.5, bench_01) == "out_band_high"  # more elastic than band
    assert classify(-1.5, bench_01) == "out_band_low"   # less elastic than band

    # ppg_02 -> Yogurt, Bijmolt 2005 fallback
    bench_02 = lookup_category(cats["ppg_02"])
    assert bench_02 is not None
    assert bench_02.source == "bijmolt_2005"
    assert classify(bench_02.mean, bench_02) == "in_band"

    # ppg_03 -> Beer, Hoch 1995
    bench_03 = lookup_category(cats["ppg_03"])
    assert bench_03 is not None
    assert bench_03.source == "hoch_1995"

    # ppg_04 -> Antimatter, no match
    bench_04 = lookup_category(cats["ppg_04"])
    assert bench_04 is None
    assert classify(-2.0, bench_04) == "no_benchmark"


def test_summary_counts_against_mock_recoveries(tmp_path: Path) -> None:
    _write_mapping(tmp_path)
    cats = _load_ppg_categories(tmp_path)
    # Mock per-PPG elasticities the agent would have computed
    recovered = {"ppg_01": -3.0, "ppg_02": -2.4, "ppg_03": -10.0, "ppg_04": -2.5}

    statuses = {
        ppg_id: classify(recovered[ppg_id], lookup_category(cats[ppg_id]))
        for ppg_id in recovered
    }

    n_in = sum(1 for s in statuses.values() if s == "in_band")
    n_out = sum(1 for s in statuses.values() if s.startswith("out_band"))
    n_none = sum(1 for s in statuses.values() if s == "no_benchmark")

    assert n_in == 2          # ppg_01 (cola, in-band) + ppg_02 (yogurt, in-band)
    assert n_out == 1         # ppg_03 (beer at -10 is way more elastic than band)
    assert n_none == 1        # ppg_04 (Antimatter has no benchmark)
    assert n_in + n_out + n_none == len(statuses)
