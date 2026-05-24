"""POST /runs/{id}/approve accepts optional grain payload.

Bodyless approve still works; a JSON body with ``modelling_grain`` /
``comparison_grains`` validates the enum and reaches the gate registry.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from core.config import ModellingGrain
from core.orchestrator.gates import gate_registry


def _client() -> TestClient:
    return TestClient(app)


def test_approve_bodyless_works() -> None:
    client = _client()
    run_id = "test_run_bodyless"
    # Prime the gate so /approve has something to release.
    gate_registry.get(run_id, "ppg_mapping")
    resp = client.post(f"/runs/{run_id}/approve?agent=ppg_mapping")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved"
    assert "applied_options" not in body
    state = gate_registry.get(run_id, "ppg_mapping")
    assert state.approved is True
    assert state.approve_payload is None
    gate_registry.drop(run_id)


def test_approve_with_grain_payload() -> None:
    client = _client()
    run_id = "test_run_with_grain"
    gate_registry.get(run_id, "ppg_mapping")
    payload = {
        "modelling_grain": "store_ppg_week",
        "comparison_grains": ["brand_week", "category_week"],
    }
    resp = client.post(f"/runs/{run_id}/approve?agent=ppg_mapping", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied_options"]["modelling_grain"] == "store_ppg_week"
    assert body["applied_options"]["comparison_grains"] == ["brand_week", "category_week"]
    state = gate_registry.get(run_id, "ppg_mapping")
    assert state.approve_payload == body["applied_options"]
    gate_registry.drop(run_id)


def test_approve_rejects_unknown_grain() -> None:
    client = _client()
    run_id = "test_run_bad_grain"
    gate_registry.get(run_id, "ppg_mapping")
    resp = client.post(
        f"/runs/{run_id}/approve?agent=ppg_mapping",
        json={"modelling_grain": "ppg_per_hour"},  # not in the enum
    )
    assert resp.status_code == 422
    gate_registry.drop(run_id)


def test_approve_rejects_unknown_fields() -> None:
    client = _client()
    run_id = "test_run_extra_field"
    gate_registry.get(run_id, "ppg_mapping")
    resp = client.post(
        f"/runs/{run_id}/approve?agent=ppg_mapping",
        json={"modelling_grain": "ppg_week", "rogue_flag": True},
    )
    # Pydantic with extra=forbid rejects unknown keys.
    assert resp.status_code == 422
    gate_registry.drop(run_id)


def test_approve_dedupes_comparison_grains() -> None:
    client = _client()
    run_id = "test_run_dedupe"
    gate_registry.get(run_id, "ppg_mapping")
    resp = client.post(
        f"/runs/{run_id}/approve?agent=ppg_mapping",
        json={"comparison_grains": ["brand_week", "brand_week", "category_week"]},
    )
    assert resp.status_code == 200
    assert resp.json()["applied_options"]["comparison_grains"] == ["brand_week", "category_week"]
    gate_registry.drop(run_id)


def test_known_grain_values_match_enum() -> None:
    # Sanity guard so the test suite catches if the enum drifts from
    # the values the API claims to accept.
    expected = {
        "ppg_week",
        "store_ppg_week",
        "category_week",
        "store_category_week",
        "brand_week",
        "store_brand_week",
    }
    assert {g.value for g in ModellingGrain} == expected
