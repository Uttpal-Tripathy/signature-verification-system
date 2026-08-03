"""API-layer tests. These exist because api/app.py had zero test coverage and its
/verify route depends on python-multipart being installed for FastAPI's File()/Form()
parameters -- an import-time RuntimeError that no unit test caught until the app was
actually started for the first time. TestClient's `with` block runs the real FastAPI
lifespan (startup/shutdown), so it reproduces that failure mode.
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from sigverify.data.synthetic import generate_synthetic_signature_image


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGVERIFY_CONFIG", "configs/lightweight_real.yaml")
    # Nonexistent checkpoint dir -> the bundle falls back to randomly initialized
    # weights (SignatureVerificationBundle's documented behavior), keeping this test
    # fast and independent of any trained checkpoint being present.
    monkeypatch.setenv("SIGVERIFY_CHECKPOINTS", str(tmp_path / "no_checkpoints_here"))

    from api.app import app

    with TestClient(app) as test_client:
        yield test_client


def _png_bytes(seed: int) -> bytes:
    image = generate_synthetic_signature_image("writer_000", seed, forged=False)
    import cv2

    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["device"] in ("cpu", "cuda")


def test_root_serves_console(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "SIGNUM" in resp.text


def test_verify_static_only(client):
    resp = client.post(
        "/api/verify",
        files={
            "reference_image": ("reference.png", io.BytesIO(_png_bytes(1)), "image/png"),
            "query_image": ("query.png", io.BytesIO(_png_bytes(2)), "image/png"),
        },
        data={"estimate_confidence": "false"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["decision"] in ("Genuine", "Review", "Forged")
    assert 0.0 <= body["combined_score"] <= 1.0001
    assert body["dynamic_similarity"] is None


def test_verify_rejects_missing_query_image(client):
    resp = client.post(
        "/api/verify",
        files={"reference_image": ("reference.png", io.BytesIO(_png_bytes(1)), "image/png")},
    )
    assert resp.status_code == 422  # FastAPI validation error, not a 500


def test_audit_chain_endpoint_reflects_a_verification(client):
    client.post(
        "/api/verify",
        files={
            "reference_image": ("reference.png", io.BytesIO(_png_bytes(1)), "image/png"),
            "query_image": ("query.png", io.BytesIO(_png_bytes(2)), "image/png"),
        },
        data={"estimate_confidence": "false"},
    )
    resp = client.get("/api/audit/verify_chain")
    assert resp.status_code == 200
    assert resp.json()["valid"] is True
