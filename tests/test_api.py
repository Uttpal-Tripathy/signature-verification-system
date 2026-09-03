"""API-layer tests. These exist because api/app.py had zero test coverage and its
/verify route depends on python-multipart being installed for FastAPI's File()/Form()
parameters -- an import-time RuntimeError that no unit test caught until the app was
actually started for the first time. TestClient's `with` block runs the real FastAPI
lifespan (startup/shutdown), so it reproduces that failure mode.
"""
from __future__ import annotations

import io
import json

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
    # Deterministically disable API-key auth for the default fixture, regardless of
    # whatever a developer's local .env happens to set -- api/app.py's load_dotenv()
    # never overrides an already-set variable, so setting it here (even to "") wins.
    # Tests that specifically exercise auth override this again with monkeypatch.
    monkeypatch.setenv("SIGVERIFY_API_KEY", "")

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


def test_verify_response_includes_alert_classification(client):
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
    assert body["alert_severity"] in ("info", "warning", "critical")
    assert isinstance(body["alert_message"], str) and body["alert_message"]


def test_alerts_recent_endpoint_reflects_verifications(client):
    client.post(
        "/api/verify",
        files={
            "reference_image": ("reference.png", io.BytesIO(_png_bytes(1)), "image/png"),
            "query_image": ("query.png", io.BytesIO(_png_bytes(2)), "image/png"),
        },
        data={"estimate_confidence": "false"},
    )
    resp = client.get("/api/alerts/recent")
    assert resp.status_code == 200
    alerts = resp.json()
    assert len(alerts) >= 1
    assert alerts[-1]["severity"] in ("info", "warning", "critical")


def _stroke_bytes(num_points: int, seed: int) -> bytes:
    import numpy as np

    rng = np.random.default_rng(seed)
    stroke = {
        "x": rng.uniform(0, 300, num_points).tolist(),
        "y": rng.uniform(0, 100, num_points).tolist(),
        "timestamp": np.linspace(0, 2000, num_points).tolist(),
        "pressure": rng.uniform(0.3, 1.0, num_points).tolist(),
        "tilt_x": rng.uniform(-30, 30, num_points).tolist(),
        "tilt_y": rng.uniform(-30, 30, num_points).tolist(),
    }
    return json.dumps(stroke).encode("utf-8")


def test_verify_live_returns_similarity_for_partial_stroke(client):
    resp = client.post(
        "/api/verify/live",
        files={
            "reference_stroke": ("reference.json", io.BytesIO(_stroke_bytes(80, 1)), "application/json"),
            "query_stroke": ("query.json", io.BytesIO(_stroke_bytes(15, 2)), "application/json"),
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0.0 <= body["dynamic_similarity"] <= 1.0001
    assert body["num_points"] == 15


def test_verify_live_rejects_too_few_points(client):
    resp = client.post(
        "/api/verify/live",
        files={
            "reference_stroke": ("reference.json", io.BytesIO(_stroke_bytes(80, 1)), "application/json"),
            "query_stroke": ("query.json", io.BytesIO(_stroke_bytes(2, 2)), "application/json"),
        },
    )
    assert resp.status_code == 400


def test_alerts_websocket_streams_new_verification(client):
    with client.websocket_connect("/ws/alerts") as ws:
        client.post(
            "/api/verify",
            files={
                "reference_image": ("reference.png", io.BytesIO(_png_bytes(1)), "image/png"),
                "query_image": ("query.png", io.BytesIO(_png_bytes(2)), "image/png"),
            },
            data={"estimate_confidence": "false"},
        )
        message = ws.receive_json()
        assert message["severity"] in ("info", "warning", "critical")
        assert "decision" in message


# ---------------------------------------------------------------- API-key auth
# `require_api_key` reads SIGVERIFY_API_KEY dynamically on every request (not
# once at app startup), so these reuse the same `client` fixture/app instance
# and just monkeypatch the env var per test -- no separate fixture needed.

def test_api_key_disabled_by_default_needs_no_header(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_api_key_rejects_request_with_no_header_when_configured(client, monkeypatch):
    monkeypatch.setenv("SIGVERIFY_API_KEY", "test-secret-key")
    resp = client.get("/api/health")
    assert resp.status_code == 401


def test_api_key_rejects_wrong_header_when_configured(client, monkeypatch):
    monkeypatch.setenv("SIGVERIFY_API_KEY", "test-secret-key")
    resp = client.get("/api/health", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_api_key_accepts_correct_header_when_configured(client, monkeypatch):
    monkeypatch.setenv("SIGVERIFY_API_KEY", "test-secret-key")
    resp = client.get("/api/health", headers={"X-API-Key": "test-secret-key"})
    assert resp.status_code == 200


def test_api_key_protects_verify_endpoint_too(client, monkeypatch):
    monkeypatch.setenv("SIGVERIFY_API_KEY", "test-secret-key")
    resp = client.post(
        "/api/verify",
        files={
            "reference_image": ("reference.png", io.BytesIO(_png_bytes(1)), "image/png"),
            "query_image": ("query.png", io.BytesIO(_png_bytes(2)), "image/png"),
        },
        data={"estimate_confidence": "false"},
    )
    assert resp.status_code == 401


def test_websocket_rejects_missing_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("SIGVERIFY_API_KEY", "test-secret-key")
    # Starlette raises WebSocketDisconnect when the server closes before accept().
    with pytest.raises(Exception), client.websocket_connect("/ws/alerts"):  # noqa: B017
        pass


def test_websocket_accepts_correct_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("SIGVERIFY_API_KEY", "test-secret-key")
    with client.websocket_connect("/ws/alerts?api_key=test-secret-key"):
        pass  # connecting without raising is the assertion
