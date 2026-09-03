"""FastAPI service exposing the signature verification pipeline over HTTP, and
serving the SIGNUM web console (web/) as the root static site.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
    # or: python api/app.py   (reads SIGVERIFY_HOST / SIGVERIFY_PORT)
Then open http://127.0.0.1:8000/ for the console; the API lives under /api/*.

Configuration is read from environment variables, which can be set directly or
via a `.env` file in the repo root (loaded automatically — see `.env.example`
for every variable this module reads, with defaults and explanations). `.env`
itself is gitignored; only the `.env.example` template is committed.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sigverify.alerts import Alert, AlertBroker, classify_alert
from sigverify.audit.ledger import AuditLedger
from sigverify.pipeline.inference import VerificationResult, verify_signature
from sigverify.pipeline.model_bundle import SignatureVerificationBundle
from sigverify.preprocessing.stroke_preprocess import preprocess_stroke_sequence
from sigverify.utils.config import load_config
from sigverify.utils.logging import get_logger

load_dotenv()  # no-op if .env doesn't exist -- real env vars (CI, containers) still win, since
# load_dotenv() never overwrites a variable that's already set in the environment.

logger = get_logger(__name__)

STATE: dict = {}
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# API-key gate for /api/* and /ws/alerts. Disabled (every request allowed) when
# SIGVERIFY_API_KEY is unset/empty, so local dev stays frictionless by default --
# set it (see .env.example) to require the key on every request once this is
# reachable from anywhere other than your own machine. This protects against
# unauthorized *programmatic* callers (another service, a script, curl); it does
# NOT hide the key from someone reading the bundled browser frontend's own
# requests, since a pure client-side app has no way to keep a secret from its own
# user -- see web/js/config.js (generated at startup, gitignored) for how the
# bundled console gets the key to send its own requests.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _configured_api_key() -> str:
    return os.environ.get("SIGVERIFY_API_KEY", "").strip()


def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
    expected = _configured_api_key()
    if not expected:
        return  # auth disabled
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


def _write_frontend_config() -> None:
    """Regenerates web/js/config.js from the current SIGVERIFY_API_KEY so the
    bundled same-origin console can authenticate itself -- see the comment above
    require_api_key for why this isn't a real secrecy boundary, just a shared gate
    value. Gitignored: never commit a real key baked into this generated file.
    """
    config_path = WEB_DIR / "js" / "config.js"
    if not config_path.parent.exists():
        return
    key = _configured_api_key()
    config_path.write_text(
        "// Auto-generated at server startup from SIGVERIFY_API_KEY -- do not edit or commit.\n"
        f"window.SIGNUM_API_KEY = {json.dumps(key)};\n",
        encoding="utf-8",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("SIGVERIFY_CONFIG", "configs/lightweight_real.yaml")
    checkpoint_dir = os.environ.get("SIGVERIFY_CHECKPOINTS", "checkpoints_real")
    cfg = load_config(config_path)
    STATE["config"] = cfg
    STATE["bundle"] = SignatureVerificationBundle(cfg, checkpoint_dir=checkpoint_dir).eval_mode()
    STATE["ledger"] = AuditLedger(cfg.audit_log.path, cfg.audit_log.hash_algo) if cfg.audit_log.enabled else None
    STATE["alert_broker"] = AlertBroker()
    _write_frontend_config()
    logger.info("Model bundle loaded from %s (config=%s)", checkpoint_dir, config_path)
    logger.info("API key auth: %s", "ENABLED" if _configured_api_key() else "disabled (SIGVERIFY_API_KEY not set)")
    yield
    STATE.clear()


app = FastAPI(title="SIGNUM — Neural Signature Forensics", version="0.1.0", lifespan=lifespan)
api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

# Only needed if the web console is served from a different origin than this API
# (e.g. a separate frontend dev server) -- the default setup (this app mounts and
# serves web/ itself, see the bottom of this file) is same-origin and needs no CORS
# at all, so the middleware is skipped entirely unless SIGVERIFY_CORS_ORIGINS is set.
_cors_origins = [origin.strip() for origin in os.environ.get("SIGVERIFY_CORS_ORIGINS", "").split(",") if origin.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled for origins: %s", _cors_origins)


class VerifyResponse(BaseModel):
    decision: str
    combined_score: float
    fused_similarity: float
    static_similarity: float
    dynamic_similarity: float | None
    calibrated_score: float | None
    anomaly_score: float | None
    is_novel: bool | None
    confidence_interval: list[float] | None
    modality_weights: dict
    shap_modality_split: dict | None
    static_heatmap_png_base64: str | None
    alert_severity: str
    alert_message: str


class LiveVerifyResponse(BaseModel):
    dynamic_similarity: float
    num_points: int


@api.get("/health")
def health() -> dict:
    return {"status": "ok", "device": str(STATE["bundle"].device) if "bundle" in STATE else "not_loaded"}


async def _read_image(upload: UploadFile) -> np.ndarray:
    raw_bytes = await upload.read()
    image = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {upload.filename}")
    return image


def _heatmap_to_base64(heatmap: np.ndarray) -> str:
    normalized = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    success, buf = cv2.imencode(".png", colored)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


@api.post("/verify", response_model=VerifyResponse)
async def verify(
    reference_image: UploadFile = File(...),
    query_image: UploadFile = File(...),
    reference_stroke: UploadFile | None = File(None),
    query_stroke: UploadFile | None = File(None),
    user_id: str | None = Form(None),
    localize: bool = Form(False),
    estimate_confidence: bool = Form(True),
) -> VerifyResponse:
    if "bundle" not in STATE:
        raise HTTPException(status_code=503, detail="Model bundle not loaded")

    ref_img = await _read_image(reference_image)
    qry_img = await _read_image(query_image)
    ref_stroke = json.loads(await reference_stroke.read()) if reference_stroke is not None else None
    qry_stroke = json.loads(await query_stroke.read()) if query_stroke is not None else None

    result: VerificationResult = verify_signature(
        STATE["bundle"],
        reference_image=ref_img,
        query_image=qry_img,
        reference_stroke=ref_stroke,
        query_stroke=qry_stroke,
        user_id=user_id,
        localize=localize,
        estimate_confidence=estimate_confidence,
    )

    if STATE.get("ledger") is not None:
        STATE["ledger"].append({"user_id": user_id, **result.to_json_safe()})

    alert: Alert | None = None
    if STATE.get("alert_broker") is not None:
        alert = classify_alert(result, user_id)
        STATE["alert_broker"].publish(alert)

    heatmap_b64 = _heatmap_to_base64(result.static_heatmap) if result.static_heatmap is not None else None

    return VerifyResponse(
        decision=result.decision,
        combined_score=result.combined_score,
        fused_similarity=result.fused_similarity,
        static_similarity=result.static_similarity,
        dynamic_similarity=result.dynamic_similarity,
        calibrated_score=result.calibrated_score,
        anomaly_score=result.anomaly_score,
        is_novel=result.is_novel,
        confidence_interval=list(result.confidence_interval) if result.confidence_interval else None,
        modality_weights=result.modality_weights,
        shap_modality_split=result.shap_modality_split,
        static_heatmap_png_base64=heatmap_b64,
        alert_severity=alert.severity.value if alert is not None else "info",
        alert_message=alert.message if alert is not None else "",
    )


@api.post("/verify/live", response_model=LiveVerifyResponse)
async def verify_live(
    reference_stroke: UploadFile = File(...),
    query_stroke: UploadFile = File(...),
) -> LiveVerifyResponse:
    """Lightweight, dynamic-branch-only similarity for streaming feedback *while* a
    user is actively signing. Skips the static CNN forward pass, fusion, anomaly
    scoring, calibration, and explainability that make a full `/api/verify` call
    take ~1.4-1.8s on CPU (see docs/results.md) — cheap enough to poll every few
    hundred milliseconds during signing without requests piling up behind each
    other. Not a substitute for `/api/verify`: the final accept/review/reject
    decision should always come from the full pipeline.
    """
    if "bundle" not in STATE:
        raise HTTPException(status_code=503, detail="Model bundle not loaded")
    bundle = STATE["bundle"]
    cfg = bundle.config

    ref_stroke = json.loads(await reference_stroke.read())
    qry_stroke = json.loads(await query_stroke.read())
    num_points = len(qry_stroke.get("x", []))
    if num_points < 5:
        raise HTTPException(status_code=400, detail="Too few points for a live estimate (minimum 5)")

    resample_points = cfg.preprocessing.stroke.resample_points
    normalize = cfg.preprocessing.stroke.normalize
    with torch.no_grad():
        ref_matrix = preprocess_stroke_sequence(ref_stroke, resample_points=resample_points, normalize=normalize)
        qry_matrix = preprocess_stroke_sequence(qry_stroke, resample_points=resample_points, normalize=normalize)
        ref_tensor = torch.from_numpy(ref_matrix).unsqueeze(0).to(bundle.device)
        qry_tensor = torch.from_numpy(qry_matrix).unsqueeze(0).to(bundle.device)
        ref_embedding, _ = bundle.dynamic_model(ref_tensor)
        qry_embedding, _ = bundle.dynamic_model(qry_tensor)
        similarity = bundle.dynamic_model.similarity(ref_embedding, qry_embedding).item()

    return LiveVerifyResponse(dynamic_similarity=(similarity + 1) / 2, num_points=num_points)


@api.get("/alerts/recent")
def alerts_recent(limit: int = 50) -> list[dict]:
    if STATE.get("alert_broker") is None:
        raise HTTPException(status_code=503, detail="Alert broker not initialized")
    return [a.to_json() for a in STATE["alert_broker"].recent(limit)]


@api.get("/audit/verify_chain")
def verify_audit_chain() -> dict:
    if STATE.get("ledger") is None:
        raise HTTPException(status_code=404, detail="Audit logging is disabled")
    return STATE["ledger"].verify_chain()


app.include_router(api)


@app.websocket("/ws/alerts")
async def alerts_ws(websocket: WebSocket) -> None:
    """Live Monitor feed: streams every new alert (from any verification, by any
    client) as it's published — the mechanism that makes this a real-time
    *monitoring* console rather than just a single-user verification tool.

    Deliberately does NOT replay history on connect (unlike `GET /api/alerts/recent`,
    which the frontend calls once on page load for that) — a client reconnecting
    after a network blip would otherwise re-receive and double-count everything
    already in the ring buffer on every reconnect.

    Authenticated via an `api_key` query parameter rather than the `X-API-Key`
    header `require_api_key` checks elsewhere — browsers' native WebSocket API
    can't set custom headers on the handshake request, so a query parameter is
    the standard way to carry a token on a WS connection.
    """
    expected = _configured_api_key()
    if expected and not secrets.compare_digest(websocket.query_params.get("api_key", ""), expected):
        await websocket.close(code=1008, reason="Missing or invalid api_key")
        return

    await websocket.accept()
    broker: AlertBroker | None = STATE.get("alert_broker")
    if broker is None:
        await websocket.close(code=1011, reason="Alert broker not initialized")
        return

    queue = broker.subscribe()
    try:
        while True:
            alert = await queue.get()
            await websocket.send_json(alert.to_json())
    except WebSocketDisconnect:
        pass
    finally:
        broker.unsubscribe(queue)


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import sys

    import uvicorn

    # `python api/app.py` (as opposed to `python -m api.app` or `uvicorn api.app:app`)
    # only puts api/'s own directory on sys.path, not the repo root -- so the
    # "api.app:app" import string below would otherwise fail with
    # `ModuleNotFoundError: No module named 'api'` when uvicorn re-imports it.
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    uvicorn.run(
        "api.app:app",
        host=os.environ.get("SIGVERIFY_HOST", "127.0.0.1"),
        port=int(os.environ.get("SIGVERIFY_PORT", "8000")),
        reload=os.environ.get("SIGVERIFY_RELOAD", "false").lower() == "true",
    )
