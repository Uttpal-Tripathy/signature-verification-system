"""FastAPI service exposing the signature verification pipeline over HTTP, and
serving the SIGNUM web console (web/) as the root static site.

Run with:
    uvicorn api.app:app --host 0.0.0.0 --port 8000
Then open http://127.0.0.1:8000/ for the console; the API lives under /api/*.

Environment variables:
    SIGVERIFY_CONFIG      path to the YAML config (default: configs/lightweight_real.yaml)
    SIGVERIFY_CHECKPOINTS path to the checkpoint directory (default: checkpoints_real/)
"""
from __future__ import annotations

import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sigverify.audit.ledger import AuditLedger
from sigverify.pipeline.inference import VerificationResult, verify_signature
from sigverify.pipeline.model_bundle import SignatureVerificationBundle
from sigverify.utils.config import load_config
from sigverify.utils.logging import get_logger

logger = get_logger(__name__)

STATE: dict = {}
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("SIGVERIFY_CONFIG", "configs/lightweight_real.yaml")
    checkpoint_dir = os.environ.get("SIGVERIFY_CHECKPOINTS", "checkpoints_real")
    cfg = load_config(config_path)
    STATE["config"] = cfg
    STATE["bundle"] = SignatureVerificationBundle(cfg, checkpoint_dir=checkpoint_dir).eval_mode()
    STATE["ledger"] = AuditLedger(cfg.audit_log.path, cfg.audit_log.hash_algo) if cfg.audit_log.enabled else None
    logger.info("Model bundle loaded from %s (config=%s)", checkpoint_dir, config_path)
    yield
    STATE.clear()


app = FastAPI(title="SIGNUM — Neural Signature Forensics", version="0.1.0", lifespan=lifespan)
api = APIRouter(prefix="/api")


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
    )


@api.get("/audit/verify_chain")
def verify_audit_chain() -> dict:
    if STATE.get("ledger") is None:
        raise HTTPException(status_code=404, detail="Audit logging is disabled")
    return STATE["ledger"].verify_chain()


app.include_router(api)

if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
