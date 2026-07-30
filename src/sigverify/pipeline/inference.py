"""End-to-end verification pipeline: raw inputs -> VerificationResult.

Mirrors the architecture's full chain: region localization -> preprocessing -> static
+ dynamic branches -> cross-attention fusion -> similarity + anomaly scoring ->
weighted decision fusion -> calibration -> explainability. Every stage past
localization is exercised even when only a static image is supplied (dynamic
similarity/attention are simply omitted and the weighted combination renormalizes),
so the pipeline works for scanned-only signatures as well as full stylus captures.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from sigverify.explainability.attention_viz import stroke_deviation_scores, top_k_deviant_indices
from sigverify.explainability.gradcam import SiameseGradCAM
from sigverify.explainability.shap_explainer import DecisionSHAPExplainer
from sigverify.pipeline.model_bundle import SignatureVerificationBundle
from sigverify.preprocessing.image_preprocess import preprocess_signature_image
from sigverify.preprocessing.stroke_preprocess import preprocess_stroke_sequence
from sigverify.utils.logging import get_logger

logger = get_logger(__name__)

ImageInput = np.ndarray | str | Path
StrokeInput = dict | str | Path | None

DECISION_WEIGHTS_FULL = {"fused": 0.5, "static": 0.2, "dynamic": 0.2, "anomaly": 0.1}


@dataclass
class VerificationResult:
    decision: str
    combined_score: float
    fused_similarity: float
    static_similarity: float
    dynamic_similarity: float | None
    calibrated_score: float | None
    anomaly_score: float | None
    is_novel: bool | None
    confidence_interval: tuple[float, float] | None
    modality_weights: dict
    static_heatmap: np.ndarray | None = None
    dynamic_deviation_scores: np.ndarray | None = None
    top_deviant_indices: np.ndarray | None = None
    shap_modality_split: dict | None = None
    timestamp: float = field(default_factory=time.time)

    def to_json_safe(self) -> dict:
        payload = {
            "decision": self.decision,
            "combined_score": self.combined_score,
            "fused_similarity": self.fused_similarity,
            "static_similarity": self.static_similarity,
            "dynamic_similarity": self.dynamic_similarity,
            "calibrated_score": self.calibrated_score,
            "anomaly_score": self.anomaly_score,
            "is_novel": self.is_novel,
            "confidence_interval": self.confidence_interval,
            "modality_weights": self.modality_weights,
            "shap_modality_split": self.shap_modality_split,
            "timestamp": self.timestamp,
        }
        if self.top_deviant_indices is not None:
            payload["top_deviant_indices"] = self.top_deviant_indices.tolist()
        return payload


def _load_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, (str, Path)):
        raw = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(image)
        return raw
    return image


def _load_stroke(stroke: StrokeInput) -> dict | None:
    if stroke is None:
        return None
    if isinstance(stroke, (str, Path)):
        import json

        with open(stroke, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return stroke


def _jitter_image(image: np.ndarray, max_angle: float, max_shift_frac: float, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    angle = rng.uniform(-max_angle, max_angle)
    shift_x = rng.uniform(-max_shift_frac, max_shift_frac) * w
    shift_y = rng.uniform(-max_shift_frac, max_shift_frac) * h
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(image, matrix, (w, h), borderValue=border_value)


def _combine_decision(
    fused_similarity: float,
    static_similarity: float,
    dynamic_similarity: float | None,
    anomaly_score: float | None,
    calibrated_score: float | None,
) -> float:
    """Weighted combination of the calibrated fused similarity, the two raw per-modality
    similarities, and the anomaly score — the "Decision Fusion" step. Missing signals
    (no dynamic capture, no enrolled anomaly model yet) drop out and the remaining
    weights renormalize, rather than being back-filled with a neutral placeholder.
    """
    fused_component = calibrated_score if calibrated_score is not None else (fused_similarity + 1) / 2
    components = {"fused": fused_component, "static": (static_similarity + 1) / 2}
    weights = {"fused": DECISION_WEIGHTS_FULL["fused"], "static": DECISION_WEIGHTS_FULL["static"]}

    if dynamic_similarity is not None:
        components["dynamic"] = (dynamic_similarity + 1) / 2
        weights["dynamic"] = DECISION_WEIGHTS_FULL["dynamic"]
    if anomaly_score is not None:
        components["anomaly"] = anomaly_score
        weights["anomaly"] = DECISION_WEIGHTS_FULL["anomaly"]

    total_weight = sum(weights.values())
    return sum(components[k] * (weights[k] / total_weight) for k in components)


def _decide(combined_score: float, cfg) -> str:
    if combined_score >= cfg.decision.accept_threshold:
        return "Genuine"
    if combined_score >= cfg.decision.review_lower_bound:
        return "Review"
    return "Forged"


class _SampleEncoding:
    """Holds one sample's (preprocessed) tensors + embeddings, computed once and reused
    across the main decision pass and the explainability pass.
    """

    __slots__ = ("dynamic_attn", "dynamic_embedding", "image_tensor", "static_embedding", "stroke_tensor")

    def __init__(self):
        self.image_tensor = None
        self.stroke_tensor = None
        self.static_embedding = None
        self.dynamic_embedding = None
        self.dynamic_attn = None


def _encode_sample(bundle: SignatureVerificationBundle, image: np.ndarray, stroke: dict | None) -> _SampleEncoding:
    cfg = bundle.config
    enc = _SampleEncoding()

    processed_image = preprocess_signature_image(
        image,
        target_size=tuple(cfg.preprocessing.image.target_size),
        binarize_method=cfg.preprocessing.image.binarize_method,
        deskew_enabled=cfg.preprocessing.image.deskew,
        denoise_h=cfg.preprocessing.image.denoise_h,
    )
    enc.image_tensor = torch.from_numpy(processed_image).unsqueeze(0).unsqueeze(0).to(bundle.device)
    enc.static_embedding = bundle.static_model.embed(enc.image_tensor)

    if stroke is not None:
        processed_stroke = preprocess_stroke_sequence(
            stroke,
            resample_points=cfg.preprocessing.stroke.resample_points,
            normalize=cfg.preprocessing.stroke.normalize,
        )
        enc.stroke_tensor = torch.from_numpy(processed_stroke).unsqueeze(0).to(bundle.device)
        enc.dynamic_embedding, enc.dynamic_attn = bundle.dynamic_model(enc.stroke_tensor)

    return enc


def verify_signature(
    bundle: SignatureVerificationBundle,
    reference_image: ImageInput,
    query_image: ImageInput,
    reference_stroke: StrokeInput = None,
    query_stroke: StrokeInput = None,
    user_id: str | None = None,
    localize: bool = False,
    explain: bool = True,
    estimate_confidence: bool = True,
    tta_rounds: int = 7,
    shap_background: np.ndarray | None = None,
) -> VerificationResult:
    bundle.eval_mode()
    cfg = bundle.config

    ref_img = _load_image(reference_image)
    qry_img = _load_image(query_image)
    ref_stroke = _load_stroke(reference_stroke)
    qry_stroke = _load_stroke(query_stroke)

    if localize:
        ref_img = bundle.localizer.crop_best_region(ref_img)
        qry_img = bundle.localizer.crop_best_region(qry_img)

    with torch.no_grad():
        ref_enc = _encode_sample(bundle, ref_img, ref_stroke)
        qry_enc = _encode_sample(bundle, qry_img, qry_stroke)

        static_similarity = float(bundle.static_model.similarity(ref_enc.static_embedding, qry_enc.static_embedding).item())
        dynamic_similarity = None
        if ref_enc.dynamic_embedding is not None and qry_enc.dynamic_embedding is not None:
            dynamic_similarity = float(
                bundle.dynamic_model.similarity(ref_enc.dynamic_embedding, qry_enc.dynamic_embedding).item()
            )

        ref_mask = torch.tensor([ref_enc.dynamic_embedding is not None], device=bundle.device)
        qry_mask = torch.tensor([qry_enc.dynamic_embedding is not None], device=bundle.device)
        ref_fusion = bundle.fusion_model(ref_enc.static_embedding, ref_enc.dynamic_embedding, ref_mask)
        qry_fusion = bundle.fusion_model(qry_enc.static_embedding, qry_enc.dynamic_embedding, qry_mask)
        fused_similarity = float(
            bundle.fusion_model.similarity(ref_fusion["fused_embedding"], qry_fusion["fused_embedding"]).item()
        )

    anomaly_score, is_novel = None, None
    if user_id is not None and user_id in bundle.anomaly_detectors:
        detector = bundle.anomaly_detectors[user_id]
        fused_np = qry_fusion["fused_embedding"].cpu().numpy()
        anomaly_score = float(detector.score(fused_np)[0])
        is_novel = bool(detector.is_novel(fused_np)[0])

    calibrated_score = None
    if bundle.calibrator is not None:
        mapped = np.array([(fused_similarity + 1) / 2])
        if hasattr(bundle.calibrator, "calibrate"):
            calibrated_score = float(bundle.calibrator.calibrate(mapped)[0])
        else:
            calibrated_score = float(bundle.calibrator.posterior_probability(mapped)[0])

    combined_score = _combine_decision(fused_similarity, static_similarity, dynamic_similarity, anomaly_score, calibrated_score)
    decision = _decide(combined_score, cfg)

    confidence_interval = None
    if estimate_confidence and tta_rounds > 1:
        confidence_interval = _estimate_confidence_interval(
            bundle, ref_img, qry_img, ref_stroke, qry_stroke, anomaly_score, calibrated_score, tta_rounds
        )

    result = VerificationResult(
        decision=decision,
        combined_score=combined_score,
        fused_similarity=fused_similarity,
        static_similarity=static_similarity,
        dynamic_similarity=dynamic_similarity,
        calibrated_score=calibrated_score,
        anomaly_score=anomaly_score,
        is_novel=is_novel,
        confidence_interval=confidence_interval,
        modality_weights={
            "static_weight": float(qry_fusion["static_weight"].item()),
            "dynamic_weight": float(qry_fusion["dynamic_weight"].item()),
        },
    )

    if explain:
        _attach_explanations(bundle, result, ref_enc, qry_enc, shap_background)

    logger.info("Verification decision=%s combined_score=%.4f", decision, combined_score)
    return result


def _estimate_confidence_interval(
    bundle: SignatureVerificationBundle,
    ref_img: np.ndarray,
    qry_img: np.ndarray,
    ref_stroke: dict | None,
    qry_stroke: dict | None,
    anomaly_score: float | None,
    calibrated_score: float | None,
    tta_rounds: int,
) -> tuple[float, float]:
    """95% CI on the combined score via test-time augmentation: small rotation/shift
    jitter on both images, re-run the static (and fusion) forward pass, and take the
    normal-approximation interval over the resulting score distribution. This reflects
    genuine model sensitivity to capture variation rather than a fabricated margin.
    """
    rng = np.random.default_rng(0)
    scores = []
    with torch.no_grad():
        for _ in range(tta_rounds):
            ref_jittered = _jitter_image(ref_img, max_angle=3.0, max_shift_frac=0.02, rng=rng)
            qry_jittered = _jitter_image(qry_img, max_angle=3.0, max_shift_frac=0.02, rng=rng)
            ref_enc = _encode_sample(bundle, ref_jittered, ref_stroke)
            qry_enc = _encode_sample(bundle, qry_jittered, qry_stroke)

            static_similarity = float(bundle.static_model.similarity(ref_enc.static_embedding, qry_enc.static_embedding).item())
            dynamic_similarity = None
            if ref_enc.dynamic_embedding is not None and qry_enc.dynamic_embedding is not None:
                dynamic_similarity = float(
                    bundle.dynamic_model.similarity(ref_enc.dynamic_embedding, qry_enc.dynamic_embedding).item()
                )
            ref_mask = torch.tensor([ref_enc.dynamic_embedding is not None], device=bundle.device)
            qry_mask = torch.tensor([qry_enc.dynamic_embedding is not None], device=bundle.device)
            ref_fusion = bundle.fusion_model(ref_enc.static_embedding, ref_enc.dynamic_embedding, ref_mask)
            qry_fusion = bundle.fusion_model(qry_enc.static_embedding, qry_enc.dynamic_embedding, qry_mask)
            fused_similarity = float(
                bundle.fusion_model.similarity(ref_fusion["fused_embedding"], qry_fusion["fused_embedding"]).item()
            )
            scores.append(_combine_decision(fused_similarity, static_similarity, dynamic_similarity, anomaly_score, calibrated_score))

    scores_arr = np.array(scores)
    mean, sem = scores_arr.mean(), scores_arr.std(ddof=1) / np.sqrt(len(scores_arr))
    return float(np.clip(mean - 1.96 * sem, 0, 1)), float(np.clip(mean + 1.96 * sem, 0, 1))


def _attach_explanations(
    bundle: SignatureVerificationBundle,
    result: VerificationResult,
    ref_enc: _SampleEncoding,
    qry_enc: _SampleEncoding,
    shap_background: np.ndarray | None,
) -> None:
    gradcam = SiameseGradCAM(bundle.static_model)
    cam_result = gradcam.explain(qry_enc.image_tensor, ref_enc.image_tensor)
    result.static_heatmap = cam_result["heatmap"]

    if qry_enc.dynamic_embedding is not None and ref_enc.dynamic_embedding is not None:
        query_seq = qry_enc.stroke_tensor.squeeze(0).cpu().numpy()
        reference_seq = ref_enc.stroke_tensor.squeeze(0).cpu().numpy()
        attn_weights = qry_enc.dynamic_attn.squeeze(0).cpu().numpy()
        deviation = stroke_deviation_scores(query_seq, reference_seq, attn_weights)
        result.dynamic_deviation_scores = deviation
        result.top_deviant_indices = top_k_deviant_indices(deviation, k=bundle.config.explainability.top_k_deviant_strokes)

    if shap_background is not None:
        features = np.array(
            [
                result.fused_similarity,
                result.static_similarity,
                result.dynamic_similarity if result.dynamic_similarity is not None else 0.0,
                result.anomaly_score if result.anomaly_score is not None else 0.0,
            ]
        )

        def decision_fn(batch: np.ndarray) -> np.ndarray:
            out = np.empty(len(batch))
            for i, row in enumerate(batch):
                out[i] = _combine_decision(row[0], row[1], row[2] if result.dynamic_similarity is not None else None, row[3] if result.anomaly_score is not None else None, result.calibrated_score)
            return out

        explainer = DecisionSHAPExplainer(decision_fn, shap_background)
        shap_result = explainer.explain(features)
        result.shap_modality_split = shap_result["modality_split"]
