#!/usr/bin/env python
"""Fit the decision calibrator (Platt scaling or SLR) and one per-writer anomaly
detector, using the already-trained static+dynamic+fusion checkpoints to produce the
fused embeddings/similarity scores these two calibration steps are fit on. Run this
after train_static.py, train_dynamic.py, and train_fusion.py.

Usage:
    python scripts/fit_calibration.py --static-manifest data/processed/demo/static_manifest.jsonl \
        --dynamic-manifest data/processed/demo/dynamic_manifest.jsonl \
        --checkpoints checkpoints/ --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from sigverify.data.datasets import load_manifest
from sigverify.models.anomaly import AnomalyDetector
from sigverify.models.calibration import PlattCalibrator, ScoreBasedLikelihoodRatio
from sigverify.models.dynamic_branch import DynamicStrokeEncoder
from sigverify.models.fusion import CrossAttentionGatedFusion
from sigverify.models.static_branch import SiameseCNN
from sigverify.preprocessing.image_preprocess import preprocess_signature_image
from sigverify.preprocessing.stroke_preprocess import preprocess_stroke_sequence
from sigverify.utils.config import load_config
from sigverify.utils.logging import get_logger
from sigverify.utils.seed import get_device, set_seed

logger = get_logger(__name__)


def load_models(cfg, ckpt_dir: Path, device):
    static_model = SiameseCNN(cfg.static_branch.backbone, cfg.static_branch.embedding_dim, cfg.static_branch.pretrained).to(device)
    dynamic_model = DynamicStrokeEncoder(
        cfg.dynamic_branch.input_dim, cfg.dynamic_branch.hidden_dim, cfg.dynamic_branch.num_layers,
        cfg.dynamic_branch.num_heads, cfg.dynamic_branch.embedding_dim, cfg.dynamic_branch.encoder,
        cfg.dynamic_branch.bidirectional, cfg.dynamic_branch.dropout,
    ).to(device)
    fusion_model = CrossAttentionGatedFusion(cfg.fusion.embedding_dim, cfg.fusion.num_heads, cfg.fusion.dropout, cfg.fusion.reliability_gating).to(device)

    for filename, model in [("static_branch.pt", static_model), ("dynamic_branch.pt", dynamic_model), ("fusion.pt", fusion_model)]:
        path = ckpt_dir / filename
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            logger.warning("%s missing — using randomly initialized weights", path)
        model.eval()
    return static_model, dynamic_model, fusion_model


@torch.no_grad()
def fused_embedding(static_model, dynamic_model, fusion_model, image_path, stroke_path, cfg, device):
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = preprocess_signature_image(raw, target_size=tuple(cfg.preprocessing.image.target_size))
    image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    static_emb = static_model.embed(image_t)

    dynamic_emb, mask_val = None, False
    if stroke_path is not None:
        with open(stroke_path, "r", encoding="utf-8") as fh:
            stroke = json.load(fh)
        stroke_matrix = preprocess_stroke_sequence(stroke, cfg.preprocessing.stroke.resample_points, cfg.preprocessing.stroke.normalize)
        stroke_t = torch.from_numpy(stroke_matrix).unsqueeze(0).to(device)
        dynamic_emb, _ = dynamic_model(stroke_t)
        mask_val = True

    mask = torch.tensor([mask_val], device=device)
    fusion_out = fusion_model(static_emb, dynamic_emb, mask)
    return fusion_out["fused_embedding"].squeeze(0).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-manifest", required=True)
    parser.add_argument("--dynamic-manifest", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoints", default="checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    ckpt_dir = Path(args.checkpoints)
    static_model, dynamic_model, fusion_model = load_models(cfg, ckpt_dir, device)

    static_records = load_manifest(args.static_manifest)
    dynamic_by_path: dict[str, str] = {}
    if args.dynamic_manifest:
        # Best-effort alignment: same writer/label/order as produced by generate_demo_data.py.
        dynamic_records = load_manifest(args.dynamic_manifest)
        grouped: dict[tuple, list[str]] = {}
        for rec in dynamic_records:
            grouped.setdefault((rec["writer_id"], rec["label"]), []).append(rec["path"])
        counters: dict[tuple, int] = {}
        for rec in static_records:
            key = (rec["writer_id"], rec["label"])
            i = counters.get(key, 0)
            paths = grouped.get(key, [])
            if i < len(paths):
                dynamic_by_path[rec["path"]] = paths[i]
            counters[key] = i + 1

    embeddings, labels, writers = [], [], []
    for rec in static_records:
        emb = fused_embedding(static_model, dynamic_model, fusion_model, rec["path"], dynamic_by_path.get(rec["path"]), cfg, device)
        embeddings.append(emb)
        labels.append(rec["label"])
        writers.append(rec["writer_id"])
    embeddings = np.stack(embeddings)

    # --- Calibrator: fit on genuine-vs-genuine / genuine-vs-forged similarity scores ---
    genuine_idx = [i for i, l in enumerate(labels) if l == "genuine"]
    forged_idx = [i for i, l in enumerate(labels) if l == "forged"]
    rng = np.random.default_rng(cfg.seed)
    n_pairs = min(500, len(genuine_idx) * (len(genuine_idx) - 1) // 2 or 1)

    scores, score_labels = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(genuine_idx, size=2, replace=False)
        sim = float(np.dot(embeddings[i], embeddings[j]))
        scores.append((sim + 1) / 2)
        score_labels.append(1)
    if forged_idx:
        for _ in range(n_pairs):
            i = rng.choice(genuine_idx)
            j = rng.choice(forged_idx)
            sim = float(np.dot(embeddings[i], embeddings[j]))
            scores.append((sim + 1) / 2)
            score_labels.append(0)

    scores_arr, labels_arr = np.array(scores), np.array(score_labels)
    if cfg.calibration.method == "platt":
        calibrator = PlattCalibrator().fit(scores_arr, labels_arr)
    else:
        calibrator = ScoreBasedLikelihoodRatio(num_bins=cfg.calibration.slr_bins).fit(scores_arr[labels_arr == 1], scores_arr[labels_arr == 0])
    calibrator.save(ckpt_dir / "calibrator.joblib")
    logger.info("Saved calibrator (%s) to %s", cfg.calibration.method, ckpt_dir / "calibrator.joblib")

    # --- Per-writer anomaly detectors: fit on that writer's genuine fused embeddings ---
    anomaly_dir = ckpt_dir / "anomaly"
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    unique_writers = sorted(set(writers))
    for writer in unique_writers:
        writer_genuine = np.stack([embeddings[i] for i in genuine_idx if writers[i] == writer])
        if len(writer_genuine) < 3:
            continue
        detector = AnomalyDetector(method=cfg.anomaly_detection.method, contamination=cfg.anomaly_detection.contamination)
        detector.fit(writer_genuine)
        detector.save(anomaly_dir / f"{writer}.joblib")
    logger.info("Saved %d per-writer anomaly detectors to %s", len(unique_writers), anomaly_dir)


if __name__ == "__main__":
    main()
