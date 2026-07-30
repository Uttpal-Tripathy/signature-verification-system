#!/usr/bin/env python
"""End-to-end evaluation of the full fused pipeline on a held-out manifest, reporting
the standard EER/AUC/accuracy benchmark metrics. Also mines the verifier's failure
cases (forgeries that scored as genuine; genuine signatures that scored as forged)
into a FailureCaseBuffer and persists them to disk, closing the adversarial loop:
the next `train_gan.py` run can sample this buffer to retarget forgery synthesis at
the verifier's *current* weaknesses.

Usage:
    python scripts/evaluate.py --static-manifest data/processed/demo/static_manifest.jsonl \
        --checkpoints checkpoints/ --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sigverify.data.datasets import load_manifest
from sigverify.models.gan_forgery import FailureCaseBuffer
from sigverify.pipeline.model_bundle import SignatureVerificationBundle
from sigverify.preprocessing.image_preprocess import preprocess_signature_image
from sigverify.utils.config import load_config
from sigverify.utils.logging import get_logger
from sigverify.utils.metrics import verification_report
from sigverify.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-manifest", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--failure-buffer-output", default="checkpoints/failure_cases.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    bundle = SignatureVerificationBundle(cfg, checkpoint_dir=args.checkpoints).eval_mode()

    records = load_manifest(args.static_manifest)
    by_writer_genuine: dict[str, list[str]] = {}
    by_writer_forged: dict[str, list[str]] = {}
    for rec in records:
        bucket = by_writer_genuine if rec["label"] == "genuine" else by_writer_forged
        bucket.setdefault(rec["writer_id"], []).append(rec["path"])

    rng = np.random.default_rng(cfg.seed)
    writers = [w for w, paths in by_writer_genuine.items() if len(paths) >= 2]
    buffer = FailureCaseBuffer()
    genuine_scores, forgery_scores = [], []

    with torch.no_grad():
        for _ in range(args.num_pairs):
            writer = rng.choice(writers)
            ref_path, qry_path = rng.choice(by_writer_genuine[writer], size=2, replace=len(by_writer_genuine[writer]) < 2)
            ref_img, qry_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED), cv2.imread(qry_path, cv2.IMREAD_UNCHANGED)
            ref_t = _to_tensor(ref_img, bundle, cfg)
            qry_t = _to_tensor(qry_img, bundle, cfg)
            score = float(bundle.static_model.similarity(bundle.static_model.embed(ref_t), bundle.static_model.embed(qry_t)).item())
            score = (score + 1) / 2
            genuine_scores.append(score)
            if score < cfg.decision.accept_threshold:
                buffer.add_false_positive(qry_t.squeeze(0))

            if by_writer_forged.get(writer):
                forged_path = rng.choice(by_writer_forged[writer])
                forged_img = cv2.imread(forged_path, cv2.IMREAD_UNCHANGED)
                forged_t = _to_tensor(forged_img, bundle, cfg)
                f_score = float(bundle.static_model.similarity(bundle.static_model.embed(ref_t), bundle.static_model.embed(forged_t)).item())
                f_score = (f_score + 1) / 2
                forgery_scores.append(f_score)
                if f_score >= cfg.decision.accept_threshold:
                    buffer.add_false_negative(forged_t.squeeze(0))

    metrics = verification_report(np.array(genuine_scores), np.array(forgery_scores) if forgery_scores else np.array([0.0]))
    logger.info("Static-branch evaluation: EER=%.4f | AUC=%.4f | accuracy@EER-threshold=%.4f", metrics["eer"], metrics["roc_auc"], metrics["accuracy_at_eer_threshold"])
    logger.info("Mined %d false negatives, %d false positives into the failure-case buffer", len(buffer.false_negatives), len(buffer.false_positives))

    output_path = Path(args.failure_buffer_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"false_negatives": list(buffer.false_negatives), "false_positives": list(buffer.false_positives)}, output_path)
    logger.info("Saved failure-case buffer to %s", output_path)


def _to_tensor(image, bundle, cfg):
    processed = preprocess_signature_image(image, target_size=tuple(cfg.preprocessing.image.target_size))
    return torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(bundle.device)


if __name__ == "__main__":
    main()
