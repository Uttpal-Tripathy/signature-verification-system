#!/usr/bin/env python
"""Run a single reference-vs-query verification and emit a Forensic Verification
Report (PDF + JSON).

Usage:
    python scripts/run_inference.py --reference ref.png --query query.png \
        --checkpoints checkpoints/ --output reports/case_001
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sigverify.audit.ledger import AuditLedger
from sigverify.pipeline.inference import verify_signature
from sigverify.pipeline.model_bundle import SignatureVerificationBundle
from sigverify.pipeline.report import generate_pdf_report, to_json
from sigverify.utils.config import load_config
from sigverify.utils.logging import get_logger
from sigverify.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--reference-stroke", default=None)
    parser.add_argument("--query-stroke", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--output", default="reports/verification_case")
    parser.add_argument("--localize", action="store_true", help="Run YOLOv8 signature-region detection before verification")
    parser.add_argument("--no-confidence", action="store_true", help="Skip the TTA confidence-interval pass (faster)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    bundle = SignatureVerificationBundle(cfg, checkpoint_dir=args.checkpoints)

    result = verify_signature(
        bundle,
        reference_image=args.reference,
        query_image=args.query,
        reference_stroke=args.reference_stroke,
        query_stroke=args.query_stroke,
        user_id=args.user_id,
        localize=args.localize,
        estimate_confidence=not args.no_confidence,
    )

    output_prefix = Path(args.output)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_payload = to_json(result, str(output_prefix) + ".json")
    generate_pdf_report(result, str(output_prefix) + ".pdf", reference_id=args.reference, query_id=args.query)

    if cfg.audit_log.enabled:
        ledger = AuditLedger(cfg.audit_log.path, cfg.audit_log.hash_algo)
        ledger.append({"reference": args.reference, "query": args.query, "user_id": args.user_id, **json_payload})

    logger.info("Decision: %s (combined_score=%.4f)", result.decision, result.combined_score)
    logger.info("Report written to %s.pdf / %s.json", output_prefix, output_prefix)


if __name__ == "__main__":
    main()
