"""Loads every trained component into one object the inference pipeline can call.

Each component is optional at load time (missing checkpoint -> freshly initialized
weights) so the pipeline is runnable immediately after `pip install -e .` for
smoke-testing and demos; real verification accuracy requires training each component
first (see scripts/train_*.py) and pointing this bundle at the resulting checkpoints.
"""
from __future__ import annotations

from pathlib import Path

import torch

from sigverify.localization.yolo_localizer import SignatureLocalizer
from sigverify.models.anomaly import AnomalyDetector
from sigverify.models.calibration import PlattCalibrator, ScoreBasedLikelihoodRatio
from sigverify.models.dynamic_branch import DynamicStrokeEncoder
from sigverify.models.fusion import CrossAttentionGatedFusion
from sigverify.models.static_branch import SiameseCNN
from sigverify.utils.config import Config
from sigverify.utils.logging import get_logger
from sigverify.utils.seed import get_device

logger = get_logger(__name__)


class SignatureVerificationBundle:
    def __init__(self, config: Config, checkpoint_dir: str | Path | None = None) -> None:
        self.config = config
        self.device = get_device(config.get("device", "cuda"))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self.static_model = SiameseCNN(
            backbone=config.static_branch.backbone,
            embedding_dim=config.static_branch.embedding_dim,
            pretrained=config.static_branch.pretrained,
        ).to(self.device)

        self.dynamic_model = DynamicStrokeEncoder(
            input_dim=config.dynamic_branch.input_dim,
            hidden_dim=config.dynamic_branch.hidden_dim,
            num_layers=config.dynamic_branch.num_layers,
            num_heads=config.dynamic_branch.num_heads,
            embedding_dim=config.dynamic_branch.embedding_dim,
            encoder=config.dynamic_branch.encoder,
            bidirectional=config.dynamic_branch.bidirectional,
            dropout=config.dynamic_branch.dropout,
        ).to(self.device)

        self.fusion_model = CrossAttentionGatedFusion(
            embedding_dim=config.fusion.embedding_dim,
            num_heads=config.fusion.num_heads,
            dropout=config.fusion.dropout,
            reliability_gating=config.fusion.reliability_gating,
        ).to(self.device)

        self.anomaly_detectors: dict[str, AnomalyDetector] = {}
        self.calibrator: PlattCalibrator | ScoreBasedLikelihoodRatio | None = None
        self._localizer: SignatureLocalizer | None = None

        if self.checkpoint_dir is not None:
            self._load_checkpoints()

    @property
    def localizer(self) -> SignatureLocalizer:
        if self._localizer is None:
            self._localizer = SignatureLocalizer(
                weights=self.config.localization.weights,
                conf_threshold=self.config.localization.conf_threshold,
                imgsz=self.config.localization.imgsz,
                fallback_full_frame=self.config.localization.fallback_full_frame,
            )
        return self._localizer

    def _load_checkpoints(self) -> None:
        mapping = {
            "static_branch.pt": self.static_model,
            "dynamic_branch.pt": self.dynamic_model,
            "fusion.pt": self.fusion_model,
        }
        for filename, module in mapping.items():
            path = self.checkpoint_dir / filename
            if path.exists():
                module.load_state_dict(torch.load(path, map_location=self.device))
                logger.info("Loaded checkpoint %s", path)
            else:
                logger.warning("Checkpoint %s not found — using randomly initialized weights", path)

        calibrator_path = self.checkpoint_dir / "calibrator.joblib"
        if calibrator_path.exists():
            method = self.config.calibration.method
            cls = PlattCalibrator if method == "platt" else ScoreBasedLikelihoodRatio
            self.calibrator = cls.load(calibrator_path)
            logger.info("Loaded calibrator %s", calibrator_path)

        anomaly_dir = self.checkpoint_dir / "anomaly"
        if anomaly_dir.exists():
            for path in anomaly_dir.glob("*.joblib"):
                self.anomaly_detectors[path.stem] = AnomalyDetector.load(path)
            logger.info("Loaded %d per-user anomaly detectors", len(self.anomaly_detectors))

    def eval_mode(self) -> SignatureVerificationBundle:
        self.static_model.eval()
        self.dynamic_model.eval()
        self.fusion_model.eval()
        return self
