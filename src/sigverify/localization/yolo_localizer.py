"""Signature region localization via YOLOv8 (Ultralytics).

Detects and crops the signature region from a scanned document/form so the
static/dynamic branches always see a tight, consistently-framed signature —
removing background noise (letterhead, stamps, printed text) before feature
extraction. Falls back to the full frame if no box clears the confidence
threshold, so the pipeline degrades gracefully on already-cropped inputs.

To specialize beyond the stock COCO-pretrained weights, fine-tune YOLOv8 on a
signature-region dataset (bounding boxes around the signature block on scanned
forms/cheques) and point `weights` at the resulting checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    box_xyxy: tuple[int, int, int, int]
    confidence: float


class SignatureLocalizer:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        fallback_full_frame: bool = True,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.fallback_full_frame = fallback_full_frame
        self._model = None
        self._weights = weights

    @property
    def model(self):
        if self._model is None:
            from ultralytics import (
                YOLO,  # deferred: heavy import, only needed at inference/train time
            )

            self._model = YOLO(self._weights)
        return self._model

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(image, imgsz=self.imgsz, conf=self.conf_threshold, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                detections.append(Detection(box_xyxy=tuple(int(v) for v in box), confidence=float(conf)))
        return detections

    def crop_best_region(self, image: np.ndarray) -> np.ndarray:
        """Return the highest-confidence signature crop, or the full image as fallback."""
        detections = self.detect(image)
        if not detections:
            if self.fallback_full_frame:
                return image
            raise RuntimeError("No signature region detected and fallback_full_frame=False")
        best = max(detections, key=lambda d: d.confidence)
        x0, y0, x1, y1 = best.box_xyxy
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return image
        return image[y0:y1, x0:x1]
