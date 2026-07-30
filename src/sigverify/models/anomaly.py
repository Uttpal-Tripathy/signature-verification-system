"""Anomaly / novelty scoring over fused embeddings — a secondary, label-free check.

The similarity model only knows "does this match the enrolled reference"; it has no
notion of "is this signature unlike anything this user has ever produced" without a
paired forgery to compare against. One-Class SVM / Isolation Forest fit on a single
user's enrolled genuine embeddings catches that case directly (Table: Out-of-
Distribution / Novelty Detection), including forgery styles never seen during
training and drifted/degraded capture conditions.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class AnomalyDetector:
    def __init__(self, method: str = "isolation_forest", contamination: float = 0.05) -> None:
        self.method = method
        if method == "isolation_forest":
            self.model = IsolationForest(contamination=contamination, random_state=42)
        elif method == "one_class_svm":
            self.model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        else:
            raise ValueError(f"Unknown anomaly method: {method}")
        self._fitted = False

    def fit(self, embeddings: np.ndarray) -> AnomalyDetector:
        """embeddings: (N, D) enrolled genuine embeddings for a single writer/user."""
        self.model.fit(embeddings)
        self._fitted = True
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Higher = more normal (in-distribution). Rescaled to roughly [0, 1] for
        combination with the calibrated similarity score in the decision-fusion stage.
        """
        if not self._fitted:
            raise RuntimeError("AnomalyDetector must be fit() before scoring")
        raw = self.model.score_samples(embeddings)
        return 1.0 / (1.0 + np.exp(-raw))  # sigmoid squash, monotonic w.r.t. raw score

    def is_novel(self, embeddings: np.ndarray) -> np.ndarray:
        """True where the sample is flagged as an outlier relative to the enrolled writer."""
        return self.model.predict(embeddings) == -1

    def save(self, path: str | Path) -> None:
        joblib.dump({"method": self.method, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> AnomalyDetector:
        payload = joblib.load(path)
        instance = cls(method=payload["method"])
        instance.model = payload["model"]
        instance._fitted = True
        return instance
