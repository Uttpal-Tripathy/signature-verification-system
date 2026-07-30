"""Decision calibration — turns a raw fused similarity score into a forensically
interpretable, legally-referenceable evidentiary metric (Gap E).

Two calibrators are provided:

* Platt scaling: a 1-D logistic regression mapping raw score -> P(genuine). Simple,
  well-calibrated when you just need a probability for the accept/review/reject gate.
* Score-based Likelihood Ratio (SLR): the standard forensic-comparison framework
  (Morrison, 2010) used in voice/handwriting forensics. Bins the genuine and forgery
  score distributions separately and reports LR = f_genuine(score) / f_forgery(score) —
  "the evidence is N times more likely under the genuine hypothesis than the forgery
  hypothesis," which is the form forensic reports and courts expect, rather than a bare
  similarity number.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression()
        self._fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> PlattCalibrator:
        """scores: raw similarity in [-1, 1] or [0, 1]; labels: 1=genuine, 0=forged."""
        self.model.fit(scores.reshape(-1, 1), labels)
        self._fitted = True
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PlattCalibrator must be fit() before use")
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> PlattCalibrator:
        instance = cls()
        instance.model = joblib.load(path)
        instance._fitted = True
        return instance


class ScoreBasedLikelihoodRatio:
    """Histogram-binned SLR calibrator following the forensic score-based LR framework."""

    def __init__(self, num_bins: int = 20, score_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.num_bins = num_bins
        self.score_range = score_range
        self.bin_edges: np.ndarray | None = None
        self.genuine_density: np.ndarray | None = None
        self.forgery_density: np.ndarray | None = None

    def fit(self, genuine_scores: np.ndarray, forgery_scores: np.ndarray) -> ScoreBasedLikelihoodRatio:
        self.bin_edges = np.linspace(*self.score_range, self.num_bins + 1)
        # Laplace-smoothed histograms so LR never divides by exactly zero.
        genuine_hist, _ = np.histogram(genuine_scores, bins=self.bin_edges)
        forgery_hist, _ = np.histogram(forgery_scores, bins=self.bin_edges)
        self.genuine_density = (genuine_hist + 1) / (genuine_hist.sum() + self.num_bins)
        self.forgery_density = (forgery_hist + 1) / (forgery_hist.sum() + self.num_bins)
        return self

    def _bin_index(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(np.digitize(scores, self.bin_edges[1:-1]), 0, self.num_bins - 1)

    def likelihood_ratio(self, scores: np.ndarray) -> np.ndarray:
        if self.genuine_density is None:
            raise RuntimeError("ScoreBasedLikelihoodRatio must be fit() before use")
        idx = self._bin_index(scores)
        return self.genuine_density[idx] / self.forgery_density[idx]

    def posterior_probability(self, scores: np.ndarray, prior_odds: float = 1.0) -> np.ndarray:
        """P(genuine | score) via Bayes: posterior_odds = LR * prior_odds."""
        lr = self.likelihood_ratio(scores)
        posterior_odds = lr * prior_odds
        return posterior_odds / (1.0 + posterior_odds)

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"bin_edges": self.bin_edges, "genuine_density": self.genuine_density, "forgery_density": self.forgery_density},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> ScoreBasedLikelihoodRatio:
        payload = joblib.load(path)
        instance = cls(num_bins=len(payload["bin_edges"]) - 1)
        instance.bin_edges = payload["bin_edges"]
        instance.genuine_density = payload["genuine_density"]
        instance.forgery_density = payload["forgery_density"]
        return instance
