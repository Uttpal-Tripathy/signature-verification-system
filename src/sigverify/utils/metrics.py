"""Verification-specific metrics: EER, FAR/FRR curves, and calibrated accuracy.

These are the standard evaluation metrics in the signature-verification literature
(CEDAR / GPDS / SVC2004 / DeepSignDB benchmarks all report EER), so results produced
here are directly comparable to published baselines.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def far_frr(genuine_scores: np.ndarray, forgery_scores: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False Accept Rate / False Reject Rate swept across a threshold grid."""
    far = np.array([(forgery_scores >= t).mean() for t in thresholds])
    frr = np.array([(genuine_scores < t).mean() for t in thresholds])
    return far, frr


def equal_error_rate(genuine_scores: np.ndarray, forgery_scores: np.ndarray, num_thresholds: int = 1000) -> dict:
    """Equal Error Rate: the point where FAR == FRR. Lower is better (0 == perfect)."""
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    far, frr = far_frr(genuine_scores, forgery_scores, thresholds)
    diff = np.abs(far - frr)
    idx = int(np.argmin(diff))
    eer = float((far[idx] + frr[idx]) / 2.0)
    return {"eer": eer, "threshold": float(thresholds[idx]), "far": float(far[idx]), "frr": float(frr[idx])}


def roc_auc(genuine_scores: np.ndarray, forgery_scores: np.ndarray) -> float:
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(forgery_scores)])
    y_score = np.concatenate([genuine_scores, forgery_scores])
    return float(roc_auc_score(y_true, y_score))


def verification_report(genuine_scores: np.ndarray, forgery_scores: np.ndarray) -> dict:
    """One-call summary used by evaluate.py and unit tests."""
    eer_stats = equal_error_rate(genuine_scores, forgery_scores)
    auc = roc_auc(genuine_scores, forgery_scores)
    threshold = eer_stats["threshold"]
    accuracy = float(
        (
            (genuine_scores >= threshold).sum() + (forgery_scores < threshold).sum()
        )
        / (len(genuine_scores) + len(forgery_scores))
    )
    return {**eer_stats, "roc_auc": auc, "accuracy_at_eer_threshold": accuracy}


def fpr_tpr_curve(genuine_scores: np.ndarray, forgery_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(forgery_scores)])
    y_score = np.concatenate([genuine_scores, forgery_scores])
    return roc_curve(y_true, y_score)
