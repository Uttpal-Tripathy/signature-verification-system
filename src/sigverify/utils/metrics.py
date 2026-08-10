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


def confusion_matrix_at_threshold(genuine_scores: np.ndarray, forgery_scores: np.ndarray, threshold: float) -> dict:
    """Binary confusion matrix at a fixed decision threshold. 'Positive' = genuine
    (score >= threshold accepts a signature as genuine). Uses whatever threshold the
    caller passes — typically the EER threshold from `equal_error_rate`, so the matrix
    is consistent with the EER/AUC numbers reported alongside it.
    """
    tp = int((genuine_scores >= threshold).sum())  # genuine correctly accepted
    fn = int((genuine_scores < threshold).sum())   # genuine wrongly rejected
    fp = int((forgery_scores >= threshold).sum())  # forgery wrongly accepted
    tn = int((forgery_scores < threshold).sum())   # forgery correctly rejected
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "threshold": float(threshold)}


def evaluation_matrix(genuine_scores: np.ndarray, forgery_scores: np.ndarray) -> dict:
    """Full evaluation metrics at the EER threshold: confusion matrix counts plus
    precision, recall (sensitivity/TPR), specificity (TNR), F1, and FAR/FRR restated
    as their confusion-matrix-derived equivalents (FAR == FPR, FRR == FNR) — the
    complete set a paper's results table or a patent disclosure's evaluation section
    would expect, not just EER/AUC.
    """
    eer_stats = equal_error_rate(genuine_scores, forgery_scores)
    cm = confusion_matrix_at_threshold(genuine_scores, forgery_scores, eer_stats["threshold"])
    tp, fn, fp, tn = cm["tp"], cm["fn"], cm["fp"], cm["tn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity / TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        **cm,
        "roc_auc": roc_auc(genuine_scores, forgery_scores),
        "eer": eer_stats["eer"],
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "accuracy": accuracy,
        "far": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "frr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }
