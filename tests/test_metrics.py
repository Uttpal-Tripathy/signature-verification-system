import numpy as np

from sigverify.utils.metrics import (
    confusion_matrix_at_threshold,
    equal_error_rate,
    evaluation_matrix,
    roc_auc,
    verification_report,
)


def test_eer_perfect_separation():
    genuine = np.full(200, 0.95)
    forgery = np.full(200, 0.05)
    result = equal_error_rate(genuine, forgery)
    assert result["eer"] < 1e-3


def test_eer_overlapping_distributions_is_worse_than_separated():
    rng = np.random.default_rng(0)
    genuine_sep = rng.normal(0.9, 0.02, 500)
    forgery_sep = rng.normal(0.1, 0.02, 500)
    separated = equal_error_rate(np.clip(genuine_sep, 0, 1), np.clip(forgery_sep, 0, 1))

    genuine_overlap = rng.normal(0.55, 0.15, 500)
    forgery_overlap = rng.normal(0.45, 0.15, 500)
    overlapping = equal_error_rate(np.clip(genuine_overlap, 0, 1), np.clip(forgery_overlap, 0, 1))

    assert separated["eer"] < overlapping["eer"]


def test_roc_auc_perfect_separation_is_one():
    genuine = np.full(100, 0.9)
    forgery = np.full(100, 0.1)
    assert roc_auc(genuine, forgery) == 1.0


def test_verification_report_keys():
    genuine = np.random.default_rng(1).uniform(0.6, 1.0, 100)
    forgery = np.random.default_rng(2).uniform(0.0, 0.4, 100)
    report = verification_report(genuine, forgery)
    for key in ("eer", "threshold", "far", "frr", "roc_auc", "accuracy_at_eer_threshold"):
        assert key in report


def test_confusion_matrix_perfect_separation():
    genuine = np.full(50, 0.9)
    forgery = np.full(50, 0.1)
    cm = confusion_matrix_at_threshold(genuine, forgery, threshold=0.5)
    assert cm == {"tp": 50, "fn": 0, "fp": 0, "tn": 50, "threshold": 0.5}


def test_confusion_matrix_counts_sum_to_sample_sizes():
    rng = np.random.default_rng(3)
    genuine = np.clip(rng.normal(0.7, 0.2, 80), 0, 1)
    forgery = np.clip(rng.normal(0.3, 0.2, 60), 0, 1)
    cm = confusion_matrix_at_threshold(genuine, forgery, threshold=0.5)
    assert cm["tp"] + cm["fn"] == 80
    assert cm["fp"] + cm["tn"] == 60


def test_evaluation_matrix_keys_and_ranges():
    genuine = np.random.default_rng(4).uniform(0.6, 1.0, 100)
    forgery = np.random.default_rng(5).uniform(0.0, 0.4, 100)
    m = evaluation_matrix(genuine, forgery)
    for key in ("tp", "fn", "fp", "tn", "precision", "recall", "specificity", "f1_score", "accuracy", "far", "frr", "roc_auc", "eer"):
        assert key in m
    for key in ("precision", "recall", "specificity", "f1_score", "accuracy", "far", "frr"):
        assert 0.0 <= m[key] <= 1.0


def test_evaluation_matrix_perfect_separation_is_all_ones():
    genuine = np.full(40, 0.95)
    forgery = np.full(40, 0.05)
    m = evaluation_matrix(genuine, forgery)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["specificity"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["accuracy"] == 1.0
