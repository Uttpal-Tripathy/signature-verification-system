import numpy as np

from sigverify.utils.metrics import equal_error_rate, roc_auc, verification_report


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
