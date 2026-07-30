import numpy as np

from sigverify.pipeline.inference import verify_signature
from sigverify.pipeline.model_bundle import SignatureVerificationBundle


def test_verify_signature_static_only(default_config, sample_image, sample_forged_image):
    bundle = SignatureVerificationBundle(default_config)
    result = verify_signature(
        bundle,
        reference_image=sample_image,
        query_image=sample_forged_image,
        explain=True,
        estimate_confidence=False,
    )

    assert result.decision in ("Genuine", "Review", "Forged")
    assert -1.0001 <= result.static_similarity <= 1.0001
    assert result.dynamic_similarity is None
    assert 0.0 <= result.combined_score <= 1.0001
    assert result.static_heatmap is not None
    assert result.static_heatmap.shape[:2] == tuple(default_config["preprocessing"]["image"]["target_size"])
    assert result.modality_weights["dynamic_weight"] == 0.0


def test_verify_signature_with_dynamic_modality(default_config, sample_image, sample_forged_image, sample_stroke, sample_forged_stroke):
    bundle = SignatureVerificationBundle(default_config)
    result = verify_signature(
        bundle,
        reference_image=sample_image,
        query_image=sample_forged_image,
        reference_stroke=sample_stroke,
        query_stroke=sample_forged_stroke,
        explain=True,
        estimate_confidence=False,
    )

    assert result.dynamic_similarity is not None
    assert result.dynamic_deviation_scores is not None
    assert result.top_deviant_indices is not None
    assert len(result.top_deviant_indices) == default_config["explainability"]["top_k_deviant_strokes"]
    assert result.modality_weights["dynamic_weight"] > 0.0


def test_verify_signature_confidence_interval(default_config, sample_image, sample_forged_image):
    bundle = SignatureVerificationBundle(default_config)
    result = verify_signature(
        bundle,
        reference_image=sample_image,
        query_image=sample_forged_image,
        explain=False,
        estimate_confidence=True,
        tta_rounds=3,
    )
    assert result.confidence_interval is not None
    lo, hi = result.confidence_interval
    assert lo <= hi


def test_verify_signature_with_shap_background(default_config, sample_image, sample_forged_image):
    bundle = SignatureVerificationBundle(default_config)
    background = np.random.default_rng(0).uniform(0, 1, size=(6, 4))
    result = verify_signature(
        bundle,
        reference_image=sample_image,
        query_image=sample_forged_image,
        explain=True,
        estimate_confidence=False,
        shap_background=background,
    )
    assert result.shap_modality_split is not None
    assert "static_contribution_pct" in result.shap_modality_split
