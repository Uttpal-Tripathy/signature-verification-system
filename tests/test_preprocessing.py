import numpy as np

from sigverify.preprocessing.image_preprocess import preprocess_signature_image
from sigverify.preprocessing.stroke_preprocess import preprocess_stroke_sequence


def test_preprocess_signature_image_shape_and_range(sample_image):
    out = preprocess_signature_image(sample_image, target_size=(224, 224))
    assert out.shape == (224, 224)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_stroke_sequence_shape_and_normalization(sample_stroke):
    out = preprocess_stroke_sequence(sample_stroke, resample_points=100, normalize="zscore")
    assert out.shape == (100, 7)
    assert np.isfinite(out).all()
    # z-score: each channel should be roughly centered near 0.
    assert np.all(np.abs(out.mean(axis=0)) < 1e-3 + 1.0)


def test_preprocess_stroke_sequence_minmax(sample_stroke):
    out = preprocess_stroke_sequence(sample_stroke, resample_points=50, normalize="minmax")
    assert out.shape == (50, 7)
    assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6
