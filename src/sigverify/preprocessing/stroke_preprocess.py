"""Dynamic (online) signature stroke preprocessing.

Raw capture from a touchscreen/stylus device is a variable-length sequence of
(x, y, pressure, tilt_x, tilt_y, timestamp) samples. This module derives velocity,
resamples every sequence to a fixed length (so batches can be tensorized), and
z-score/min-max normalizes each channel independently.
"""
from __future__ import annotations

import numpy as np

DEFAULT_FEATURES = ("x", "y", "pressure", "velocity", "tilt_x", "tilt_y", "timestamp_delta")


def _resample_1d(values: np.ndarray, num_points: int) -> np.ndarray:
    if len(values) == num_points:
        return values
    original_idx = np.linspace(0.0, 1.0, num=len(values))
    target_idx = np.linspace(0.0, 1.0, num=num_points)
    return np.interp(target_idx, original_idx, values)


def compute_velocity(x: np.ndarray, y: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
    dt = np.diff(timestamp, prepend=timestamp[0])
    dt[dt == 0] = 1e-6
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    return np.sqrt(dx**2 + dy**2) / dt


def stroke_to_feature_matrix(stroke: dict) -> np.ndarray:
    """Convert a raw stroke dict (parallel arrays) into an (N, len(DEFAULT_FEATURES)) matrix."""
    x = np.asarray(stroke["x"], dtype=np.float64)
    y = np.asarray(stroke["y"], dtype=np.float64)
    timestamp = np.asarray(stroke.get("timestamp", np.arange(len(x))), dtype=np.float64)
    pressure = np.asarray(stroke.get("pressure", np.ones_like(x)), dtype=np.float64)
    tilt_x = np.asarray(stroke.get("tilt_x", np.zeros_like(x)), dtype=np.float64)
    tilt_y = np.asarray(stroke.get("tilt_y", np.zeros_like(x)), dtype=np.float64)

    velocity = compute_velocity(x, y, timestamp)
    timestamp_delta = np.diff(timestamp, prepend=timestamp[0])

    return np.stack([x, y, pressure, velocity, tilt_x, tilt_y, timestamp_delta], axis=1)


def normalize_features(matrix: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "zscore":
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        return (matrix - mean) / std
    if method == "minmax":
        lo = matrix.min(axis=0, keepdims=True)
        hi = matrix.max(axis=0, keepdims=True)
        span = np.clip(hi - lo, 1e-8, None)
        return (matrix - lo) / span
    raise ValueError(f"Unknown normalization method: {method}")


def preprocess_stroke_sequence(
    stroke: dict,
    resample_points: int = 256,
    normalize: str = "zscore",
) -> np.ndarray:
    """Raw stroke dict -> fixed-length (resample_points, 7) float32 normalized array."""
    matrix = stroke_to_feature_matrix(stroke)
    resampled = np.stack(
        [_resample_1d(matrix[:, i], resample_points) for i in range(matrix.shape[1])], axis=1
    )
    normalized = normalize_features(resampled, method=normalize)
    return normalized.astype(np.float32)
