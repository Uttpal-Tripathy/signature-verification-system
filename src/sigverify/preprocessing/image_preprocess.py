"""Static signature image preprocessing.

Pipeline: grayscale -> denoise -> deskew -> binarize -> crop-to-content -> resize/normalize.
Implements the "Preprocessing" stage of the architecture (noise removal, skew correction,
binarization, normalization) ahead of the static/dynamic feature-extraction branches.
"""
from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray, h: int = 10) -> np.ndarray:
    return cv2.fastNlMeansDenoising(image, None, h=h, templateWindowSize=7, searchWindowSize=21)


def binarize(image: np.ndarray, method: str = "otsu") -> np.ndarray:
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
        )
    elif method == "sauvola":
        from skimage.filters import threshold_sauvola

        thresh = threshold_sauvola(image, window_size=25)
        binary = ((image < thresh) * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown binarize method: {method}")
    return binary


def estimate_skew_angle(binary: np.ndarray) -> float:
    """Estimate rotation angle (degrees) of ink strokes via minAreaRect over foreground pixels."""
    coords = np.column_stack(np.where(binary > 0))
    if coords.shape[0] < 10:
        return 0.0
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle)


def deskew(binary: np.ndarray, angle: float | None = None) -> np.ndarray:
    if angle is None:
        angle = estimate_skew_angle(binary)
    if abs(angle) < 0.5:
        return binary
    h, w = binary.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(binary, rot_matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)


def crop_to_content(binary: np.ndarray, padding: int = 10) -> np.ndarray:
    coords = cv2.findNonZero(binary)
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    y0, y1 = max(0, y - padding), min(binary.shape[0], y + h + padding)
    x0, x1 = max(0, x - padding), min(binary.shape[1], x + w + padding)
    return binary[y0:y1, x0:x1]


def resize_and_normalize(binary: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    resized = cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def preprocess_signature_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    binarize_method: str = "otsu",
    deskew_enabled: bool = True,
    denoise_h: int = 10,
) -> np.ndarray:
    """Full pipeline: raw scanned/camera image -> normalized (H, W) float32 array in [0, 1]."""
    gray = to_grayscale(image)
    gray = denoise(gray, h=denoise_h)
    binary = binarize(gray, method=binarize_method)
    if deskew_enabled:
        binary = deskew(binary)
    cropped = crop_to_content(binary)
    return resize_and_normalize(cropped, target_size)
