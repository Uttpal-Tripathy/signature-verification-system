"""Synthetic signature generator.

Real forensic accuracy requires training on real datasets (CEDAR/GPDS/ICDAR for
static, SVC2004/MOBISIG/DeepSignDB for dynamic — see data/README.md). This module
exists purely so the full pipeline (preprocessing -> both branches -> fusion ->
anomaly -> calibration -> explainability -> report) is exercisable end-to-end without
any external download: for CI smoke tests, local demos, and unit tests. Numbers
produced by a model trained only on this synthetic data are not meaningful accuracy
claims.

Each synthetic "writer" is a deterministic random curve (a sum of a few sine waves
with writer-specific frequency/phase/amplitude, seeded from the writer id) rendered
either as a static stroke image or as an (x, y, pressure, tilt, timestamp) sequence.
Genuine samples add small jitter; forged samples resample a *different* writer's
curve parameters warped toward the target writer's bounding box, so genuine-vs-forged
is a real (if easy) discrimination task rather than pure noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _writer_curve_params(writer_seed: int, num_harmonics: int = 4) -> dict:
    rng = np.random.default_rng(writer_seed)
    return {
        "freqs": rng.uniform(1.0, 4.0, size=num_harmonics),
        "phases": rng.uniform(0, 2 * np.pi, size=num_harmonics),
        "amps": rng.uniform(0.2, 1.0, size=num_harmonics),
        "y_offset_freqs": rng.uniform(0.5, 3.0, size=num_harmonics),
    }


def _sample_curve(params: dict, num_points: int, rng: np.random.Generator, jitter: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2 * np.pi, num_points)
    x = t / (2 * np.pi)
    y = np.zeros_like(t)
    for freq, phase, amp, yfreq in zip(params["freqs"], params["phases"], params["amps"], params["y_offset_freqs"]):
        y += amp * np.sin(freq * t + phase) * np.cos(yfreq * t * 0.3)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    if jitter > 0:
        x = x + rng.normal(0, jitter, size=x.shape)
        y = y + rng.normal(0, jitter, size=y.shape)
    return x, y


def generate_synthetic_signature_image(
    writer_id: str, sample_seed: int, forged: bool = False, size: tuple[int, int] = (300, 900)
) -> np.ndarray:
    """Returns a (H, W) uint8 grayscale image with a hand-writing-like stroke on white."""
    writer_seed = abs(hash(writer_id)) % (2**31)
    rng = np.random.default_rng(sample_seed)

    params = _writer_curve_params(rng.integers(0, 2**31) if forged else writer_seed)
    x, y = _sample_curve(params, num_points=400, rng=rng, jitter=0.01 if not forged else 0.03)

    h, w = size
    canvas = np.full((h, w), 255, dtype=np.uint8)
    px = (x * (w - 40) + 20).astype(np.int32)
    py = (y * (h - 60) + 30).astype(np.int32)
    thickness = rng.integers(2, 4)
    for i in range(1, len(px)):
        cv2.line(canvas, (px[i - 1], py[i - 1]), (px[i], py[i]), color=0, thickness=int(thickness))
    return canvas


def generate_synthetic_stroke(writer_id: str, sample_seed: int, forged: bool = False, num_points: int = 300) -> dict:
    """Returns a raw stroke dict compatible with `stroke_preprocess.stroke_to_feature_matrix`."""
    writer_seed = abs(hash(writer_id)) % (2**31)
    rng = np.random.default_rng(sample_seed)

    params = _writer_curve_params(rng.integers(0, 2**31) if forged else writer_seed)
    x, y = _sample_curve(params, num_points=num_points, rng=rng, jitter=0.01 if not forged else 0.03)

    timestamp = np.linspace(0, 3.0, num_points) + rng.normal(0, 0.002, num_points).cumsum() * 0.01
    pressure = 0.5 + 0.4 * np.sin(np.linspace(0, 6 * np.pi, num_points)) + rng.normal(0, 0.03, num_points)
    tilt_x = rng.normal(0, 5, num_points)
    tilt_y = rng.normal(0, 5, num_points)

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "pressure": np.clip(pressure, 0, 1).tolist(),
        "tilt_x": tilt_x.tolist(),
        "tilt_y": tilt_y.tolist(),
        "timestamp": timestamp.tolist(),
    }


def build_demo_dataset(
    output_dir: str | Path,
    num_writers: int = 8,
    genuine_per_writer: int = 8,
    forged_per_writer: int = 4,
    seed: int = 42,
) -> dict:
    """Writes synthetic static images + dynamic strokes to disk and produces JSONL
    manifests. Returns the paths to the two manifests.
    """
    output_dir = Path(output_dir)
    static_root = output_dir / "static"
    dynamic_root = output_dir / "dynamic"
    static_manifest_path = output_dir / "static_manifest.jsonl"
    dynamic_manifest_path = output_dir / "dynamic_manifest.jsonl"

    rng = np.random.default_rng(seed)
    static_records, dynamic_records = [], []

    for w in range(num_writers):
        writer_id = f"writer_{w:03d}"
        for split, forged, count in (("genuine", False, genuine_per_writer), ("forged", True, forged_per_writer)):
            img_dir = static_root / writer_id / split
            stroke_dir = dynamic_root / writer_id / split
            img_dir.mkdir(parents=True, exist_ok=True)
            stroke_dir.mkdir(parents=True, exist_ok=True)

            for i in range(count):
                sample_seed = int(rng.integers(0, 2**31))

                image = generate_synthetic_signature_image(writer_id, sample_seed, forged=forged)
                img_path = img_dir / f"{i:03d}.png"
                cv2.imwrite(str(img_path), image)
                static_records.append({"path": str(img_path), "writer_id": writer_id, "label": split})

                stroke = generate_synthetic_stroke(writer_id, sample_seed, forged=forged)
                stroke_path = stroke_dir / f"{i:03d}.json"
                with open(stroke_path, "w", encoding="utf-8") as fh:
                    json.dump(stroke, fh)
                dynamic_records.append({"path": str(stroke_path), "writer_id": writer_id, "label": split})

    with open(static_manifest_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in static_records)
    with open(dynamic_manifest_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in dynamic_records)

    return {"static_manifest": str(static_manifest_path), "dynamic_manifest": str(dynamic_manifest_path)}
