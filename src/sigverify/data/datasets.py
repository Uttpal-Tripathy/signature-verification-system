"""Manifest-driven datasets for the static and dynamic branches.

A manifest is a JSON-lines file, one record per sample:

    {"path": "data/processed/static/writer_003/genuine/007.png", "writer_id": "writer_003", "label": "genuine"}
    {"path": "data/processed/dynamic/writer_003/forged/012.json",  "writer_id": "writer_003", "label": "forged"}

`label` is one of "genuine" | "forged" (skilled or random forgery — the manifest can
add a `forgery_type` field for finer-grained bookkeeping, ignored by the loader).
This indirection lets any public dataset (CEDAR, GPDS, ICDAR, SVC2004, MOBISIG,
DeepSignDB, ...) plug in via a small per-dataset manifest-building script instead of
each having its own Dataset subclass — see `scripts/generate_demo_data.py` for a
worked example that builds both a manifest and the files it points to.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from sigverify.preprocessing.image_preprocess import (
    binarize,
    crop_to_content,
    denoise,
    deskew,
    random_affine_jitter,
    resize_and_normalize,
    to_grayscale,
)
from sigverify.preprocessing.stroke_preprocess import preprocess_stroke_sequence


def load_manifest(manifest_path: str | Path) -> list[dict]:
    records = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_writers(manifest_path: str | Path, val_fraction: float = 0.2, seed: int = 42) -> tuple[set[str], set[str]]:
    """Writer-disjoint train/val split: no writer appears in both splits, so validation
    EER reflects generalization to unseen writers rather than memorized handwriting.
    """
    records = load_manifest(manifest_path)
    writer_ids = sorted({r["writer_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(writer_ids)
    num_val = max(1, int(len(writer_ids) * val_fraction))
    val_writers = set(writer_ids[:num_val])
    train_writers = set(writer_ids[num_val:])
    return train_writers, val_writers


def kfold_writers(
    manifest_path: str | Path, k: int = 5, seed: int = 42, max_writers: int | None = None,
) -> list[tuple[set[str], set[str]]]:
    """Writer-disjoint K-fold split: every writer is held out exactly once, so the K
    (train_writers, val_writers) pairs together cover every writer's EER/AUC/accuracy
    under an unseen-writer evaluation, and mean +/- std across folds is a far more
    defensible number than a single arbitrary train/val split (the standard concern with
    small writer counts: one split's "best" checkpoint can be a lucky or unlucky sample).
    `max_writers` caps the writer pool (deterministically, after shuffling) to bound
    training time on CPU-only hardware — state this cap explicitly wherever results using
    it are reported.
    """
    records = load_manifest(manifest_path)
    writer_ids = sorted({r["writer_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(writer_ids)
    if max_writers is not None:
        writer_ids = writer_ids[:max_writers]
    if k > len(writer_ids):
        raise ValueError(f"k={k} folds requested but only {len(writer_ids)} writers available")

    folds: list[list[str]] = [[] for _ in range(k)]
    for i, writer in enumerate(writer_ids):
        folds[i % k].append(writer)

    splits = []
    all_writers = set(writer_ids)
    for fold in folds:
        val_writers = set(fold)
        train_writers = all_writers - val_writers
        splits.append((train_writers, val_writers))
    return splits


class StaticSignatureTripletDataset(Dataset):
    """Yields (anchor, positive, negative) image triplets: anchor/positive are two
    genuine signatures from the same writer, negative is either a forgery of that
    writer or a genuine signature from a different writer (impostor).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        target_size: tuple[int, int] = (224, 224),
        impostor_ratio: float = 0.5,
        writer_ids: set[str] | None = None,
        cache_in_memory: bool = False,
        augment: bool = False,
        binarize_method: str = "otsu",
        deskew_enabled: bool = True,
        denoise_h: int = 10,
        seed: int = 0,
        shared_denoise_cache: dict[str, np.ndarray] | None = None,
    ):
        self.records = load_manifest(manifest_path)
        if writer_ids is not None:
            self.records = [r for r in self.records if r["writer_id"] in writer_ids]
        self.target_size = target_size
        self.impostor_ratio = impostor_ratio
        self.binarize_method = binarize_method
        self.deskew_enabled = deskew_enabled
        self.denoise_h = denoise_h
        # Random rotation/translation/scale applied before binarization, each call
        # freshly reseeded (unlike the cached denoise step below) — the whole point of
        # augmentation is that the model sees a different jitter of the same signature
        # every epoch, which is what actually fights the overfitting documented in
        # docs/results.md (best checkpoint being epoch 1 with too little raw data).
        # Never applied at evaluation/inference time (only this triplet-training path
        # uses it).
        self.augment = augment
        self._rng = np.random.default_rng(seed)
        # Denoising dominates preprocessing cost (~0.4-1.6s/image) and is re-paid on
        # every __getitem__ call by default, i.e. every epoch. Caching the DENOISED
        # (but not yet binarized/deskewed/augmented) image lets every epoch after the
        # first skip straight to the cheap steps, while still being compatible with
        # fresh-per-epoch augmentation (which must run after the cached step, not be
        # baked into it). `shared_denoise_cache` lets several dataset instances that draw
        # from the same underlying image pool (e.g. one per fold in k-fold
        # cross-validation, where the same writer's images appear in several folds'
        # training sets) share one cache by reference instead of each re-denoising the
        # same files from scratch.
        self.cache_in_memory = cache_in_memory
        self._denoised_cache: dict[str, np.ndarray] = shared_denoise_cache if shared_denoise_cache is not None else {}

        self.by_writer_genuine: dict[str, list[dict]] = {}
        self.by_writer_forged: dict[str, list[dict]] = {}
        for rec in self.records:
            bucket = self.by_writer_genuine if rec["label"] == "genuine" else self.by_writer_forged
            bucket.setdefault(rec["writer_id"], []).append(rec)

        self.anchors = [
            rec for writer, recs in self.by_writer_genuine.items() if len(recs) >= 2 for rec in recs
        ]
        self.writers = list(self.by_writer_genuine.keys())

    def __len__(self) -> int:
        return len(self.anchors)

    def _denoised(self, path: str) -> np.ndarray:
        if self.cache_in_memory and path in self._denoised_cache:
            return self._denoised_cache[path]
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(path)
        gray = denoise(to_grayscale(raw), h=self.denoise_h)
        if self.cache_in_memory:
            self._denoised_cache[path] = gray
        return gray

    def _load_image(self, path: str) -> torch.Tensor:
        gray = self._denoised(path)
        if self.augment:
            gray = random_affine_jitter(gray, self._rng)
        binary = binarize(gray, method=self.binarize_method)
        if self.deskew_enabled:
            binary = deskew(binary)
        cropped = crop_to_content(binary)
        processed = resize_and_normalize(cropped, self.target_size)
        return torch.from_numpy(processed).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_rec = self.anchors[idx]
        writer = anchor_rec["writer_id"]
        genuine_pool = [r for r in self.by_writer_genuine[writer] if r["path"] != anchor_rec["path"]]
        positive_rec = random.choice(genuine_pool) if genuine_pool else anchor_rec

        use_skilled_forgery = self.by_writer_forged.get(writer) and random.random() > self.impostor_ratio
        if use_skilled_forgery:
            negative_rec = random.choice(self.by_writer_forged[writer])
        else:
            other_writer = random.choice([w for w in self.writers if w != writer] or [writer])
            negative_rec = random.choice(self.by_writer_genuine[other_writer])

        return self._load_image(anchor_rec["path"]), self._load_image(positive_rec["path"]), self._load_image(negative_rec["path"])


class DynamicStrokeDataset(Dataset):
    """Yields (stroke_tensor, label) where label is 1=genuine, 0=forged, for a
    verification pair pre-formed by the manifest (`reference_path` + `query_path`
    fields), or raw per-sample stroke tensors when only feature extraction is needed.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        resample_points: int = 256,
        normalize: str = "zscore",
        writer_ids: set[str] | None = None,
    ):
        self.records = load_manifest(manifest_path)
        if writer_ids is not None:
            self.records = [r for r in self.records if r["writer_id"] in writer_ids]
        self.resample_points = resample_points
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.records)

    def _load_stroke(self, path: str) -> torch.Tensor:
        with open(path, "r", encoding="utf-8") as fh:
            stroke = json.load(fh)
        matrix = preprocess_stroke_sequence(stroke, self.resample_points, self.normalize)
        return torch.from_numpy(matrix)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        label = 1 if rec["label"] == "genuine" else 0
        return self._load_stroke(rec["path"]), label


class DynamicStrokeTripletDataset(Dataset):
    """Triplet version of DynamicStrokeDataset (anchor/positive genuine same writer,
    negative forged-or-impostor) for metric-learning training of the dynamic branch,
    mirroring StaticSignatureTripletDataset.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        resample_points: int = 256,
        normalize: str = "zscore",
        impostor_ratio: float = 0.5,
        writer_ids: set[str] | None = None,
    ):
        self.records = load_manifest(manifest_path)
        if writer_ids is not None:
            self.records = [r for r in self.records if r["writer_id"] in writer_ids]
        self.resample_points = resample_points
        self.normalize = normalize
        self.impostor_ratio = impostor_ratio

        self.by_writer_genuine: dict[str, list[dict]] = {}
        self.by_writer_forged: dict[str, list[dict]] = {}
        for rec in self.records:
            bucket = self.by_writer_genuine if rec["label"] == "genuine" else self.by_writer_forged
            bucket.setdefault(rec["writer_id"], []).append(rec)

        self.anchors = [rec for writer, recs in self.by_writer_genuine.items() if len(recs) >= 2 for rec in recs]
        self.writers = list(self.by_writer_genuine.keys())

    def __len__(self) -> int:
        return len(self.anchors)

    def _load_stroke(self, path: str) -> torch.Tensor:
        with open(path, "r", encoding="utf-8") as fh:
            stroke = json.load(fh)
        matrix = preprocess_stroke_sequence(stroke, self.resample_points, self.normalize)
        return torch.from_numpy(matrix)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_rec = self.anchors[idx]
        writer = anchor_rec["writer_id"]
        genuine_pool = [r for r in self.by_writer_genuine[writer] if r["path"] != anchor_rec["path"]]
        positive_rec = random.choice(genuine_pool) if genuine_pool else anchor_rec

        use_skilled_forgery = self.by_writer_forged.get(writer) and random.random() > self.impostor_ratio
        if use_skilled_forgery:
            negative_rec = random.choice(self.by_writer_forged[writer])
        else:
            other_writer = random.choice([w for w in self.writers if w != writer] or [writer])
            negative_rec = random.choice(self.by_writer_genuine[other_writer])

        return self._load_stroke(anchor_rec["path"]), self._load_stroke(positive_rec["path"]), self._load_stroke(negative_rec["path"])
