#!/usr/bin/env python
"""Convert the real, publicly-downloadable CEDAR (static) and MOBISIG (dynamic)
signature datasets into sigverify manifests.

Both are genuine public research datasets fetched directly (no gated request form):

* CEDAR — 55 writers, 24 genuine + 24 skilled forgeries each. Mirror used:
  https://github.com/nikostsagk/signature-verification (original: cedar.buffalo.edu).
* MOBISIG — 83 writers, finger-drawn signatures on a capacitive touchscreen, 45
  genuine + ~20 skilled forgeries each. Source: https://www.ms.sapientia.ro/~manyi/mobisig.html

Usage:
    python scripts/prepare_real_datasets.py --max-writers 15
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CEDAR_DIR = REPO_ROOT / "data" / "raw" / "cedar" / "extracted"
MOBISIG_DIR = REPO_ROOT / "data" / "raw" / "mobisig" / "extracted"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "real"

CEDAR_GENUINE_RE = re.compile(r"original_(\d+)_(\d+)\.png")
CEDAR_FORGED_RE = re.compile(r"forgeries_(\d+)_(\d+)\.png")
MOBISIG_GENUINE_RE = re.compile(r"SIGN_GEN_USER(\d+)_USER\d+_(\d+)\.csv")
MOBISIG_FORGED_RE = re.compile(r"SIGN_FOR_USER(\d+)_USER\d+_(\d+)\.csv")


def build_cedar_manifest(max_writers: int | None) -> Path:
    records = []
    writers_seen: set[str] = set()

    for pattern, label, folder in [
        (CEDAR_GENUINE_RE, "genuine", "full_org"),
        (CEDAR_FORGED_RE, "forged", "full_forg"),
    ]:
        for path in sorted((CEDAR_DIR / folder).glob("*.png")):
            match = pattern.match(path.name)
            if not match:
                continue
            writer_id = f"cedar_writer_{match.group(1)}"
            writers_seen.add(writer_id)
            records.append({"path": str(path), "writer_id": writer_id, "label": label})

    allowed = writers_seen
    if max_writers is not None:
        allowed = set(sorted(writers_seen, key=lambda w: int(w.rsplit("_", 1)[1]))[:max_writers])
        records = [r for r in records if r["writer_id"] in allowed]

    manifest_path = OUTPUT_DIR / "cedar_static_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)
    print(f"CEDAR: {len(records)} samples across {len(allowed)} writers -> {manifest_path}")
    return manifest_path


def _csv_to_stroke_json(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    stroke = {
        "x": df["x"].tolist(),
        "y": df["y"].tolist(),
        "timestamp": df["timestamp"].tolist(),
        "pressure": df["pressure"].tolist() if "pressure" in df.columns else [1.0] * len(df),
        # MOBISIG is finger-drawn on a capacitive touchscreen — there is no stylus
        # tilt sensor, so tilt_x/tilt_y are genuinely absent (left at 0), not estimated.
        "tilt_x": [0.0] * len(df),
        "tilt_y": [0.0] * len(df),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(stroke, fh)


def build_mobisig_manifest(max_writers: int | None) -> Path:
    user_dirs = sorted(
        (p for p in MOBISIG_DIR.iterdir() if p.is_dir() and p.name.startswith("USER")),
        key=lambda p: int(p.name.replace("USER", "")),
    )
    if max_writers is not None:
        user_dirs = user_dirs[:max_writers]

    records = []
    stroke_json_root = OUTPUT_DIR / "mobisig_strokes"

    for user_dir in user_dirs:
        writer_id = f"mobisig_{user_dir.name.lower()}"
        for pattern, label in [(MOBISIG_GENUINE_RE, "genuine"), (MOBISIG_FORGED_RE, "forged")]:
            for csv_path in sorted(user_dir.glob("*.csv")):
                match = pattern.match(csv_path.name)
                if not match:
                    continue
                out_path = stroke_json_root / writer_id / label / (csv_path.stem + ".json")
                _csv_to_stroke_json(csv_path, out_path)
                records.append({"path": str(out_path), "writer_id": writer_id, "label": label})

    manifest_path = OUTPUT_DIR / "mobisig_dynamic_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)
    print(f"MOBISIG: {len(records)} samples across {len(user_dirs)} writers -> {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-writers", type=int, default=15, help="Cap per-dataset writer count for fast notebook runs; omit/None for the full dataset")
    parser.add_argument("--full", action="store_true", help="Use every writer in both datasets (overrides --max-writers)")
    args = parser.parse_args()

    max_writers = None if args.full else args.max_writers
    build_cedar_manifest(max_writers)
    build_mobisig_manifest(max_writers)


if __name__ == "__main__":
    main()
