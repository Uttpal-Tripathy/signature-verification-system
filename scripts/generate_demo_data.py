#!/usr/bin/env python
"""Generate a small synthetic dataset so the full pipeline can be trained/run/tested
end-to-end without any external download. See sigverify.data.synthetic for caveats:
this is for wiring/smoke-testing, not for producing meaningful accuracy numbers.

Usage:
    python scripts/generate_demo_data.py --output data/processed/demo --num-writers 12
"""
from __future__ import annotations

import argparse

from sigverify.data.synthetic import build_demo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/processed/demo")
    parser.add_argument("--num-writers", type=int, default=12)
    parser.add_argument("--genuine-per-writer", type=int, default=10)
    parser.add_argument("--forged-per-writer", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = build_demo_dataset(
        args.output,
        num_writers=args.num_writers,
        genuine_per_writer=args.genuine_per_writer,
        forged_per_writer=args.forged_per_writer,
        seed=args.seed,
    )
    print("Static manifest: ", paths["static_manifest"])
    print("Dynamic manifest:", paths["dynamic_manifest"])


if __name__ == "__main__":
    main()
