"""
extract_features.py — DAiSEE videos -> per-frame face/gaze/pose features.

Runs the same MediaPipe pipelines MHBAP uses live, on T uniformly sampled
frames per 10 s clip, and writes one .npz per official split.

Usage:
  python -m scripts.daisee.extract_features --root /path/to/DAiSEE \
      --out data/daisee_features --t 32
  # quick check on a few clips:
  python -m scripts.daisee.extract_features --root ... --limit 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ml.training.DaiseeDataset import VisualExtractor, extract_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="unzipped DAiSEE folder")
    ap.add_argument("--out", type=Path, default=Path("data/daisee_features"))
    ap.add_argument("--t", type=int, default=32, help="frames sampled per clip")
    ap.add_argument("--limit", type=int, default=None, help="clips per split (debug)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ex = VisualExtractor()
    for split in args.splits:
        target = args.out / f"{split}.npz"
        if target.exists() and args.limit is None:
            print(f"[skip] {target} exists")
            continue
        arrays = extract_split(args.root, split, t=args.t, limit=args.limit, extractor=ex)
        np.savez_compressed(target, **arrays)
        print(f"[done] {split}: {len(arrays['y'])} clips, "
              f"face found in {100 * arrays['valid'].mean():.1f}% of frames -> {target}")


if __name__ == "__main__":
    main()
