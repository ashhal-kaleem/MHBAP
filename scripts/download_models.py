#!/usr/bin/env python3
"""
scripts/download_models.py
Download and cache all ML model weights required by MHBAP.

Usage:
    python scripts/download_models.py            # download all
    python scripts/download_models.py --check    # verify already downloaded
    python scripts/download_models.py --model bert  # single model

Models downloaded:
    bert       bert-base-uncased (text encoder)         ~420 MB
    wav2vec2   facebook/wav2vec2-base (audio encoder)   ~360 MB
    mediapipe  face_landmarker, pose_landmarker, etc.   ~40 MB total
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "bert": {
        "hf_name": "bert-base-uncased",
        "local_dir": MODEL_DIR / "bert-base-uncased",
        "size_hint": "~420 MB",
    },
    "wav2vec2": {
        "hf_name": "facebook/wav2vec2-base",
        "local_dir": MODEL_DIR / "wav2vec2-base",
        "size_hint": "~360 MB",
    },
}

MEDIAPIPE_MODELS = {
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "pose_landmarker": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face_detector":   "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
}

def _download_hf(name: str, cfg: dict) -> bool:
    """Download a HuggingFace model via transformers snapshot_download."""
    local_dir: Path = cfg["local_dir"]
    if (local_dir / "config.json").exists():
        print(f"  [SKIP] {name} already at {local_dir}")
        return True
    print(f"  [DL] {name} ({cfg['size_hint']}) → {local_dir}")
    try:
        from transformers import AutoTokenizer, AutoModel
        AutoTokenizer.from_pretrained(cfg["hf_name"], cache_dir=str(local_dir))
        AutoModel.from_pretrained(cfg["hf_name"], cache_dir=str(local_dir))
        print(f"  [OK] {name}")
        return True
    except Exception as e:
        print(f"  [ERR] {name}: {e}")
        return False


def _download_mediapipe() -> bool:
    """Download MediaPipe task files."""
    import urllib.request
    mp_dir = MODEL_DIR / "mediapipe"
    mp_dir.mkdir(exist_ok=True)
    ok = True
    for name, url in MEDIAPIPE_MODELS.items():
        dest = mp_dir / url.split("/")[-1]
        if dest.exists():
            print(f"  [SKIP] mediapipe/{name}")
            continue
        print(f"  [DL] mediapipe/{name}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  [OK] mediapipe/{name} → {dest}")
        except Exception as e:
            print(f"  [ERR] mediapipe/{name}: {e}")
            ok = False
    return ok


def _check() -> None:
    all_ok = True
    for name, cfg in MODELS.items():
        exists = (cfg["local_dir"] / "config.json").exists()
        status = "✅" if exists else "❌ MISSING"
        print(f"  {status}  {name}")
        if not exists:
            all_ok = False
    mp_dir = MODEL_DIR / "mediapipe"
    for name, url in MEDIAPIPE_MODELS.items():
        dest = mp_dir / url.split("/")[-1]
        status = "✅" if dest.exists() else "❌ MISSING"
        print(f"  {status}  mediapipe/{name}")
        if not dest.exists():
            all_ok = False
    if not all_ok:
        print("\nRun: python scripts/download_models.py  to download missing models.")
        sys.exit(1)
    print("\nAll models present.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MHBAP model weights")
    parser.add_argument("--check", action="store_true", help="Only verify, don't download")
    parser.add_argument("--model", choices=list(MODELS) + ["mediapipe"], default=None)
    args = parser.parse_args()

    if args.check:
        print("Checking model cache...")
        _check()
        return

    targets = [args.model] if args.model else list(MODELS) + ["mediapipe"]
    print(f"Downloading: {targets}")
    print(f"Cache dir  : {MODEL_DIR}\n")

    results = []
    for t in targets:
        if t == "mediapipe":
            results.append(_download_mediapipe())
        elif t in MODELS:
            results.append(_download_hf(t, MODELS[t]))

    if all(results):
        print("\n✅ All models ready. Update config.py MODEL_DIR if needed.")
    else:
        print("\n⚠️  Some downloads failed. Check network and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
