"""
sync_weights.py — Copy Colab-trained artifacts into the MHBAP repo.

Run from repo root after downloading from Colab:
    python colab/sync_weights.py

Looks for files in:
  1. Downloads folder (browser download default)
  2. Current directory
  3. Explicit --src PATH
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path

WEIGHTS_DIR  = Path(__file__).parent.parent / "ml" / "models" / "weights"
DOWNLOADS    = Path.home() / "Downloads"

FILES = {
    "tcmt_trained.pt":        WEIGHTS_DIR / "tcmt_trained.pt",
    "tcmt_eval_metrics.json": WEIGHTS_DIR / "tcmt_eval_metrics.json",
}


def find_file(name: str, src_dir: Path | None) -> Path | None:
    candidates = [
        src_dir / name if src_dir else None,
        DOWNLOADS / name,
        Path.cwd() / name,
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=None,
                        help="Directory containing downloaded Colab files")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    any_copied = False

    for fname, dst in FILES.items():
        src = find_file(fname, args.src)
        if src is None:
            print(f"  NOT FOUND: {fname}  (checked Downloads + cwd)")
            continue
        shutil.copy2(src, dst)
        sz = dst.stat().st_size / 1024**2
        print(f"  Copied {fname}  →  {dst}  ({sz:.2f} MB)")
        any_copied = True

    if not any_copied:
        print("\nNothing copied. Download files from Colab Cell 12 first.")
        return

    # Quick sanity check on checkpoint
    try:
        import torch
        ck = torch.load(WEIGHTS_DIR / "tcmt_trained.pt", map_location="cpu")
        sd = ck.get("state_dict", ck)
        keys = list(sd.keys())
        print(f"\nCheckpoint OK — {len(keys)} weight tensors")
        if "test_metrics" in ck:
            em = ck["test_metrics"].get("emotion", {})
            print(f"  Test emotion accuracy : {em.get('accuracy','?'):.4f}")
            print(f"  Test macro F1         : {em.get('macro_f1','?'):.4f}")
    except ImportError:
        print("PyTorch not installed in this env — skipping checkpoint check.")
    except Exception as e:
        print(f"Checkpoint check error: {e}")

    print("\nSync complete. Commit with:")
    print("  git add ml/models/weights/tcmt_trained.pt ml/models/weights/tcmt_eval_metrics.json")
    print('  git commit -m "feat(ml): update TCMT weights from Colab GPU training"')
    print("  git push origin feature/production-hardening")


if __name__ == "__main__":
    main()
