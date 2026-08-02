"""
cache_eval_test_split.py — Generate and cache the real held-out test split.

Run this ONCE after training to create the npz file that
benchmark.py and ablation.py load from:

    python scripts/cache_eval_test_split.py

What it does
------------
1. Calls make_real_dataset(seed=42) with the same parameters used during
   training — same FER2013 / RAF-DB / WESAD sample counts, same seed, so
   the deterministic shuffle produces the identical train/val/test split.
2. Saves only the test portion to
   ml/datasets/processed/eval_test_split.npz
   (compressed numpy archive, ~few hundred KB).

After this script completes successfully, the evaluation endpoints
GET /api/v1/evaluation/benchmark  and  GET /api/v1/evaluation/ablation
will load real test data instead of raising RuntimeError.

Note on data availability
-------------------------
FER2013 and WESAD are fetched via HuggingFace datasets streaming.
RAF-DB download may fail intermittently (observed in training logs) —
the script warns but continues; any records successfully fetched are
included.  To reproduce the exact training split, run under the same
network conditions as the original training session.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

SAVE_PATH = Path("ml/datasets/processed/eval_test_split.npz")
CHECKPOINT_PATH = Path("ml/models/weights/tcmt_trained.pt")
METRICS_PATH = Path("ml/models/weights/tcmt_eval_metrics.json")

# Match training parameters exactly
FER_SAMPLES   = 6000
RAF_SAMPLES   = 3000
WESAD_SAMPLES = 2000
SEED          = 42


def main() -> None:
    print("=" * 60)
    print("MHBAP — Real Test Split Cache Generator")
    print("=" * 60)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load dataset ────────────────────────────────────────────────
    print(f"\n[1/3] Loading real datasets (seed={SEED}) ...")
    print(f"      FER2013={FER_SAMPLES}  RAF-DB={RAF_SAMPLES}  WESAD={WESAD_SAMPLES}")
    from ml.training.real_dataset import make_real_dataset

    train_s, val_s, test_s = make_real_dataset(
        fer_samples=FER_SAMPLES,
        raf_samples=RAF_SAMPLES,
        wesad_samples=WESAD_SAMPLES,
        seed=SEED,
    )

    n_train = len(train_s["X"])
    n_val   = len(val_s["X"])
    n_test  = len(test_s["X"])
    print(f"      Split: train={n_train}  val={n_val}  test={n_test}")

    em_dist = np.bincount(test_s["emotion"], minlength=4).tolist()
    print(f"      Test emotion class distribution: {em_dist}")

    # ── Step 2: Save test split ─────────────────────────────────────────────
    print(f"\n[2/3] Saving test split to {SAVE_PATH} ...")
    np.savez_compressed(
        str(SAVE_PATH),
        X=test_s["X"],
        emotion=test_s["emotion"],
        stress=test_s["stress"],
        engagement=test_s["engagement"],
        attention=test_s["attention"],
        fatigue=test_s["fatigue"],
    )
    size_kb = SAVE_PATH.stat().st_size / 1024
    print(f"      Saved ({size_kb:.1f} KB).")

    # ── Step 3: Cross-check against saved checkpoint metrics ────────────────
    print(f"\n[3/3] Cross-checking against saved checkpoint ...")
    if not CHECKPOINT_PATH.exists():
        print(f"      WARN: checkpoint not found at {CHECKPOINT_PATH}; skipping cross-check.")
        print("\nDone.  Run training first to enable metric verification.")
        return

    import torch
    from ml.fusion.tcmt import TCMT
    from ml.evaluation.metrics import compute_all_metrics

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = (
        ckpt["state_dict"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt
        else ckpt
    )
    model = TCMT()
    model.load_state_dict(state_dict)
    model.eval()

    X_te = torch.tensor(test_s["X"], dtype=torch.float32)
    with torch.no_grad():
        out = model(X_te)

    em_pred = np.array(out["emotion_logits"])
    st_pred = np.array(out["stress"]).squeeze() / 10.0
    en_pred = np.array(out["engagement"]).squeeze()
    at_pred = np.array(out["attention"]).squeeze()
    fa_pred = np.array(out["fatigue"]).squeeze()

    recomputed = compute_all_metrics(
        {k: test_s[k] for k in ("emotion", "stress", "engagement", "attention", "fatigue")},
        {"emotion": em_pred, "stress": st_pred,
         "engagement": en_pred, "attention": at_pred, "fatigue": fa_pred},
    )

    print("\n      Recomputed test metrics (from cached split + checkpoint):")
    em = recomputed["emotion"]
    print(f"        emotion  accuracy  : {em['accuracy']:.4f}")
    print(f"        emotion  macro_f1  : {em['macro_f1']:.4f}")
    print(f"        emotion  per_class : {em['per_class_f1']}")
    print(f"        stress   rmse      : {recomputed['stress']['rmse']:.4f}")
    print(f"        stress   r2        : {recomputed['stress']['r2']:.4f}")

    if METRICS_PATH.exists():
        saved = json.loads(METRICS_PATH.read_text())
        saved_f1    = saved["emotion"]["macro_f1"]
        recomp_f1   = em["macro_f1"]
        delta       = abs(saved_f1 - recomp_f1)
        match_ok    = delta < 1e-6

        print(f"\n      Saved macro_f1  : {saved_f1:.6f}")
        print(f"      Recomp macro_f1 : {recomp_f1:.6f}")
        print(f"      Delta           : {delta:.2e}")
        print(f"      Match           : {'PASS ✓' if match_ok else 'FAIL ✗  (split mismatch — different data was downloaded)'}")

        if not match_ok:
            print(
                "\n      NOTE: A mismatch usually means the HuggingFace streaming iterator\n"
                "      returned different shards than during training (e.g. RAF-DB partially\n"
                "      failed).  The cached split is still valid for consistent evaluation;\n"
                "      however, the benchmark metrics will differ from the checkpoint's\n"
                "      embedded test_metrics.  Re-train or document the discrepancy."
            )
    else:
        print(f"      INFO: {METRICS_PATH} not found; skipped saved-vs-recomputed comparison.")

    print("\n" + "=" * 60)
    print("Cache generation complete.")
    print(f"  File: {SAVE_PATH.resolve()}")
    print(f"  Size: {SAVE_PATH.stat().st_size / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
