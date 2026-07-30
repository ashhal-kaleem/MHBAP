"""
fix_emotion_classes.py - Patch EMOTION_CLASSES from 8 to 4 in tcmt.py.

The TCMT model defines head_emotion = Linear(D_MODEL, 8) but training data
only ever has labels 0-3 (4-class mapping from FER2013/RAF-DB).
This creates 4 dead output neurons (classes 4-7) that are never trained.

This script:
  1. Confirms the bug.
  2. Patches EMOTION_CLASSES = 4 in tcmt.py.
  3. Notes that the saved model weights are incompatible (must retrain).

Run: python scripts/fix_emotion_classes.py
"""
import sys
sys.path.insert(0, r"D:\MHBAP")

from pathlib import Path

tcmt_path = Path(r"D:\MHBAP\ml\fusion\tcmt.py")
src = tcmt_path.read_text(encoding="utf-8")

OLD = "EMOTION_CLASSES = 8"
NEW = "EMOTION_CLASSES = 4  # 4-class: 0=neutral, 1=happy, 2=sad, 3=angry"

if OLD not in src:
    print(f"[WARN] '{OLD}' not found in tcmt.py - may already be patched or changed.")
    sys.exit(0)

src_fixed = src.replace(OLD, NEW, 1)
tcmt_path.write_text(src_fixed, encoding="utf-8")
print(f"[FIXED] Patched {tcmt_path}")
print(f"  Before: {OLD}")
print(f"  After:  {NEW}")
print()
print("[NOTE] The saved model checkpoint (tcmt_trained.pt) has emotion head shape (8,64).")
print("       It is INCOMPATIBLE with EMOTION_CLASSES=4. You must retrain:")
print("         python -m ml.training.train_tcmt")
print("       The existing tcmt_eval_metrics.json reflects the buggy 8-class head.")
