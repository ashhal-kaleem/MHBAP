"""
audit_pipeline.py — Systematic data leakage and metric integrity audit.

Checks:
  1. Label construction: are labels derived from features BEFORE or AFTER split?
  2. RNG state leakage: does the same rng seed produce same records each time?
  3. Split disjointness: are train/val/test truly non-overlapping?
  4. Duplicate detection: exact-match on feature vectors
  5. The stress scaling bug in _evaluate()
  6. The EMOTION_CLASSES mismatch (model outputs 8 classes, data has 4)
  7. Independent metric recomputation from saved weights + fresh test set
  8. Baseline comparison (majority class, random)
"""
import sys, json, copy
sys.path.insert(0, r"D:\MHBAP")

import numpy as np
import torch

ISSUES = []
WARNINGS = []

def flag(msg, critical=True):
    entry = f"{'[CRITICAL]' if critical else '[WARN]'} {msg}"
    print(entry, flush=True)
    if critical:
        ISSUES.append(msg)
    else:
        WARNINGS.append(msg)

def ok(msg):
    print(f"[OK] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("AUDIT 1: EMOTION_CLASSES mismatch")
print("="*70)

from ml.fusion.tcmt import TCMT, EMOTION_CLASSES
print(f"  EMOTION_CLASSES in tcmt.py = {EMOTION_CLASSES}  (model head outputs this many logits)")
print(f"  MHBAP 4-class mapping used in dataset construction")
# The model head is Linear(D_MODEL, 8) but data only has labels 0-3
m = TCMT()
state = torch.load(r"D:\MHBAP\ml\models\weights\tcmt_trained.pt", map_location="cpu")
m.load_state_dict(state["state_dict"])
m.eval()

dummy = np.random.rand(1, 58).astype(np.float32)
out = m(dummy)
logit_shape = out["emotion_logits"].shape
print(f"  Model output shape: emotion_logits={logit_shape}")
if logit_shape[-1] != 4:
    flag(f"Model outputs {logit_shape[-1]} emotion logits but training labels are 0-3 (4 classes). "
         f"argmax over 8 logits where only classes 0-3 appear in labels — "
         f"classes 4-7 are dead neurons, effectively random. Accuracy looks high "
         f"only because argmax over dead neurons still tends to land in 0-3.")
else:
    ok("Emotion head output dim matches 4-class labels")
