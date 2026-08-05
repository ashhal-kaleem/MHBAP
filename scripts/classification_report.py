"""
classification_report.py -- Full per-class precision/recall/F1 + confusion matrix.

Loads the saved checkpoint, re-runs inference on the held-out test set
(regenerated with the same seed=42), and prints:
  - sklearn classification_report
  - Confusion matrix (raw counts + normalized)
  - ROC-AUC per class (OvR)
  - Per-regression-head: RMSE, MAE, R²

Writes: scripts/classification_report.txt
Run:    python scripts/classification_report.py
"""
import sys, json
sys.path.insert(0, r"D:\MHBAP")
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support,
)
from scipy.special import softmax

from ml.training.RealDataset import make_real_dataset
from ml.fusion.Tcmt import TCMT
from ml.evaluation.metrics import compute_all_metrics

CLASS_NAMES = ["neutral", "happy/pos", "sad/fear", "angry/neg"]
CKPT  = r"D:\MHBAP\ml\models\weights\tcmt_trained.pt"
OUT   = r"D:\MHBAP\scripts\classification_report.txt"

print("Loading dataset (seed=42)...", flush=True)
_, _, test_s = make_real_dataset(seed=42)
X_te = test_s["X"]
y_te = test_s["emotion"]
print(f"Test set: {len(y_te)} samples")
print(f"Class dist: {np.bincount(y_te, minlength=4).tolist()}")

print("Loading checkpoint...", flush=True)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = TCMT()
model.load_state_dict(ck["state_dict"])
model.eval()

print("Running inference...", flush=True)
with torch.no_grad():
    out = model(torch.tensor(X_te))

logits  = np.array(out["emotion_logits"])
probs   = softmax(logits, axis=1)
pred_cls = np.argmax(logits, axis=1)
st_pred  = np.array(out["stress"]).squeeze() / 10.0
en_pred  = np.array(out["engagement"]).squeeze()
at_pred  = np.array(out["attention"]).squeeze()
fa_pred  = np.array(out["fatigue"]).squeeze()

lines = []
lines.append("=" * 70)
lines.append("MHBAP TCMT -- Full Classification & Regression Report (v5 weights)")
lines.append("=" * 70)
lines.append(f"\nTest set: {len(y_te)} samples (seed=42, held-out, never used in training)")
lines.append(f"Class distribution: {np.bincount(y_te, minlength=4).tolist()}")
lines.append(f"  0=neutral  1=happy/pos  2=sad/fear  3=angry/neg\n")

# sklearn classification report
lines.append("── Per-class Precision / Recall / F1 ──")
cr = classification_report(y_te, pred_cls, target_names=CLASS_NAMES,
                            labels=[0,1,2,3], zero_division=0)
lines.append(cr)

# Confusion matrix
lines.append("── Confusion Matrix (rows=true, cols=predicted) ──")
cm = confusion_matrix(y_te, pred_cls, labels=[0,1,2,3])
hdr = f"{'':12s}" + "".join(f"{n:>12s}" for n in CLASS_NAMES)
lines.append(hdr)
for i, row in enumerate(cm):
    lines.append(f"{CLASS_NAMES[i]:12s}" + "".join(f"{v:>12d}" for v in row))

lines.append("\n── Confusion Matrix (normalized by true class) ──")
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
lines.append(hdr)
for i, row in enumerate(cm_norm):
    lines.append(f"{CLASS_NAMES[i]:12s}" + "".join(f"{v:>12.3f}" for v in row))

# ROC-AUC per class (OvR)
lines.append("\n── ROC-AUC (One-vs-Rest per class) ──")
try:
    for c, name in enumerate(CLASS_NAMES):
        y_bin = (y_te == c).astype(int)
        auc = roc_auc_score(y_bin, probs[:, c])
        lines.append(f"  {name:14s}: {auc:.4f}")
    macro_auc = roc_auc_score(y_te, probs, multi_class="ovr", average="macro",
                               labels=[0,1,2,3])
    lines.append(f"  {'macro OvR':14s}: {macro_auc:.4f}")
except Exception as e:
    lines.append(f"  ROC-AUC error: {e}")

# Regression heads
lines.append("\n── Regression Heads ──")
from ml.evaluation.metrics import regression_metrics
for name, pred, true_key in [
    ("stress     (REAL GT)",     st_pred, "stress"),
    ("engagement (PROXY label)", en_pred, "engagement"),
    ("attention  (PROXY label)", at_pred, "attention"),
    ("fatigue    (PROXY label)", fa_pred, "fatigue"),
]:
    m = regression_metrics(test_s[true_key], pred)
    lines.append(f"  {name}: RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")

report = "\n".join(lines)
open(OUT, "w", encoding="utf-8").write(report)
print(report)
print(f"\nReport written to: {OUT}")
