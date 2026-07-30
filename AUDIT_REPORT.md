# MHBAP TCMT Pipeline Audit Report

**Audited:** `ml/training/train_tcmt.py`, `ml/training/real_dataset.py`,
`ml/fusion/tcmt.py`, `ml/evaluation/metrics.py`, `ml/models/weights/tcmt_eval_metrics.json`

---

## CRITICAL ISSUE 1: `EMOTION_CLASSES = 8` but labels are 4-class

**File:** `ml/fusion/tcmt.py`, line with `EMOTION_CLASSES = 8`
**Impact:** Reported emotion accuracy **0.9993** and macro-F1 **0.9986** are inflated.

### Root Cause

```python
# tcmt.py
EMOTION_CLASSES = 8
self.head_emotion = nn.Linear(D_MODEL, 8)  # 8-class head
```

But the dataset maps FER2013 and RAF-DB to exactly 4 classes:

```python
FER_TO_MHBAP  = {0:3, 1:3, 2:2, 3:1, 4:2, 5:1, 6:0}  # outputs 0,1,2,3
RAF_TO_MHBAP  = {1:1, 2:2, 3:3, 4:1, 5:2, 6:3, 7:0}  # outputs 0,1,2,3
```

Training labels are always in `{0, 1, 2, 3}`. `CrossEntropyLoss` never activates
output neurons 4–7, so they remain near-zero. At inference, `argmax` over 8 outputs
where neurons 4–7 are near-zero will almost always pick from 0–3 — giving
artificially high accuracy. The model has not learned 8 classes; it has learned
4 classes with 4 dead neurons attached.

### Evidence

The saved confusion matrix only shows a 4×4 matrix (classes 0–3), confirming the
labels never included classes 4–7. Yet the head outputs 8 logits.

### Fix

Change `EMOTION_CLASSES = 8` → `EMOTION_CLASSES = 4` in `ml/fusion/tcmt.py`,
then retrain. The existing `tcmt_trained.pt` is incompatible (emotion head is 8×64)
and must be regenerated.

Script: `python scripts/fix_emotion_classes.py` patches the constant.

---

## CRITICAL ISSUE 2: Emotion accuracy inflated by label-biased feature construction

**File:** `ml/training/real_dataset.py`, `_build_feature()` and `_emotion_biases()`
**Impact:** Emotion accuracy of 99.9% does NOT reflect real-world generalisation.

### Root Cause

`_build_feature()` hard-codes emotion-class-specific values into the feature vector:

```python
def _emotion_biases(emotion: int) -> dict:
    if emotion == 1:  # happy
        return dict(gaze=[0.2,0.2,0.7,0.7,0.8], v_energy=0.6, ...)
    if emotion == 2:  # sad
        return dict(gaze=[0.6,0.6,0.5,0.5,0.4], v_energy=0.3, ...)
    if emotion == 3:  # angry
        return dict(gaze=[0.3,0.3,0.6,0.6,0.5], v_energy=0.8, hci_err=0.30)
```

These biases are deterministically injected **before** splitting. The TCMT model
learns to decode these injected patterns — not to infer emotion from independent
behavioural signals. The model is essentially inverting its own input encoder.

**A trivial linear classifier on `v_energy` and `gaze[4]` alone would achieve
similar accuracy**, because the emotion label is encoded in those values.

### Consequence for Test Metrics

This is not data leakage in the sense of train/test contamination, but the test
set suffers the same problem: labels are recoverable from features by design, so
test accuracy provides no evidence of generalisation to real behaviour.

---

## WARNING: Derived labels for engagement/attention/fatigue

**File:** `ml/training/real_dataset.py`, `_derive_labels()`

Engagement, attention, and fatigue labels are computed as weighted sums of feature
vector components:

```python
engagement = 0.35*fix_stab + 0.30*speak_rt + 0.25*ksr + 0.1*rng...
attention  = 0.5*(1-blink_a) + 0.35*fix_stab + 0.15*rng...
fatigue    = 0.4*pause_r + 0.3*dwell_t + 0.2*(1-energy) + 0.1*rng...
```

The model is trained to predict these labels FROM THE SAME feature vector `x`.
This means:
- R² for engagement (0.61), attention (0.17), fatigue (0.52) reflect the model's
  ability to learn a known algebraic function, not a real behavioural signal.
- No external ground truth exists for these labels.

**These metrics should be labelled "synthetic label regression" in any reporting.**

---

## FINDING: Stress scaling is internally consistent

The stress head outputs `sigmoid(...) * 10` (range [0, 10]).
Training uses `_forward_train` which applies only `sigmoid(...)` (range [0, 1]) for MSE loss — consistent with labels in [0, 1].
Evaluation divides the model output by 10.0 (`st_pred / 10.0`) before computing metrics.
**No bug.** Both paths correctly handle the scale difference.

---

## FINDING: Split disjointness is correct by construction

`make_real_dataset()` shuffles all records with `np.random.default_rng(seed)` and
performs a single contiguous slice: `[:n_train]`, `[n_train:n_train+n_val]`, `[n_train+n_val:]`.
- No sample appears in more than one split.
- No preprocessing statistics (mean, std) are computed across the full pool.
- All feature transforms are per-sample and deterministic.

**No statistical leakage from the split procedure.**

---

## FINDING: Test set used exactly once

Early stopping selects the best model via `val_f1`. The test set is evaluated
once at the end after restoring `best_state`. Hyperparameters are hardcoded
defaults — not tuned by iterating on test metrics.

**Test set integrity is maintained.**

---

## FINDING: RAF-DB HuggingFace split issue (non-critical)

The "test" split for `deanngkl/raf-db-7emotions` is unavailable on HuggingFace
(only "train" exists). `load_rafdb()` catches this and logs a warning. Only 1,500
RAF-DB samples (from "train") enter the pool. These are intermixed with FER2013
before the global shuffle, so some RAF-DB samples end up in our test set alongside
FER2013 samples. **This is acceptable** since make_real_dataset maintains its own
clean split, but the HuggingFace train/test split boundaries are discarded.

---

## FINDING: Metric recomputation matches saved values

Recomputing metrics from the saved checkpoint against freshly regenerated test
splits (seed=42) produces values that match `tcmt_eval_metrics.json` exactly.
The saved metrics are authentic — they were not manually edited.

---

## Summary Table

| Issue | Severity | Affects Metric | Fix |
|---|---|---|---|
| `EMOTION_CLASSES=8`, 4-class labels | **CRITICAL** | emotion accuracy, F1 | Set to 4, retrain |
| Emotion labels encoded in features | **CRITICAL** | emotion accuracy | Disclose; use real AUs |
| Derived labels (engagement/attention/fatigue) | WARNING | R², RMSE for these heads | Disclose as synthetic |
| Stress scaling | None (correct) | stress RMSE | No action |
| Split disjointness | None (correct) | all | No action |
| Test set used once | None (correct) | all | No action |
| RAF-DB split fallback | Minor | n/a | Document |

---

## Required Actions Before Any Publication/Presentation

1. **Run `python scripts/fix_emotion_classes.py`** to patch the head size.
2. **Retrain**: `python -m ml.training.train_tcmt`
3. **Add disclosure** in CHANGELOG/README: emotion labels are class-biased proxies,
   not real Action Unit measurements.
4. **Add disclosure**: engagement/attention/fatigue labels are synthetic (algebraic
   functions of input features) due to absence of public ground-truth datasets.
5. **Add ablation**: compare TCMT vs a label-from-feature oracle (linear regression
   directly on x) to quantify how much performance comes from the label bias.
6. **Report corrected metrics** after retraining with 4-class head.
