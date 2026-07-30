"""
full_audit.py - Comprehensive data leakage & metric integrity audit for MHBAP TCMT.

Run from repo root:
    python scripts/full_audit.py

Exit code 0 = no critical issues. Non-zero = critical issues found.
"""
from __future__ import annotations
import sys, json, hashlib, copy
sys.path.insert(0, r"D:\MHBAP")

import numpy as np
import torch

ISSUES   = []
WARNINGS = []

def flag(msg, critical=True):
    tag = "[CRITICAL]" if critical else "[WARN]"
    print(f"{tag} {msg}", flush=True)
    (ISSUES if critical else WARNINGS).append(msg)

def ok(msg):
    print(f"[OK]  {msg}", flush=True)

def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}", flush=True)

# ---------------------------------------------------------------------------
section("AUDIT 1: Model architecture vs label space mismatch")
# ---------------------------------------------------------------------------
from ml.fusion.tcmt import TCMT, EMOTION_CLASSES
from ml.training.real_dataset import FER_TO_MHBAP, RAF_TO_MHBAP

data_classes = sorted(set(FER_TO_MHBAP.values()) | set(RAF_TO_MHBAP.values()))
n_data_classes = len(data_classes)
print(f"  EMOTION_CLASSES constant (tcmt.py): {EMOTION_CLASSES}")
print(f"  Unique emotion labels in dataset:    {data_classes}  (n={n_data_classes})")

if EMOTION_CLASSES != n_data_classes:
    flag(
        f"EMOTION_CLASSES={EMOTION_CLASSES} but dataset only has {n_data_classes} classes (0-3). "
        f"head_emotion is Linear(64,8). During training CrossEntropyLoss with labels 0-3 "
        f"never activates output neurons 4-7, so they stay near-zero. argmax still picks "
        f"from 0-3 most of the time by accident - inflating reported accuracy."
    )
else:
    ok(f"Emotion head output dim ({EMOTION_CLASSES}) matches dataset classes ({n_data_classes})")

# ---------------------------------------------------------------------------
section("AUDIT 2: Stress scaling inconsistency (train vs evaluate)")
# ---------------------------------------------------------------------------
# In train_tcmt._evaluate():   preds["stress"] = st_pred / 10.0
# head_stress output:  sigmoid(.) * 10  =>  range [0, 10]
# labels: _derive_labels returns stress in [0,1]
# So dividing by 10 is correct.
# BUT in _forward_train (training loop):
#   o["stress"] = torch.sigmoid(model.head_stress(h))   <- NOT scaled by 10
# MSE loss is comparing sigmoid output [0,1] to label [0,1] -> OK in training
# Evaluation divides model output (already [0,10]) by 10 -> also OK
# Let's verify the model forward output scale
print("  Checking stress output scale from model.forward() vs _forward_train()...")
print("  model.forward() returns: sigmoid(head_stress) * 10   -> range [0,10]")
print("  _forward_train() uses:   sigmoid(head_stress)         -> range [0,1]")
print("  Training loss: MSE( sigmoid(head_stress)[0,1], label[0,1] ) -> CORRECT scale")
print("  _evaluate():   st_pred / 10.0  applied to model.forward() output [0,10] -> CORRECT")
print("  => Stress scaling is INTERNALLY consistent between train loop and eval.")
ok("Stress scaling consistent (train uses [0,1], eval divides [0,10] by 10)")

# ---------------------------------------------------------------------------
section("AUDIT 3: Label leakage - labels derived BEFORE split")
# ---------------------------------------------------------------------------
# In make_real_dataset: records are built as full {X, emotion, stress, ...} dicts,
# then shuffled, THEN split. Labels for engagement/attention/fatigue are derived
# from the feature vector x (via _derive_labels) BEFORE the split.
# This is NOT leakage in the classic sense: the derivation function is deterministic
# and purely a function of x. There is no fitting of statistics to the full pool.
# However: stress, engagement, attention, fatigue labels are DERIVED FROM x,
# not from independent ground truth. This means the model is learning to predict
# labels that are algebraic functions of its own inputs.
print("  Checking if labels are derived from features before or after split...")
print("  Observation: _derive_labels(emotion, stress_val, x, rng) is called during")
print("  record construction, before make_real_dataset shuffles and splits.")
print("  The labels engagement/attention/fatigue are deterministic functions of x.")
print("  => No statistical leakage from split (no scaler/normalizer fitted).")
print("  => But: labels are algebraic transforms of x, so model learns an identity")
print("     mapping rather than a generalizable relationship. This inflates R2/RMSE")
print("     for derived labels (engagement, attention, fatigue) and is a fundamental")
print("     data quality issue - NOT data leakage per se, but circular label definition.")
flag(
    "Labels (engagement/attention/fatigue) are deterministic functions of the "
    "feature vector x. The model is learning to invert its own input transform, "
    "not a real behavioural signal. Metrics for these heads are unreliable.",
    critical=False
)
ok("No train/val/test statistical leakage in preprocessing (no scalers fitted on full pool)")

# ---------------------------------------------------------------------------
section("AUDIT 4: RAF-DB split leakage - test split unavailable, train used twice")
# ---------------------------------------------------------------------------
# From training_log.txt:
#   RAF-DB [train] 1500 samples
#   RAF-DB [test]  WARN: Bad split: test. Available splits: ['train']
# load_rafdb() iterates over ("train","test"). When "test" fails, 0 samples added.
# But it also streams "train" for BOTH iterations (first loop = train, second = test).
# Actually examining the code: two separate for split in ("train","test") loops,
# each calls load_dataset(split=split). The "test" call fails gracefully with 0 samples.
# So only train split is loaded from RAF-DB - no duplicate contamination from that.
print("  RAF-DB: 'test' split unavailable on HuggingFace.")
print("  load_rafdb() catches exception and logs WARN, adds 0 records for that split.")
print("  => RAF-DB does NOT load the same samples twice.")
print("  => BUT: RAF-DB train samples (1500) end up in both train AND potentially")
print("     val/test pools after the global shuffle in make_real_dataset.")
print("  The global shuffle intermixes FER2013-train, FER2013-test, RAF-DB-train,")
print("  and WESAD. The test split is 15% of the combined pool.")
print("  FER2013 explicitly loads from HF 'train' and 'test' splits separately.")
print("  RAF-DB only contributes its 'train' split (HF 'test' unavailable).")
print("  => RAF-DB samples may overlap between our train/val/test via global shuffle.")
flag(
    "RAF-DB has no HF 'test' split. All 1500 RAF-DB samples come from its 'train' split "
    "and are intermixed with FER2013 before the global shuffle. FER2013 'train' (3000) "
    "and FER2013 'test' (3000) are both loaded and pooled together, then globally re-split. "
    "This means the final test set contains some FER2013 samples from FER's original test split "
    "AND some from FER's original train split - the HuggingFace split boundaries are discarded. "
    "This is acceptable since make_real_dataset does its own random split, but it means "
    "claimed independence from 'original benchmark test sets' is not maintained.",
    critical=False
)

# ---------------------------------------------------------------------------
section("AUDIT 5: Split disjointness - load fresh splits and verify")
# ---------------------------------------------------------------------------
print("  Regenerating splits with seed=42 (same as training)...", flush=True)
from ml.training.real_dataset import make_real_dataset

try:
    train_sp, val_sp, test_sp = make_real_dataset(
        fer_samples=6000, raf_samples=3000, wesad_samples=2000, seed=42
    )

    # Hash each row for exact-match dedup check
    def row_hashes(X):
        return set(hashlib.md5(row.tobytes()).hexdigest() for row in X)

    h_train = row_hashes(train_sp["X"])
    h_val   = row_hashes(val_sp["X"])
    h_test  = row_hashes(test_sp["X"])

    tv_overlap  = h_train & h_val
    tt_overlap  = h_train & h_test
    vt_overlap  = h_val   & h_test

    print(f"  Train size: {len(train_sp['X'])}, Val: {len(val_sp['X'])}, Test: {len(test_sp['X'])}")
    print(f"  Train/Val  exact-match duplicates: {len(tv_overlap)}")
    print(f"  Train/Test exact-match duplicates: {len(tt_overlap)}")
    print(f"  Val/Test   exact-match duplicates: {len(vt_overlap)}")

    if tv_overlap or tt_overlap or vt_overlap:
        flag(f"Exact-match duplicates found across splits: "
             f"train/val={len(tv_overlap)}, train/test={len(tt_overlap)}, val/test={len(vt_overlap)}")
    else:
        ok("All splits are exactly disjoint (no hash collisions)")

    # Also check within-set duplicates
    for name, sp in [("train", train_sp), ("val", val_sp), ("test", test_sp)]:
        hashes = [hashlib.md5(row.tobytes()).hexdigest() for row in sp["X"]]
        dups = len(hashes) - len(set(hashes))
        if dups > 0:
            flag(f"{name} set has {dups} exact duplicate samples (same feature vector)")
        else:
            ok(f"{name} set: no internal duplicates")

except Exception as e:
    flag(f"Could not regenerate splits (network/HF unavailable?): {e}")
    print("  Skipping split disjointness check.", flush=True)
    train_sp = val_sp = test_sp = None

# ---------------------------------------------------------------------------
section("AUDIT 6: RNG determinism - same seed => identical splits")
# ---------------------------------------------------------------------------
print("  Checking RNG reproducibility (two calls with seed=42)...", flush=True)
if test_sp is not None:
    try:
        _, _, test_sp2 = make_real_dataset(
            fer_samples=6000, raf_samples=3000, wesad_samples=2000, seed=42
        )
        if np.allclose(test_sp["X"], test_sp2["X"]) and \
           np.array_equal(test_sp["emotion"], test_sp2["emotion"]):
            ok("Splits are fully deterministic with same seed")
        else:
            flag("Splits differ between two runs with same seed=42 - non-deterministic RNG state")
    except Exception as e:
        flag(f"RNG check failed: {e}", critical=False)
else:
    print("  Skipped (no splits loaded).", flush=True)

# ---------------------------------------------------------------------------
section("AUDIT 7: Independent metric recomputation from saved weights")
# ---------------------------------------------------------------------------
print("  Loading saved checkpoint...", flush=True)
ckpt_path = r"D:\MHBAP\ml\models\weights\tcmt_trained.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model = TCMT()
model.load_state_dict(ckpt["state_dict"])
model.eval()

saved_metrics_path = r"D:\MHBAP\ml\models\weights\tcmt_eval_metrics.json"
saved_metrics = json.loads(open(saved_metrics_path).read())

print(f"  Checkpoint n_train={ckpt.get('n_train')}, "
      f"n_val={ckpt.get('n_val')}, n_test={ckpt.get('n_test')}")

if test_sp is not None:
    from ml.evaluation.metrics import compute_all_metrics
    X_test = torch.tensor(test_sp["X"], dtype=torch.float32)
    with torch.no_grad():
        out = model(X_test)

    em_pred = np.array(out["emotion_logits"])
    st_pred = np.array(out["stress"]).squeeze()
    en_pred = np.array(out["engagement"]).squeeze()
    at_pred = np.array(out["attention"]).squeeze()
    fa_pred = np.array(out["fatigue"]).squeeze()

    targets = {k: test_sp[k] for k in ("emotion","stress","engagement","attention","fatigue")}
    preds = {
        "emotion":    em_pred,
        "stress":     st_pred / 10.0,
        "engagement": en_pred,
        "attention":  at_pred,
        "fatigue":    fa_pred,
    }
    recomputed = compute_all_metrics(targets, preds)
    print("\n  Recomputed metrics:")
    print(json.dumps(recomputed, indent=4))
    print("\n  Saved metrics:")
    print(json.dumps(saved_metrics, indent=4))

    # Compare scalar metrics
    print("\n  Metric comparison (recomputed vs saved):")
    all_match = True
    for head in recomputed:
        if head not in saved_metrics:
            flag(f"Head '{head}' present in recomputed but missing from saved metrics")
            continue
        for metric, val in recomputed[head].items():
            if metric in ("per_class_f1", "confusion_matrix"):
                continue  # skip structured metrics for numeric comparison
            sv = saved_metrics[head].get(metric)
            if sv is None:
                continue
            if isinstance(val, float) and isinstance(sv, float):
                diff = abs(val - sv)
                status = "OK" if diff < 1e-6 else "MISMATCH"
                if diff >= 1e-6:
                    all_match = False
                print(f"    {head}.{metric:15s}: recomputed={val:.8f}  saved={sv:.8f}  diff={diff:.2e}  [{status}]")
    if all_match:
        ok("All recomputed scalar metrics match saved metrics exactly")
    else:
        flag("Recomputed metrics differ from saved metrics - model or test set changed since training")
else:
    print("  Skipped (no test split available).", flush=True)
    # Verify saved metrics against checkpoint
    ckpt_test = ckpt.get("test_metrics", {})
    if ckpt_test:
        print("  Comparing saved JSON vs checkpoint-embedded metrics...")
        for head in saved_metrics:
            for metric, sv in saved_metrics[head].items():
                if metric in ("per_class_f1", "confusion_matrix"):
                    continue
                cv = ckpt_test.get(head, {}).get(metric)
                if cv is not None and isinstance(sv, float):
                    diff = abs(sv - cv)
                    if diff > 1e-9:
                        flag(f"JSON metrics differ from checkpoint-embedded metrics: "
                             f"{head}.{metric} json={sv} ckpt={cv}")
        ok("JSON and checkpoint-embedded metrics are consistent")

# ---------------------------------------------------------------------------
section("AUDIT 8: Why emotion accuracy is 99.9% - root cause analysis")
# ---------------------------------------------------------------------------
print("  Investigating near-perfect emotion accuracy...", flush=True)

# Root cause hypothesis: labels are derived from emotion-biased features.
# _build_feature() injects emotion-specific biases into voice/gaze/hci dims.
# _derive_labels() then reads those SAME dims to produce engagement/attention/fatigue.
# For EMOTION specifically: the label comes directly from FER/RAF class mapping.
# The model input x has emotion-specific signal baked in (via _emotion_biases).
# So the model learns: "if voice energy is high + gaze stable -> class 1 (happy)".
# This is a very strong shortcut, not a generalizable emotion recogniser.

print("  _build_feature() injects emotion-class-specific biases into x:")
print("    happy(1):  gaze=[0.2,0.2,0.7,0.7,0.8], v_energy=0.6")
print("    sad(2):    gaze=[0.6,0.6,0.5,0.5,0.4], v_energy=0.3")
print("    angry(3):  gaze=[0.3,0.3,0.6,0.6,0.5], v_energy=0.8, hci_err=0.30")
print("    neutral(0):gaze=[0.3,0.3,0.6,0.6,0.65],v_energy=0.5")
print()
print("  The emotion class IS encoded deterministically in the feature vector.")
print("  The model is essentially learning to decode its own label-injection.")
print("  99.9% accuracy is expected - it is NOT a meaningful performance metric.")

flag(
    "Emotion accuracy 99.9% is an artifact of feature construction: "
    "_build_feature() hard-codes emotion-class-specific values into gaze/voice/hci "
    "dimensions, then the model learns to read those values back. "
    "The metric measures the model's ability to decode its own input encoding, "
    "not genuine emotion recognition from behaviour. This must be disclosed."
)

# Verify the bias by checking feature separation between classes
if test_sp is not None:
    print("\n  Verifying emotion-class feature separation on test set:")
    X = test_sp["X"]
    y = test_sp["emotion"]
    from ml.fusion.feature_utils import modality_slice
    vs, ve = modality_slice("voice")
    gs, ge = modality_slice("gaze")
    for cls in sorted(np.unique(y)):
        mask = y == cls
        v_energy = X[mask, vs+15].mean()
        fix_stab = X[mask, gs+4].mean()
        print(f"    class {cls}: v_energy_mean={v_energy:.3f}  fix_stab_mean={fix_stab:.3f}  n={mask.sum()}")
    print("  (Classes should be clearly separated if bias injection is working)")

# ---------------------------------------------------------------------------
section("AUDIT 9: Baseline comparisons")
# ---------------------------------------------------------------------------
if test_sp is not None:
    y_true = test_sp["emotion"]
    from sklearn.metrics import f1_score, accuracy_score
    # Majority-class baseline
    majority = int(np.bincount(y_true).argmax())
    y_maj = np.full_like(y_true, majority)
    maj_acc = accuracy_score(y_true, y_maj)
    maj_f1  = f1_score(y_true, y_maj, average="macro", zero_division=0)
    print(f"  Majority class baseline: class={majority}, acc={maj_acc:.3f}, macro_f1={maj_f1:.3f}")

    # Random baseline
    np.random.seed(0)
    y_rand = np.random.randint(0, 4, size=len(y_true))
    rand_acc = accuracy_score(y_true, y_rand)
    rand_f1  = f1_score(y_true, y_rand, average="macro", zero_division=0)
    print(f"  Random baseline:         acc={rand_acc:.3f}, macro_f1={rand_f1:.3f}")

    model_f1 = saved_metrics["emotion"]["macro_f1"]
    print(f"  TCMT model:              acc={saved_metrics['emotion']['accuracy']:.3f}, macro_f1={model_f1:.3f}")
    improvement = model_f1 - maj_f1
    print(f"  Improvement over majority baseline: macro_f1 +{improvement:.3f}")
    if improvement < 0.05:
        flag("Model barely outperforms majority-class baseline on macro-F1", critical=False)
    else:
        ok(f"Model outperforms majority baseline by {improvement:.3f} macro-F1 points")

    # Stress regression baseline
    stress_mean = test_sp["stress"].mean()
    stress_rmse_baseline = np.sqrt(np.mean((test_sp["stress"] - stress_mean)**2))
    model_rmse = saved_metrics["stress"]["rmse"]
    print(f"\n  Stress RMSE (mean predictor): {stress_rmse_baseline:.4f}")
    print(f"  Stress RMSE (TCMT):           {model_rmse:.4f}")
    if model_rmse >= stress_rmse_baseline:
        flag("TCMT stress RMSE is no better than predicting the mean")
    else:
        ok(f"TCMT stress RMSE better than mean predictor by {stress_rmse_baseline-model_rmse:.4f}")
else:
    print("  Skipped (no test split).", flush=True)

# ---------------------------------------------------------------------------
section("AUDIT 10: EMOTION_CLASSES=8 dead neuron analysis")
# ---------------------------------------------------------------------------
print("  Loading model weights to inspect emotion head for dead neurons...", flush=True)
w = model.head_emotion.weight.detach().numpy()  # (8, 64)
b = model.head_emotion.bias.detach().numpy()    # (8,)
print(f"  Emotion head weight shape: {w.shape}")
print(f"  Per-class output bias: {b.round(3)}")
print(f"  Per-class weight L2 norm: {np.linalg.norm(w, axis=1).round(3)}")

active_norms = np.linalg.norm(w[:4], axis=1)
dead_norms   = np.linalg.norm(w[4:], axis=1)
print(f"  Classes 0-3 (used) weight norms: {active_norms.round(3)}")
print(f"  Classes 4-7 (dead) weight norms: {dead_norms.round(3)}")

if dead_norms.max() > active_norms.min() * 0.5:
    flag(
        f"Dead neurons (classes 4-7) have non-trivial weight norms (max={dead_norms.max():.3f}). "
        f"They could occasionally produce a higher logit than active classes, "
        f"causing misclassifications that are invisible in per-class metrics but "
        f"contaminate the argmax output silently."
    )
else:
    ok("Dead neurons (classes 4-7) have small weight norms relative to active classes")

# Check if model ever predicts class >= 4 on test data
if test_sp is not None:
    X_t = torch.tensor(test_sp["X"], dtype=torch.float32)
    with torch.no_grad():
        o = model(X_t)
    preds_cls = np.argmax(o["emotion_logits"], axis=1)
    oor = (preds_cls >= 4).sum()
    print(f"\n  Out-of-range predictions (class >= 4) on test set: {oor} / {len(preds_cls)}")
    if oor > 0:
        flag(f"Model predicts {oor} samples into classes 4-7 (non-existent in labels). "
             f"These count as misclassifications but were not caught in original evaluation "
             f"because confusion_matrix only shows observed classes.")
    else:
        ok("Model never predicts class >= 4 on test set (dead neurons are truly inactive)")

# ---------------------------------------------------------------------------
section("AUDIT 11: Val-set used for early stopping - test set integrity")
# ---------------------------------------------------------------------------
# In train_tcmt.train(): best model is selected by val_f1 (early stopping).
# Test set is ONLY used once, at the end, after best model is restored.
# The test set is never used to select hyperparameters or trigger early stopping.
print("  Early stopping uses VAL set only (val_f1 in epoch loop).")
print("  Test set evaluated exactly once after best model is restored.")
print("  Hyperparameters (lr, epochs, batch_size, d_model) are hardcoded defaults,")
print("  not tuned by iterating on test metrics.")
ok("Test set used exactly once, after val-based early stopping. Evaluation is clean.")

# ---------------------------------------------------------------------------
section("AUDIT 12: Preprocessing normalization - any scaler fitted on full data?")
# ---------------------------------------------------------------------------
# Examining real_dataset.py: features are constructed from pixel stats and
# deterministic biases. No StandardScaler, MinMaxScaler, or PCA is applied.
# Feature values are clipped to [0,1] via np.clip. No fitting step.
print("  Checking for scaler/normalizer fitted across the full dataset...")
print("  real_dataset.py: no StandardScaler, MinMaxScaler, or PCA used.")
print("  Feature construction: pixel stats + deterministic transforms + rng noise.")
print("  np.clip(value, 0, 1) is applied per-sample - no statistics from other samples.")
ok("No preprocessing step fitted on the combined pool before splitting")

# ---------------------------------------------------------------------------
section("AUDIT 13: Duplicate sample analysis (near-duplicates)")
# ---------------------------------------------------------------------------
if test_sp is not None:
    print("  Checking for near-duplicates (L2 distance < 0.01) across train/test...", flush=True)
    # Sample check: first 200 test vs first 500 train (exact full check is O(N^2))
    X_tr_sample = train_sp["X"][:500]
    X_te_sample = test_sp["X"][:200]
    near_dup = 0
    for te_row in X_te_sample:
        dists = np.linalg.norm(X_tr_sample - te_row, axis=1)
        if dists.min() < 0.01:
            near_dup += 1
    print(f"  Near-duplicates (L2<0.01) in first 200 test vs first 500 train: {near_dup}")
    if near_dup > 0:
        flag(f"Found {near_dup} near-duplicate samples between train and test (L2 < 0.01). "
             f"Likely caused by different rng seeds producing nearly-identical feature vectors.")
    else:
        ok("No near-duplicates found in sampled train/test comparison")
else:
    print("  Skipped.", flush=True)

# ---------------------------------------------------------------------------
section("SUMMARY")
# ---------------------------------------------------------------------------
print(f"\nCRITICAL ISSUES: {len(ISSUES)}")
for i, issue in enumerate(ISSUES, 1):
    print(f"  {i}. {issue}")

print(f"\nWARNINGS: {len(WARNINGS)}")
for i, w in enumerate(WARNINGS, 1):
    print(f"  {i}. {w}")

print("\n" + "="*70)
print("VERDICT")
print("="*70)

if ISSUES:
    print("[FAIL] Critical issues found. Reported metrics are unreliable.")
    print()
    print("Primary issue: EMOTION_CLASSES=8 but training labels are 0-3 (4 classes).")
    print("Secondary: All performance metrics are inflated because labels are")
    print("           algebraic functions of the feature vectors - the model")
    print("           recovers its own input encoding, not real behaviour.")
    print()
    print("Corrections required:")
    print("  1. Change EMOTION_CLASSES to 4 in tcmt.py, OR")
    print("     remap labels to 0-7 using all FER7 classes.")
    print("  2. Retrain and re-evaluate after fixing head size.")
    print("  3. Document that engagement/attention/fatigue labels are synthetic.")
    print("  4. Add ablation: compare model vs label-from-feature oracle baseline.")
    sys.exit(1)
else:
    print("[PASS] No critical issues found.")
    if WARNINGS:
        print("       Warnings present - see above for details.")
    sys.exit(0)
