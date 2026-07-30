"""
run_training_v5.py -- FINAL retrain. All fixes applied.

vs v4:
  - FER_TO_MHBAP: Disgust(1) remapped 3→2 (was double-counting angry class)
  - RAF_TO_MHBAP: Disgust(3) remapped 3→2 for consistency
  - WeightedRandomSampler retained (oversample minority classes)
  - Weighted CE loss retained
  - 50 epochs

This is the final authoritative training run. Weights and metrics from this
run are what go into AUDIT_REPORT.md and the benchmark table.

Run: python scripts/run_training_v5.py
Log: scripts/training_log_v5.txt
"""
import sys, os
sys.path.insert(0, r"D:\MHBAP")
os.environ["PYTHONIOENCODING"] = "utf-8"

log_path = r"D:\MHBAP\scripts\training_log_v5.txt"

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

from ml.training.train_tcmt import train

print("=" * 60)
print("TCMT Retraining v5 -- FINAL authoritative run")
print("All fixes applied:")
print("  1. EMOTION_CLASSES=4 (audit fix)")
print("  2. No circular bias injection (audit fix)")
print("  3. shuffle(buffer_size=4000) on streaming iterators (v3 fix)")
print("  4. FER Disgust(1) remapped 3->2; RAF Disgust(3) remapped 3->2")
print("  5. WeightedRandomSampler for minority class oversampling")
print("  6. Inverse-frequency CrossEntropyLoss weights")
print("  7. 50 epochs")
print("=" * 60)

import json
metrics = train(
    epochs=50,
    fer_samples=6000,
    raf_samples=3000,
    wesad_samples=2000,
    seed=42,
    verbose=True,
)

print("\nFinal test metrics (v5):")
print(json.dumps(metrics, indent=2))
log_file.close()
print(f"\nLog written to: {log_path}", file=sys.__stdout__)
