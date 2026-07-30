"""
run_training_v4.py -- Retrain TCMT after minority-class oversampling fix.

Fixes applied vs v3:
  1. WeightedRandomSampler added -- minority classes (esp. class 3, 198 samples)
     are oversampled so each class appears ~equally per epoch. This is the
     correct fix for genuine data scarcity, not just loss reweighting alone.
  2. Weighted CE loss retained as defence-in-depth.
  3. Epochs raised to 50 to give the model more time with balanced batches.

Run: python scripts/run_training_v4.py
Log: scripts/training_log_v4.txt
"""
import sys, os
sys.path.insert(0, r"D:\MHBAP")
os.environ["PYTHONIOENCODING"] = "utf-8"

log_path = r"D:\MHBAP\scripts\training_log_v4.txt"

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
print("TCMT Retraining v4 -- WeightedRandomSampler for minority classes")
print("Fixes vs v3:")
print("  1. WeightedRandomSampler (oversample class 3, ~198 samples)")
print("  2. Weighted CE loss retained")
print("  3. 50 epochs")
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

print("\nFinal test metrics (v4):")
print(json.dumps(metrics, indent=2))
log_file.close()
print(f"\nLog written to: {log_path}", file=sys.__stdout__)
