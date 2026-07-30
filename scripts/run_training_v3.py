"""
run_training_v3.py -- Retrain TCMT after class-collapse root-cause fix.

Fixes applied vs v2:
  1. ds.shuffle(seed, buffer_size=4000) added to load_fer2013() and load_rafdb()
     streaming iterators -- previously these took raw shard order which was
     class-sorted, producing 54% class-3 skew and causing head collapse on
     classes 1 and 2 (F1=0.0).
  2. Inverse-frequency class weights added to CrossEntropyLoss so any residual
     imbalance does not silence minority classes during training.

Run: python scripts/run_training_v3.py
Log: scripts/training_log_v3.txt
"""
import sys, os
sys.path.insert(0, r"D:\MHBAP")
os.environ["PYTHONIOENCODING"] = "utf-8"

log_path = r"D:\MHBAP\scripts\training_log_v3.txt"

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

import numpy as np

from ml.training.train_tcmt import train

print("=" * 60)
print("TCMT Retraining v3 -- Class-collapse fix")
print("Fixes vs v2:")
print("  1. shuffle(buffer_size=4000) on FER2013/RAF-DB streaming iterators")
print("  2. Inverse-frequency CrossEntropyLoss class weights")
print("=" * 60)

metrics = train(
    epochs=40,
    fer_samples=6000,
    raf_samples=3000,
    wesad_samples=2000,
    seed=42,
    verbose=True,
)

import json
print("\nFinal test metrics (v3):")
print(json.dumps(metrics, indent=2))
log_file.close()
print(f"\nLog written to: {log_path}", file=sys.__stdout__)
