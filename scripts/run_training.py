"""Run TCMT training with real datasets. Execute from repo root."""
import sys, os
sys.path.insert(0, r"D:\MHBAP")
os.chdir(r"D:\MHBAP")

from ml.training.train_tcmt import train
results = train(epochs=35, fer_samples=6000, raf_samples=3000, wesad_samples=2000, verbose=True)
print("\n=== FINAL TEST METRICS ===")
import json
print(json.dumps(results, indent=2))
print("TRAINING_COMPLETE")
