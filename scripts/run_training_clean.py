"""
run_training_clean.py -- Retrain TCMT from scratch after audit fixes.
Run: python scripts/run_training_clean.py
Logs to: scripts/training_log_v2.txt
"""
import sys, io, os
sys.path.insert(0, r"D:\MHBAP")
os.environ["PYTHONIOENCODING"] = "utf-8"

# Redirect stdout to both console and log file
log_path = r"D:\MHBAP\scripts\training_log_v2.txt"

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

from ml.training.train_tcmt import train

print("=" * 60)
print("TCMT Retraining -- Post-Audit Clean Run")
print("Fixes applied:")
print("  - EMOTION_CLASSES=4 (was 8)")
print("  - No circular bias injection in feature construction")
print("  - Label provenance annotated in output JSON")
print("=" * 60)

metrics = train(
    epochs=35,
    fer_samples=6000,
    raf_samples=3000,
    wesad_samples=2000,
    seed=42,
    verbose=True,
)

print("\nFinal test metrics:")
import json
print(json.dumps(metrics, indent=2))
log_file.close()
print(f"\nLog written to: {log_path}", file=sys.__stdout__)
