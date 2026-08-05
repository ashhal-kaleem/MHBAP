"""
evaluate_real.py — Full evaluation on real held-out test sets.

Usage:
    python scripts/evaluate_real.py [--out PATH]

Loads trained TCMT weights, runs on real test split,
outputs full metrics JSON including confusion matrices, latency.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from ml.fusion.Tcmt import TCMT
from ml.training.RealDataset import make_real_dataset
from ml.evaluation.metrics import compute_all_metrics

WEIGHT_PATH  = Path(__file__).parent.parent / "ml" / "models" / "weights" / "tcmt_trained.pt"
METRICS_PATH = Path(__file__).parent.parent / "ml" / "models" / "weights" / "tcmt_eval_metrics.json"

EMOTION_NAMES = {0: "neutral", 1: "happy", 2: "sad", 3: "angry"}

def measure_latency(model: TCMT, n_runs: int = 200) -> dict:
    """Inference latency on single sample (ms)."""
    x = torch.randn(1, 1, 58)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(x.squeeze(1))
            times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times[10:])  # skip warmup
    return {
        "mean_ms":   float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "p99_ms":    float(np.percentile(arr, 99)),
    }


def main(out: Path = METRICS_PATH):
    print("[eval] Loading model...", flush=True)
    ck    = torch.load(str(WEIGHT_PATH), map_location="cpu", weights_only=False)
    model = TCMT()
    model.load_state_dict(ck["state_dict"])
    model.eval()

    print("[eval] Loading real test split...", flush=True)
    _, _, test = make_real_dataset(
        fer_samples=6000, raf_samples=3000, wesad_samples=2000, seed=42)
    print(f"[eval] Test samples: {len(test['X'])}", flush=True)

    X = torch.tensor(test["X"], dtype=torch.float32)
    with torch.no_grad():
        raw = model(X)

    # Predictions
    emo_logits = np.array(raw["emotion_logits"])
    st_pred    = np.array(raw["stress"]).squeeze() / 10.0
    en_pred    = np.array(raw["engagement"]).squeeze()
    at_pred    = np.array(raw["attention"]).squeeze()
    fa_pred    = np.array(raw["fatigue"]).squeeze()

    targets = {k: test[k] for k in ("emotion","stress","engagement","attention","fatigue")}
    preds   = dict(emotion=emo_logits, stress=st_pred,
                   engagement=en_pred, attention=at_pred, fatigue=fa_pred)

    metrics = compute_all_metrics(targets, preds)

    print("[eval] Measuring latency...", flush=True)
    metrics["latency"] = measure_latency(model)

    # Pretty print
    print("\n=== REAL-DATA TEST SET METRICS ===")
    for head, m in metrics.items():
        if head == "latency":
            print(f"\nLatency: mean={m['mean_ms']:.2f}ms  "
                  f"p95={m['p95_ms']:.2f}ms  p99={m['p99_ms']:.2f}ms")
            continue
        print(f"\n[{head}]")
        for k, v in m.items():
            if k in ("confusion_matrix", "per_class_f1"):
                continue
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if head == "emotion":
            print("  per_class_f1:", m.get("per_class_f1", {}))
            print("  confusion_matrix:")
            for row in m.get("confusion_matrix", []):
                print("   ", row)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\n[eval] Saved → {out}", flush=True)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=METRICS_PATH)
    args = p.parse_args()
    main(args.out)
