import numpy as np
import torch
import json
from pathlib import Path
import sys

sys.path.append(r"d:\MHBAP")

from ml.training.real_dataset import make_real_dataset
from ml.fusion.tcmt import TCMT
from ml.evaluation.metrics import compute_all_metrics

def main():
    print("Loading real dataset splits...")
    train_split, val_split, test_split = make_real_dataset(
        fer_samples=6000,
        raf_samples=3000,
        wesad_samples=2000,
        seed=42
    )

    # Load saved model
    ckpt_path = r"d:\MHBAP\ml\models\weights\tcmt_trained.pt"
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    model = TCMT()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Recompute metrics on test set
    X_test = torch.tensor(test_split["X"], dtype=torch.float32)
    with torch.no_grad():
        out = model(X_test)
    
    em_pred = np.array(out["emotion_logits"])
    st_pred = np.array(out["stress"]).squeeze()
    en_pred = np.array(out["engagement"]).squeeze()
    at_pred = np.array(out["attention"]).squeeze()
    fa_pred = np.array(out["fatigue"]).squeeze()

    targets = {k: test_split[k] for k in ("emotion","stress","engagement","attention","fatigue")}
    preds   = {
        "emotion":    em_pred,
        "stress":     st_pred / 10.0,
        "engagement": en_pred,
        "attention":  at_pred,
        "fatigue":    fa_pred,
    }

    recomputed = compute_all_metrics(targets, preds)
    print("\n--- Recomputed Metrics ---")
    print(json.dumps(recomputed, indent=2))

    print("\n--- Saved Metrics ---")
    saved_path = r"d:\MHBAP\ml\models\weights\tcmt_eval_metrics.json"
    with open(saved_path, 'r') as f:
        saved = json.load(f)
    print(json.dumps(saved, indent=2))

    # Compare them
    for head in recomputed:
        print(f"\nComparing {head}:")
        for metric in recomputed[head]:
            v_rec = recomputed[head][metric]
            v_sav = saved[head].get(metric)
            if isinstance(v_rec, dict):
                print(f"  {metric}:")
                for k in v_rec:
                    diff = abs(v_rec[k] - v_sav[k])
                    print(f"    class {k}: recomputed={v_rec[k]:.6f}, saved={v_sav[k]:.6f}, diff={diff:.6e}")
            elif isinstance(v_rec, list):
                print(f"  {metric} lists match: {v_rec == v_sav}")
            else:
                diff = abs(v_rec - v_sav)
                print(f"  {metric}: recomputed={v_rec:.6f}, saved={v_sav:.6f}, diff={diff:.6e}")

if __name__ == "__main__":
    main()
