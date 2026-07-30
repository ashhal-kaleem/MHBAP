"""
train_tcmt.py — Training loop for TCMT using real public datasets.

Datasets used:
  - FER2013 (clip-benchmark/wds_fer2013)  → emotion (face images)
  - RAF-DB  (deanngkl/raf-db-7emotions)   → emotion (face images)
  - WESAD   (LouisSimon/wesad-parquet)    → stress  (physiological)

Usage (from repo root):
    python -m ml.training.train_tcmt [--epochs N] [--out PATH]

Saves checkpoint to ml/models/weights/tcmt_trained.pt
Saves metrics  to ml/models/weights/tcmt_eval_metrics.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.fusion.tcmt import TCMT, EMOTION_CLASSES
from ml.training.real_dataset import make_real_dataset
from ml.evaluation.metrics import compute_all_metrics

WEIGHT_PATH  = Path(__file__).parent.parent / "models" / "weights" / "tcmt_trained.pt"
METRICS_PATH = Path(__file__).parent.parent / "models" / "weights" / "tcmt_eval_metrics.json"


def _to_tensors(split: dict) -> TensorDataset:
    X   = torch.tensor(split["X"],          dtype=torch.float32)
    emo = torch.tensor(split["emotion"],     dtype=torch.long)
    st  = torch.tensor(split["stress"],      dtype=torch.float32).unsqueeze(-1)
    eng = torch.tensor(split["engagement"],  dtype=torch.float32).unsqueeze(-1)
    att = torch.tensor(split["attention"],   dtype=torch.float32).unsqueeze(-1)
    fat = torch.tensor(split["fatigue"],     dtype=torch.float32).unsqueeze(-1)
    return TensorDataset(X, emo, st, eng, att, fat)


def _forward_train(model: TCMT, X: torch.Tensor) -> dict:
    """Forward pass keeping tensors for backprop."""
    if X.dim() == 2:
        X = X.unsqueeze(1)
    B, T, F = X.shape
    tokens = torch.cat([model.mod_proj(X[:, t, :]) for t in range(T)], dim=1)
    cls    = model.cls_token.expand(B, -1, -1)
    enc    = model.encoder(torch.cat([cls, tokens], dim=1))
    h      = enc[:, 0, :]
    return {
        "emo_logits": model.head_emotion(h),
        "stress":     torch.sigmoid(model.head_stress(h)),
        "engagement": torch.sigmoid(model.head_engagement(h)),
        "attention":  torch.sigmoid(model.head_attention(h)),
        "fatigue":    torch.sigmoid(model.head_fatigue(h)),
    }


def _evaluate(model: TCMT, split: dict) -> Dict[str, Dict[str, float]]:
    model.eval()
    X = torch.tensor(split["X"], dtype=torch.float32)
    with torch.no_grad():
        out = model(X)
    em_pred = np.array(out["emotion_logits"])
    st_pred = np.array(out["stress"]).squeeze()
    en_pred = np.array(out["engagement"]).squeeze()
    at_pred = np.array(out["attention"]).squeeze()
    fa_pred = np.array(out["fatigue"]).squeeze()

    targets = {k: split[k] for k in ("emotion","stress","engagement","attention","fatigue")}
    preds   = {
        "emotion":    em_pred,
        "stress":     st_pred / 10.0,   # model scales 0-10, labels 0-1
        "engagement": en_pred,
        "attention":  at_pred,
        "fatigue":    fa_pred,
    }
    return compute_all_metrics(targets, preds)


def train(
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 42,
    verbose: bool = True,
    out: Path = WEIGHT_PATH,
    fer_samples: int = 6000,
    raf_samples: int = 3000,
    wesad_samples: int = 2000,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    train_split, val_split, test_split = make_real_dataset(
        fer_samples=fer_samples,
        raf_samples=raf_samples,
        wesad_samples=wesad_samples,
        seed=seed,
    )
    print(f"[TCMT] Data ready in {time.time()-t0:.1f}s", flush=True)

    train_ds = _to_tensors(train_split)

    # WeightedRandomSampler: oversample minority emotion classes so each
    # class appears equally often per epoch. Computed from training labels only.
    emo_labels  = train_split["emotion"]                          # (N,) int64
    class_count = np.bincount(emo_labels, minlength=EMOTION_CLASSES).astype(float)
    class_count = np.where(class_count == 0, 1.0, class_count)
    sample_wts  = (1.0 / class_count)[emo_labels]                # weight per sample
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_wts, dtype=torch.float64),
        num_samples=len(train_ds),
        replacement=True,
    )
    loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)

    model  = TCMT()
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Class-weighted CE: inverse-frequency weights so minority classes are not
    # drowned out by the majority class. Computed from training labels only.
    emo_counts = np.bincount(train_split["emotion"], minlength=EMOTION_CLASSES).astype(float)
    emo_counts = np.where(emo_counts == 0, 1.0, emo_counts)   # avoid divide-by-zero
    emo_weights = torch.tensor(1.0 / emo_counts, dtype=torch.float32)
    emo_weights = emo_weights / emo_weights.sum() * EMOTION_CLASSES  # scale so mean≈1
    ce  = nn.CrossEntropyLoss(weight=emo_weights)
    print(f"[TCMT] Emotion class counts (train): {emo_counts.astype(int).tolist()}", flush=True)
    print(f"[TCMT] CE class weights: {emo_weights.tolist()}", flush=True)

    mse = nn.MSELoss()

    best_f1, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for X, emo, st, eng, att, fat in loader:
            opt.zero_grad()
            o = _forward_train(model, X)
            loss = (ce(o["emo_logits"], emo)
                    + mse(o["stress"],     st)
                    + mse(o["engagement"], eng)
                    + mse(o["attention"],  att)
                    + mse(o["fatigue"],    fat))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * X.size(0)
        sched.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            vm = _evaluate(model, val_split)
            f1 = vm["emotion"]["macro_f1"]
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"loss={ep_loss/len(train_ds):.4f}  "
                  f"val_f1={f1:.3f}  "
                  f"val_stress_rmse={vm['stress']['rmse']:.3f}", flush=True)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate(model, test_split)
    if verbose:
        print("\n[TCMT] Test-set metrics (real held-out data):", flush=True)
        for head, m in test_metrics.items():
            print(f"  {head}: {m}", flush=True)

    save_path = Path(out)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "test_metrics": test_metrics,
                "datasets": ["FER2013","RAF-DB","WESAD"],
                "n_train": len(train_split["X"]),
                "n_val":   len(val_split["X"]),
                "n_test":  len(test_split["X"])}, save_path)
    # Annotate proxy labels so readers know which metrics are GT vs synthetic
    annotated = dict(test_metrics)
    annotated["_label_provenance"] = {
        "emotion":    "REAL - FER2013/RAF-DB dataset class annotations",
        "stress":     "MIXED - WESAD GT physio (stress samples) + AU/HCI proxy (emotion samples)",
        "engagement": "PROXY - derived from random noise dims; no public GT dataset",
        "attention":  "PROXY - derived from random noise dims; no public GT dataset",
        "fatigue":    "PROXY - derived from random noise dims; no public GT dataset",
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(annotated, indent=2))
    print(f"[TCMT] Saved weights: {save_path}", flush=True)
    print(f"[TCMT] Saved metrics: {METRICS_PATH}", flush=True)
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,  default=35)
    parser.add_argument("--fer_samples",  type=int,  default=6000)
    parser.add_argument("--raf_samples",  type=int,  default=3000)
    parser.add_argument("--wesad_samples",type=int,  default=2000)
    parser.add_argument("--out",          type=Path, default=WEIGHT_PATH)
    args = parser.parse_args()
    train(epochs=args.epochs, fer_samples=args.fer_samples,
          raf_samples=args.raf_samples, wesad_samples=args.wesad_samples,
          out=args.out)
