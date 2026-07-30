"""
train_tcmt.py — Training loop for TCMT.

Usage (from repo root):
    python -m ml.training.train_tcmt [--epochs N] [--samples N] [--out PATH]

Saves checkpoint to ml/models/weights/tcmt_trained.pt
Prints real metrics every epoch.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.fusion.tcmt import TCMT, EMOTION_CLASSES
from ml.training.dataset import make_dataset
from ml.evaluation.metrics import (
    emotion_metrics, regression_metrics, compute_all_metrics
)

WEIGHT_PATH = Path(__file__).parent.parent / "models" / "weights" / "tcmt_trained.pt"
METRICS_PATH = Path(__file__).parent.parent / "models" / "weights" / "tcmt_eval_metrics.json"


def _to_tensors(split: dict) -> TensorDataset:
    X = torch.tensor(split["X"], dtype=torch.float32)
    emo = torch.tensor(split["emotion"], dtype=torch.long)
    stress = torch.tensor(split["stress"], dtype=torch.float32).unsqueeze(-1)
    eng    = torch.tensor(split["engagement"], dtype=torch.float32).unsqueeze(-1)
    att    = torch.tensor(split["attention"], dtype=torch.float32).unsqueeze(-1)
    fat    = torch.tensor(split["fatigue"], dtype=torch.float32).unsqueeze(-1)
    return TensorDataset(X, emo, stress, eng, att, fat)


def _evaluate(model: TCMT, split: dict) -> Dict[str, Dict[str, float]]:
    model.eval()
    X = torch.tensor(split["X"], dtype=torch.float32)
    with torch.no_grad():
        out = model(X)
    # out values are numpy arrays (model returns numpy in inference mode)
    # Rebuild as torch for consistency
    em_pred = np.array(out["emotion_logits"])
    st_pred = np.array(out["stress"]).squeeze()
    en_pred = np.array(out["engagement"]).squeeze()
    at_pred = np.array(out["attention"]).squeeze()
    fa_pred = np.array(out["fatigue"]).squeeze()

    targets = {
        "emotion":    split["emotion"],
        "stress":     split["stress"],
        "engagement": split["engagement"],
        "attention":  split["attention"],
        "fatigue":    split["fatigue"],
    }
    preds = {
        "emotion":    em_pred,
        "stress":     st_pred / 10.0,     # model scales stress to 0-10, labels 0-1
        "engagement": en_pred,
        "attention":  at_pred,
        "fatigue":    fa_pred,
    }
    return compute_all_metrics(targets, preds)


def train(
    n_samples: int = 3000,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 42,
    verbose: bool = True,
    out: Path = WEIGHT_PATH,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"[TCMT] Generating {n_samples} samples …")
    train_split, val_split, test_split = make_dataset(n_samples=n_samples, seed=seed)

    train_ds = _to_tensors(train_split)
    val_ds   = _to_tensors(val_split)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = TCMT()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ce_loss  = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_f1 = -1.0
    best_state  = None

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for X, emo, stress, eng, att, fat in loader:
            opt.zero_grad()
            out = _forward_train(model, X)
            loss = (
                ce_loss(out["emotion_logits_t"], emo)
                + mse_loss(out["stress_t"], stress)
                + mse_loss(out["engagement_t"], eng)
                + mse_loss(out["attention_t"], att)
                + mse_loss(out["fatigue_t"], fat)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * X.size(0)
        sched.step()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            val_m = _evaluate(model, val_split)
            f1 = val_m["emotion"]["macro_f1"]
            st_rmse = val_m["stress"]["rmse"]
            print(f"  Epoch {epoch:3d}/{epochs}  loss={ep_loss/len(train_ds):.4f}"
                  f"  val_emotion_f1={f1:.3f}  val_stress_rmse={st_rmse:.3f}")
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_metrics = _evaluate(model, test_split)
    if verbose:
        print("\n[TCMT] Test-set metrics:")
        for head, m in test_metrics.items():
            print(f"  {head}: {m}")

    # Save — use explicit Path to avoid any local variable shadowing
    save_path = Path(out) if not isinstance(out, Path) else out
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "test_metrics": test_metrics}, save_path)
    metrics_path = Path(METRICS_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(test_metrics, indent=2))
    if verbose:
        print(f"[TCMT] Saved weights → {save_path}")
        print(f"[TCMT] Saved metrics → {metrics_path}")

    return test_metrics


def _forward_train(model: TCMT, X: torch.Tensor) -> dict:
    """Run model forward keeping tensors (not numpy) for backprop."""
    import torch as _t
    if X.dim() == 2:
        X = X.unsqueeze(1)
    B, T, F = X.shape
    tokens_list = []
    for t in range(T):
        tokens_list.append(model.mod_proj(X[:, t, :]))
    tokens = _t.cat(tokens_list, dim=1)
    cls = model.cls_token.expand(B, -1, -1)
    seq = _t.cat([cls, tokens], dim=1)
    enc = model.encoder(seq)
    cls_out = enc[:, 0, :]
    return {
        "emotion_logits_t": model.head_emotion(cls_out),
        "stress_t":         _t.sigmoid(model.head_stress(cls_out)),
        "engagement_t":     _t.sigmoid(model.head_engagement(cls_out)),
        "attention_t":      _t.sigmoid(model.head_attention(cls_out)),
        "fatigue_t":        _t.sigmoid(model.head_fatigue(cls_out)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=WEIGHT_PATH)
    args = parser.parse_args()
    train(n_samples=args.samples, epochs=args.epochs, out=args.out)
