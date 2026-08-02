"""save_trained_tcmt.py — train TCMT and persist weights.
Run from repo root: python scripts/save_trained_tcmt.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from ml.fusion.tcmt import TCMT
from ml.training.dataset import make_dataset
from ml.evaluation.metrics import emotion_metrics, regression_metrics

WEIGHT_PATH  = ROOT / "ml" / "models" / "weights" / "tcmt_trained.pt"
METRICS_PATH = ROOT / "ml" / "models" / "weights" / "tcmt_eval_metrics.json"


def _tensors(split):
    X  = torch.tensor(split["X"],         dtype=torch.float32)
    em = torch.tensor(split["emotion"],    dtype=torch.long)
    st = torch.tensor(split["stress"],     dtype=torch.float32).unsqueeze(-1)
    en = torch.tensor(split["engagement"], dtype=torch.float32).unsqueeze(-1)
    at = torch.tensor(split["attention"],  dtype=torch.float32).unsqueeze(-1)
    fa = torch.tensor(split["fatigue"],    dtype=torch.float32).unsqueeze(-1)
    return TensorDataset(X, em, st, en, at, fa)


def _fwd(model, X):
    if X.dim() == 2:
        X = X.unsqueeze(1)
    B, T, _ = X.shape
    toks = torch.cat([model.mod_proj(X[:, t, :]) for t in range(T)], dim=1)
    cls  = model.cls_token.expand(B, -1, -1)
    enc  = model.encoder(torch.cat([cls, toks], dim=1))
    h    = enc[:, 0, :]
    return {
        "emo": model.head_emotion(h),
        "st":  torch.sigmoid(model.head_stress(h)),
        "en":  torch.sigmoid(model.head_engagement(h)),
        "at":  torch.sigmoid(model.head_attention(h)),
        "fa":  torch.sigmoid(model.head_fatigue(h)),
    }


def run():
    torch.manual_seed(42)
    np.random.seed(42)
    print("[TCMT] Building dataset …")
    tr, va, te = make_dataset(n_samples=3000, seed=42)
    loader = DataLoader(_tensors(tr), batch_size=64, shuffle=True)
    model  = TCMT()
    opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    ce     = nn.CrossEntropyLoss()
    mse    = nn.MSELoss()
    best_f1, best_sd = -1.0, None

    for ep in range(1, 31):
        model.train()
        ep_loss = 0.0
        for X, em, st, en, at, fa in loader:
            opt.zero_grad()
            o = _fwd(model, X)
            loss = (ce(o["emo"], em) + mse(o["st"], st)
                    + mse(o["en"], en) + mse(o["at"], at) + mse(o["fa"], fa))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * X.size(0)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            model.eval()
            Xv = torch.tensor(va["X"], dtype=torch.float32)
            with torch.no_grad():
                ov = _fwd(model, Xv)
            f1 = f1_score(va["emotion"], ov["emo"].argmax(1).numpy(),
                          average="macro", zero_division=0)
            print(f"  ep {ep:3d}  loss={ep_loss/len(tr['X']):.4f}  val_f1={f1:.3f}")
            if f1 > best_f1:
                best_f1 = f1
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    if best_sd:
        model.load_state_dict(best_sd)

    # Test evaluation
    model.eval()
    Xt = torch.tensor(te["X"], dtype=torch.float32)
    with torch.no_grad():
        ot = _fwd(model, Xt)

    test_m = {
        "emotion":    emotion_metrics(te["emotion"], ot["emo"].numpy()),
        "stress":     regression_metrics(te["stress"],     ot["st"].squeeze().numpy()),
        "engagement": regression_metrics(te["engagement"], ot["en"].squeeze().numpy()),
        "attention":  regression_metrics(te["attention"],  ot["at"].squeeze().numpy()),
        "fatigue":    regression_metrics(te["fatigue"],    ot["fa"].squeeze().numpy()),
    }
    print("\n[TCMT] Test metrics:")
    for head, m in test_m.items():
        print(f"  {head}: {m}")

    wp = Path(WEIGHT_PATH)
    wp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "test_metrics": test_m}, wp)
    Path(METRICS_PATH).write_text(json.dumps(test_m, indent=2))
    print("[TCMT] Weights  -> " + str(wp))
    print("[TCMT] Metrics  -> " + str(METRICS_PATH))
    return test_m


if __name__ == "__main__":
    run()
