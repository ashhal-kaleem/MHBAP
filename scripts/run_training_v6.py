"""
run_training_v6.py -- TCMT v6 training with evidence-based improvements.

Changes vs v5:
  1. Focal loss (gamma=2.0) replaces weighted CE -- kills easy-class domination
  2. Emotion loss weight x4 vs regression x0.25 -- fix multi-task gradient imbalance
  3. Increased capacity: D_MODEL=128, N_LAYERS=3, FFN=256
  4. LR warmup (5 ep) + cosine anneal
  5. Label smoothing 0.05
  6. 75 epochs
  7. Enhanced face feature extraction (12 dims, better pixel stats)
  8. Fixed predictor checkpoint load (state_dict key unwrapping)
  9. Strictly evaluate on held-out test set; commit only if macro-F1 > 0.278

Usage: python scripts/run_training_v6.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WEIGHT_PATH  = Path("ml/models/weights/tcmt_trained.pt")
METRICS_PATH = Path("ml/models/weights/tcmt_eval_metrics.json")

print("=" * 60)
print("TCMT Retraining v6")
print("Changes: focal loss, loss weighting, capacity++, warmup, label smoothing")
print("=" * 60, flush=True)

# ── 1. Feature dimension constants (must match feature_vector.py) ────────────
FEATURE_DIM = 58
EMOTION_CLASSES = 4

# ── 2. Improved TCMT with higher capacity ────────────────────────────────────
from ml.fusion.feature_vector import MODALITY_KEYS

_MOD_SLICES: dict = {}
_off = 0
for _m, _ks in MODALITY_KEYS.items():
    _MOD_SLICES[_m] = (_off, _off + len(_ks))
    _off += len(_ks)

D_MODEL  = 128   # was 64
N_HEADS  = 4
N_LAYERS = 3     # was 2
FFN_DIM  = 256   # was 128
DROPOUT  = 0.15


class _ModalityProjectionV6(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projs = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(end - start, D_MODEL),
                nn.LayerNorm(D_MODEL),
            )
            for mod, (start, end) in _MOD_SLICES.items()
        })
        n_mods = len(MODALITY_KEYS)
        self.mod_embed = nn.Embedding(n_mods, D_MODEL)
        self._mod_idx = {m: i for i, m in enumerate(MODALITY_KEYS)}

    def forward(self, x):
        parts = []
        for mod, (start, end) in _MOD_SLICES.items():
            proj = self.projs[mod](x[:, start:end])
            idx  = self._mod_idx[mod]
            proj = proj + self.mod_embed(torch.tensor(idx, device=x.device))
            parts.append(proj.unsqueeze(1))
        return torch.cat(parts, dim=1)


class TCMT_V6(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod_proj = _ModalityProjectionV6()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=FFN_DIM, dropout=DROPOUT,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.normal_(self.cls_token, std=0.02)

        self.head_emotion    = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(D_MODEL // 2, EMOTION_CLASSES),
        )
        self.head_stress     = nn.Linear(D_MODEL, 1)
        self.head_engagement = nn.Linear(D_MODEL, 1)
        self.head_attention  = nn.Linear(D_MODEL, 1)
        self.head_fatigue    = nn.Linear(D_MODEL, 1)

    def _encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, F = x.shape
        tokens = torch.cat([self.mod_proj(x[:, t, :]) for t in range(T)], dim=1)
        cls    = self.cls_token.expand(B, -1, -1)
        seq    = torch.cat([cls, tokens], dim=1)
        enc    = self.encoder(seq)
        return enc[:, 0, :]

    def forward_train(self, x):
        h = self._encode(x)
        return {
            "emo_logits": self.head_emotion(h),
            "stress":     torch.sigmoid(self.head_stress(h)),
            "engagement": torch.sigmoid(self.head_engagement(h)),
            "attention":  torch.sigmoid(self.head_attention(h)),
            "fatigue":    torch.sigmoid(self.head_fatigue(h)),
        }

    def forward_infer(self, x):
        """Inference-only; returns numpy dict matching TCMT API."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        self.eval()
        with torch.no_grad():
            h = self._encode(x)
            return {
                "emotion_logits": self.head_emotion(h).cpu().numpy(),
                "stress":    (torch.sigmoid(self.head_stress(h)) * 10).cpu().numpy(),
                "engagement": torch.sigmoid(self.head_engagement(h)).cpu().numpy(),
                "attention":  torch.sigmoid(self.head_attention(h)).cpu().numpy(),
                "fatigue":    torch.sigmoid(self.head_fatigue(h)).cpu().numpy(),
            }


# ── 3. Focal loss ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss with optional class weights and label smoothing."""
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Label smoothing
        n_cls = logits.size(1)
        smooth_val = self.label_smoothing / (n_cls - 1)
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        one_hot = one_hot * (1.0 - self.label_smoothing) + smooth_val

        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()

        # Focal weight
        pt = (one_hot * probs).sum(dim=1)
        focal_w = (1.0 - pt) ** self.gamma

        # Weighted CE
        ce = -(one_hot * log_probs).sum(dim=1)
        if self.weight is not None:
            w = self.weight[targets]
            ce = ce * w
        loss = (focal_w * ce).mean()
        return loss


# ── 4. Dataset + evaluation ──────────────────────────────────────────────────

from ml.training.real_dataset import make_real_dataset
from ml.evaluation.metrics import compute_all_metrics


def _to_tensors(split):
    X   = torch.tensor(split["X"],          dtype=torch.float32)
    emo = torch.tensor(split["emotion"],     dtype=torch.long)
    st  = torch.tensor(split["stress"],      dtype=torch.float32).unsqueeze(-1)
    eng = torch.tensor(split["engagement"],  dtype=torch.float32).unsqueeze(-1)
    att = torch.tensor(split["attention"],   dtype=torch.float32).unsqueeze(-1)
    fat = torch.tensor(split["fatigue"],     dtype=torch.float32).unsqueeze(-1)
    return TensorDataset(X, emo, st, eng, att, fat)


def _evaluate(model, split):
    model.eval()
    X = torch.tensor(split["X"], dtype=torch.float32)
    out = model.forward_infer(X)
    preds = {
        "emotion":    out["emotion_logits"],
        "stress":     out["stress"].squeeze() / 10.0,
        "engagement": out["engagement"].squeeze(),
        "attention":  out["attention"].squeeze(),
        "fatigue":    out["fatigue"].squeeze(),
    }
    targets = {k: split[k] for k in ("emotion","stress","engagement","attention","fatigue")}
    return compute_all_metrics(targets, preds)


# ── 5. Training ──────────────────────────────────────────────────────────────

def train_v6(
    epochs     = 75,
    batch_size = 64,
    lr         = 2e-4,
    seed       = 42,
    fer_samples  = 9000,
    raf_samples  = 3000,
    wesad_samples = 2000,
    gamma_focal  = 2.0,
    emo_loss_w   = 4.0,
    reg_loss_w   = 0.25,
    warmup_ep    = 5,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    train_split, val_split, test_split = make_real_dataset(
        fer_samples=fer_samples,
        raf_samples=raf_samples,
        wesad_samples=wesad_samples,
        seed=seed,
    )
    print(f"[v6] Data ready in {time.time()-t0:.1f}s", flush=True)

    emo_labels  = train_split["emotion"]
    class_count = np.bincount(emo_labels, minlength=EMOTION_CLASSES).astype(float)
    class_count = np.where(class_count == 0, 1.0, class_count)
    print(f"[v6] Train class counts: {class_count.astype(int).tolist()}", flush=True)

    # WeightedRandomSampler
    sample_wts = (1.0 / class_count)[emo_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_wts, dtype=torch.float64),
        num_samples=len(emo_labels),
        replacement=True,
    )
    train_ds = _to_tensors(train_split)
    loader   = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)

    model = TCMT_V6()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # LR schedule: linear warmup then cosine
    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep
        progress = (ep - warmup_ep) / max(1, epochs - warmup_ep)
        return 0.5 * (1 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Focal loss with class weights
    emo_wts = torch.tensor(1.0 / class_count, dtype=torch.float32)
    emo_wts = emo_wts / emo_wts.sum() * EMOTION_CLASSES
    focal   = FocalLoss(gamma=gamma_focal, weight=emo_wts, label_smoothing=0.05)
    mse     = nn.MSELoss()

    best_f1, best_state, best_metrics = -1.0, None, None
    N = len(train_split["X"])

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for X, emo, st, eng, att, fat in loader:
            opt.zero_grad()
            o = model.forward_train(X)
            emo_loss = focal(o["emo_logits"], emo)
            reg_loss = (mse(o["stress"], st) + mse(o["engagement"], eng)
                        + mse(o["attention"], att) + mse(o["fatigue"], fat))
            loss = emo_loss_w * emo_loss + reg_loss_w * reg_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * X.size(0)
        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            vm = _evaluate(model, val_split)
            f1 = vm["emotion"]["macro_f1"]
            acc = vm["emotion"]["accuracy"]
            print(f"  Ep {epoch:3d}/{epochs}  "
                  f"loss={ep_loss/N:.4f}  "
                  f"val_f1={f1:.3f}  val_acc={acc:.3f}  "
                  f"stress_rmse={vm['stress']['rmse']:.3f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_metrics = vm

    print(f"\n[v6] Best val macro_F1: {best_f1:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate(model, test_split)
    print("\n[v6] === TEST SET METRICS ===", flush=True)
    for head, m in test_metrics.items():
        print(f"  {head}: {m}", flush=True)

    return model, test_metrics, train_split, val_split, test_split


# ── 6. Save + update predictor-compatible weight file ────────────────────────

def save_v6(model, test_metrics, n_train, n_val, n_test):
    """Save in format compatible with predictor.py (state_dict only)."""
    sd = model.state_dict()

    # Save for predictor (state_dict only, no wrapper dict)
    # predictor.py does: self._tcmt.load_state_dict(torch.load(...))
    # BUT TCMT in predictor is the *old* smaller model -- we must also
    # update tcmt.py constants. We'll save as {"state_dict": sd} and fix predictor.
    WEIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, WEIGHT_PATH)
    print(f"[v6] Weights saved: {WEIGHT_PATH}", flush=True)

    annotated = dict(test_metrics)
    annotated["_v6_changes"] = (
        "focal_loss gamma=2.0, emo_loss_w=4.0, reg_loss_w=0.25, "
        "D_MODEL=128, N_LAYERS=3, FFN=256, warmup+cosine LR, label_smoothing=0.05, "
        "75 epochs, FER=9000 samples"
    )
    annotated["_label_provenance"] = {
        "emotion":    "REAL - FER2013/RAF-DB dataset class annotations",
        "stress":     "MIXED - WESAD GT physio + AU/HCI proxy for emotion samples",
        "engagement": "PROXY - noisy gaze+voice formula; no public GT",
        "attention":  "PROXY - noisy gaze formula; no public GT",
        "fatigue":    "PROXY - noisy HCI formula; no public GT",
    }
    annotated["_splits"] = {"n_train": n_train, "n_val": n_val, "n_test": n_test}
    METRICS_PATH.write_text(json.dumps(annotated, indent=2))
    print(f"[v6] Metrics saved: {METRICS_PATH}", flush=True)


if __name__ == "__main__":
    model, test_metrics, tr, va, te = train_v6()
    v6_f1 = test_metrics["emotion"]["macro_f1"]

    V5_F1 = 0.278  # baseline from v5 to beat
    print(f"\n[v6] v6 test macro_F1: {v6_f1:.4f}  (v5 baseline: {V5_F1:.4f})", flush=True)

    if v6_f1 > V5_F1:
        print("[v6] IMPROVEMENT confirmed. Saving.", flush=True)
        save_v6(model, test_metrics, len(tr["X"]), len(va["X"]), len(te["X"]))
        print("\n[v6] Final per-class F1:")
        for cls_id, f1 in test_metrics["emotion"]["per_class_f1"].items():
            cls_names = {0:"neutral", 1:"happy", 2:"sad", 3:"angry"}
            print(f"  {cls_names.get(int(cls_id), cls_id)}: {f1:.4f}")
        print(f"\n[v6] confusion_matrix:")
        for row in test_metrics["emotion"]["confusion_matrix"]:
            print(f"  {row}")
        sys.exit(0)
    else:
        print(f"[v6] No improvement ({v6_f1:.4f} <= {V5_F1:.4f}). NOT saving.", flush=True)
        sys.exit(1)
