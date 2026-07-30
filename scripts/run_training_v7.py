"""
run_training_v7.py -- TCMT v7: improved face features + balanced sampling + tuned training.

Key improvements vs v6:
  1. Richer face feature extraction: std dev, contrast, gradient proxy, region ratios
     → more discriminative signal from face images (still 12 dims, no FEATURE_DIM change)
  2. Per-class cap in dataset builder: limit neutral/happy to 3x angry count
     → avoids 12:1 class ratio in raw data
  3. Higher focal gamma=3.0, emo_loss_w=6.0
  4. 100 epochs, batch_size=128
  5. Save state_dict compatible with predictor.py (state_dict key)
  6. Update tcmt.py with v6 architecture constants and save-compatible forward()

Usage: cd D:\\MHBAP && python scripts/run_training_v7.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WEIGHT_PATH  = Path("ml/models/weights/tcmt_trained.pt")
METRICS_PATH = Path("ml/models/weights/tcmt_eval_metrics.json")

FEATURE_DIM     = 58
EMOTION_CLASSES = 4
D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 3
FFN_DIM  = 256
DROPOUT  = 0.15

print("=" * 60)
print("TCMT v7: richer face features + class cap + tuned loss")
print("=" * 60, flush=True)

# ── Model (same as v6) ───────────────────────────────────────────────────────

from ml.fusion.feature_vector import MODALITY_KEYS

_MOD_SLICES: dict = {}
_off = 0
for _m, _ks in MODALITY_KEYS.items():
    _MOD_SLICES[_m] = (_off, _off + len(_ks))
    _off += len(_ks)


class _ModalityProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.projs = nn.ModuleDict({
            mod: nn.Sequential(nn.Linear(end - start, D_MODEL), nn.LayerNorm(D_MODEL))
            for mod, (start, end) in _MOD_SLICES.items()
        })
        n = len(MODALITY_KEYS)
        self.mod_embed = nn.Embedding(n, D_MODEL)
        self._idx = {m: i for i, m in enumerate(MODALITY_KEYS)}

    def forward(self, x):
        parts = []
        for mod, (s, e) in _MOD_SLICES.items():
            p = self.projs[mod](x[:, s:e])
            p = p + self.mod_embed(torch.tensor(self._idx[mod], device=x.device))
            parts.append(p.unsqueeze(1))
        return torch.cat(parts, dim=1)


class TCMT_V7(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod_proj = _ModalityProj()
        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder   = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.normal_(self.cls_token, std=0.02)

        self.head_emotion = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2), nn.GELU(),
            nn.Dropout(0.15), nn.Linear(D_MODEL // 2, EMOTION_CLASSES),
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
        enc    = self.encoder(torch.cat([cls, tokens], dim=1))
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


# ── Improved face feature extraction ────────────────────────────────────────

def _img_to_face_features_v7(img_array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    12-dim face features from image pixel statistics.
    Improvements over v6:
      - std dev (texture) per region for discriminative signal
      - gradient proxy (edge energy) in brow/mouth
      - inter-region difference (brow vs mouth, left vs right eye asymmetry)
      - lip corner ratio (happy indicator)
      - all 12 dims packed with more discriminative combinations
    """
    gray = img_array.mean(axis=2) if img_array.ndim == 3 else img_array.astype(float)
    H, W = gray.shape
    g = gray / 255.0

    def reg_stats(r0, r1, c0, c1):
        a = g[int(H*r0):int(H*r1), int(W*c0):int(W*c1)]
        if a.size == 0:
            return 0.5, 0.1
        return float(a.mean()), float(a.std())

    brow_m,  brow_s  = reg_stats(0.10, 0.30, 0.20, 0.80)
    el_m,    el_s    = reg_stats(0.25, 0.45, 0.10, 0.45)
    er_m,    er_s    = reg_stats(0.25, 0.45, 0.55, 0.90)
    nose_m,  _       = reg_stats(0.40, 0.60, 0.30, 0.70)
    mth_m,   mth_s   = reg_stats(0.60, 0.85, 0.25, 0.75)
    jaw_m,   jaw_s   = reg_stats(0.75, 0.95, 0.20, 0.80)
    ml_m,    _       = reg_stats(0.60, 0.80, 0.20, 0.45)  # mouth left corner
    mr_m,    _       = reg_stats(0.60, 0.80, 0.55, 0.80)  # mouth right corner
    up_m,    _       = reg_stats(0.00, 0.20, 0.20, 0.80)  # upper forehead

    # Gradient proxy: abs diff between adjacent rows (edge energy)
    brow_region = g[int(H*0.10):int(H*0.30), int(W*0.20):int(W*0.80)]
    mth_region  = g[int(H*0.60):int(H*0.85), int(W*0.25):int(W*0.75)]
    brow_grad = float(np.abs(np.diff(brow_region, axis=0)).mean()) if brow_region.size > 1 else 0.1
    mth_grad  = float(np.abs(np.diff(mth_region,  axis=0)).mean()) if mth_region.size > 1 else 0.1

    # Eye asymmetry (asymmetry higher in disgust/anger)
    eye_asym = float(abs(el_m - er_m))

    # Mouth corner asymmetry (smile = symmetric; frown = can be asymmetric)
    mouth_asym = float(abs(ml_m - mr_m))

    # Lip aperture proxy (dark mouth open area)
    lip_aperture = float(np.clip(1.0 - mth_m * 1.5, 0, 1))

    # Brow-mouth contrast (angry: dark brow + tense mouth)
    brow_mouth_contrast = float(np.clip(abs(brow_m - mth_m) * 2, 0, 1))

    n = rng.uniform
    return np.array([
        float(np.clip(brow_grad * 8,            0, 1)),   # AU1: brow edge energy
        float(np.clip(brow_s * 6,               0, 1)),   # AU2: brow texture
        float(np.clip(1.0 - brow_m * 1.5,       0, 1)),   # AU3: brow darkness (furrow)
        float(np.clip(1.0 - el_m + n(-0.02,0.02), 0, 1)), # AU4: left eye open
        float(np.clip(1.0 - er_m + n(-0.02,0.02), 0, 1)), # AU5: right eye open
        float(np.clip(eye_asym * 4,             0, 1)),   # AU6: eye asymmetry
        float(np.clip(mth_grad * 8,             0, 1)),   # AU7: mouth edge energy
        float(np.clip(1.0 - mouth_asym * 3,    0, 1)),   # AU8: mouth symmetry (high=smile)
        float(np.clip(lip_aperture,             0, 1)),   # AU9: lip aperture
        float(np.clip(jaw_s * 5,               0, 1)),   # AU10: jaw texture
        float(np.clip(brow_mouth_contrast,      0, 1)),   # AU11: brow-mouth contrast
        float(np.clip(mth_s * 5,               0, 1)),   # AU12: mouth texture
    ], dtype=np.float32)
