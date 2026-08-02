"""
tcmt.py — Temporal Cross-Modal Transformer for MHBAP.

Architecture
------------
Input : (B, T, 58) float32  — B=batch, T=time steps, 58 feature dims
Output: dict of 5 head tensors:
  emotion_logits : (B, 4)   — 4-class: 0=neutral,1=happy,2=sad,3=angry
  stress         : (B, 1)   — sigmoid → [0, 10] scaled
  engagement     : (B, 1)   — sigmoid → [0, 1]
  attention      : (B, 1)   — sigmoid → [0, 1]
  fatigue        : (B, 1)   — sigmoid → [0, 1]

Design choices for research-grade MITACS presentation
------------------------------------------------------
1. Per-modality linear projections with LayerNorm before shared transformer —
   lets the model learn modality-specific embeddings.
2. Learnable modality-type embeddings added at input —
   explicit cross-modal positional signal.
3. Shared TransformerEncoder (4 heads, 3 layers, d_model=128, dim_feedforward=256).
4. CLS token aggregation for temporal summary.
5. Separate prediction heads per behavioural target (2-layer MLP for emotion).
"""
from __future__ import annotations
import math
from typing import Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from ml.fusion.feature_vector import MODALITY_KEYS, FEATURE_DIM

# Rebuild slices locally in case feature_vector hasn't set _SLICES at module level
_MOD_SLICES: Dict[str, tuple] = {}
_off = 0
for _m, _ks in MODALITY_KEYS.items():
    _MOD_SLICES[_m] = (_off, _off + len(_ks))
    _off += len(_ks)

D_MODEL  = 128
N_HEADS  = 4
N_LAYERS = 3
FFN_DIM  = 256
EMOTION_CLASSES = 4   # 0=neutral, 1=happy, 2=sad, 3=angry


if _TORCH_AVAILABLE:
    class _ModalityProjection(nn.Module):
        """Project each modality slice to D_MODEL with Linear + LayerNorm."""
        def __init__(self) -> None:
            super().__init__()
            self.projs = nn.ModuleDict({
                mod: nn.Sequential(nn.Linear(end - start, D_MODEL), nn.LayerNorm(D_MODEL))
                for mod, (start, end) in _MOD_SLICES.items()
            })
            # Learnable modality-type embedding
            n_mods = len(MODALITY_KEYS)
            self.mod_embed = nn.Embedding(n_mods, D_MODEL)
            self._mod_idx = {m: i for i, m in enumerate(MODALITY_KEYS)}

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, FEATURE_DIM)  →  out: (B, n_mods, D_MODEL)
            parts = []
            for mod, (start, end) in _MOD_SLICES.items():
                proj = self.projs[mod](x[:, start:end])          # (B, D_MODEL)
                idx  = self._mod_idx[mod]
                proj = proj + self.mod_embed(
                    torch.tensor(idx, device=x.device))
                parts.append(proj.unsqueeze(1))                   # (B,1,D_MODEL)
            return torch.cat(parts, dim=1)                        # (B,n_mods,D_MODEL)


    class TCMT(nn.Module):
        """
        Temporal Cross-Modal Transformer.

        Input shape : (B, T, FEATURE_DIM)  or  (B, FEATURE_DIM) for T=1
        """
        def __init__(self) -> None:
            super().__init__()
            self.mod_proj = _ModalityProjection()
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEADS,
                dim_feedforward=FFN_DIM, dropout=0.15,
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))

            # Output heads
            self.head_emotion = nn.Sequential(
                nn.Linear(D_MODEL, D_MODEL // 2), nn.GELU(),
                nn.Dropout(0.15), nn.Linear(D_MODEL // 2, EMOTION_CLASSES),
            )
            self.head_stress     = nn.Linear(D_MODEL, 1)
            self.head_engagement = nn.Linear(D_MODEL, 1)
            self.head_attention  = nn.Linear(D_MODEL, 1)
            self.head_fatigue    = nn.Linear(D_MODEL, 1)

        def forward(self, x) -> Dict[str, "torch.Tensor"]:
            # Accept numpy arrays or torch tensors, shapes (B,F) or (B,T,F)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x.astype(np.float32))
            if x.dim() == 2:
                x = x.unsqueeze(1)                               # (B,1,F)
            B, T, F = x.shape

            # Project each time step
            tokens_list = []
            for t in range(T):
                tokens_list.append(self.mod_proj(x[:, t, :]))    # (B, n_mods, D)
            # (B, T*n_mods, D)
            tokens = torch.cat(tokens_list, dim=1)

            # Prepend CLS token
            cls = self.cls_token.expand(B, -1, -1)               # (B,1,D)
            seq = torch.cat([cls, tokens], dim=1)                 # (B,1+T*n_mods,D)

            enc = self.encoder(seq)                               # (B, seq_len, D)
            cls_out = enc[:, 0, :]                                # (B, D)

            return {
                "emotion_logits": self.head_emotion(cls_out).detach().numpy(),
                "stress":    (torch.sigmoid(self.head_stress(cls_out)) * 10).detach().numpy(),
                "engagement": torch.sigmoid(self.head_engagement(cls_out)).detach().numpy(),
                "attention":  torch.sigmoid(self.head_attention(cls_out)).detach().numpy(),
                "fatigue":    torch.sigmoid(self.head_fatigue(cls_out)).detach().numpy(),
            }


else:
    # Fallback when PyTorch not installed
    class TCMT:  # type: ignore[no-redef]
        """Stub TCMT — returns rule-based predictions when torch absent."""
        def __init__(self) -> None:
            pass

        def __call__(self, x: np.ndarray) -> Dict[str, np.ndarray]:
            B = x.shape[0] if hasattr(x, "shape") else 1
            return {
                "emotion_logits": np.ones((B, EMOTION_CLASSES)) / EMOTION_CLASSES,
                "stress":     np.full((B, 1), 5.0),
                "engagement": np.full((B, 1), 0.5),
                "attention":  np.full((B, 1), 0.5),
                "fatigue":    np.full((B, 1), 0.5),
            }
