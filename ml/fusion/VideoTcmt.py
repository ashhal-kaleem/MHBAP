"""
VideoTcmt.py — temporal cross-modal transformer for real video benchmarks.

Differences from ml/fusion/Tcmt.py (kept unchanged for the live demo):
  * Learnable TEMPORAL position embedding. The original TCMT has modality
    embeddings but no time embedding, so it cannot tell frame order apart.
  * Key-padding mask for frames where no face was detected.
  * Modality dropout / masking with a learned "missing" token, so the same
    trained model can be ablated by removing modalities at test time, and
    separate models can be trained per modality subset.
  * `temporal=False` collapses the clip to its mean frame (T=1): the
    static baseline for the temporal ablation.
  * One 4-level classification head per DAiSEE affective state.

Input : x (B, T, 28), valid (B, T) bool
Output: logits (B, n_states, n_levels)
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from ml.training.DaiseeDataset import (
    AFFECTIVE_STATES, N_LEVELS, VISUAL_SLICES,
)


class VideoTCMT(nn.Module):
    def __init__(
        self,
        modalities: Sequence[str] = ("face", "gaze", "pose"),
        max_t: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.15,
        temporal: bool = True,
        n_states: int = len(AFFECTIVE_STATES),
        n_levels: int = N_LEVELS,
    ) -> None:
        super().__init__()
        self.modalities = list(modalities)
        assert self.modalities and all(m in VISUAL_SLICES for m in self.modalities)
        self.temporal = temporal
        self.n_states, self.n_levels = n_states, n_levels
        self.projs = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(e - s, d_model), nn.LayerNorm(d_model))
            for m, (s, e) in VISUAL_SLICES.items() if m in self.modalities
        })
        self.mod_embed = nn.Parameter(torch.randn(len(self.modalities), d_model) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(max_t, d_model) * 0.02)
        self.missing = nn.Parameter(torch.zeros(d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_states * n_levels)

    def forward(
        self,
        x: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        drop_modalities: Sequence[str] = (),
    ) -> torch.Tensor:
        B, T, _ = x.shape
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
        if not self.temporal:
            w = valid.float().unsqueeze(-1)
            x = (x * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1.0)
            valid = valid.any(1, keepdim=True)
            T = 1
        toks = []
        for i, m in enumerate(self.modalities):
            s, e = VISUAL_SLICES[m]
            if m in drop_modalities:
                h = self.missing.expand(B, T, -1)
            else:
                h = self.projs[m](x[:, :, s:e])
            toks.append(h + self.mod_embed[i] + self.time_embed[:T])
        tokens = torch.stack(toks, dim=2).reshape(B, T * len(self.modalities), -1)
        pad = (~valid).repeat_interleave(len(self.modalities), dim=1)
        cls = self.cls.expand(B, -1, -1)
        seq = torch.cat([cls, tokens], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], 1)
        h = self.encoder(seq, src_key_padding_mask=pad)
        out = self.head(self.norm(h[:, 0]))
        return out.view(B, self.n_states, self.n_levels)
