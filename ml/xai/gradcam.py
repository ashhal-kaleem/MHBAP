"""
gradcam.py — True Grad-CAM for TCMT face modality encoder.

Hooks into the face projection layer (_ModalityProjection.projs["face"])
and computes gradient-weighted activation maps per output head.

Returns a per-feature saliency vector over the 12 face AU dims,
normalised to [0,1] so it can be rendered as a heatmap in the dashboard.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np

from ml.fusion.tcmt import TCMT, _TORCH_AVAILABLE
from ml.fusion.feature_utils import modality_slice

FACE_DIM = 12   # au_* keys


class GradCAM:
    """
    Gradient-weighted Class Activation Map over face AU features.

    Parameters
    ----------
    model : TCMT  (must be a real torch TCMT, not the stub)
    """
    def __init__(self, model: TCMT) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("GradCAM requires PyTorch")
        self._model = model
        self._activations: Optional[np.ndarray] = None
        self._gradients:   Optional[np.ndarray] = None
        self._hook_handles = []

    # ------------------------------------------------------------------ #
    def _register_hooks(self) -> None:
        import torch
        face_layer = self._model.mod_proj.projs["face"]   # nn.Linear(12, 64)

        def fwd_hook(module, inp, out):
            # out: (B, 64) — face projection activations
            self._activations = out.detach().cpu().numpy()

        def bwd_hook(module, grad_in, grad_out):
            # grad_out[0]: (B, 64)
            self._gradients = grad_out[0].detach().cpu().numpy()

        self._hook_handles.append(face_layer.register_forward_hook(fwd_hook))
        # Use full backward hook to avoid FutureWarning in torch >= 2.x
        self._hook_handles.append(
            face_layer.register_full_backward_hook(bwd_hook)
        )

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------ #
    def explain(
        self,
        feature_vector: np.ndarray,   # (FEATURE_DIM,)
        head: str = "stress",
    ) -> np.ndarray:
        """
        Compute GradCAM saliency over the 12 face AU features.

        Returns
        -------
        np.ndarray shape (12,)  values in [0, 1]
        """
        import torch

        self._model.train()   # enable grad even in eval mode
        self._register_hooks()

        try:
            x = torch.tensor(feature_vector, dtype=torch.float32,
                             requires_grad=True).unsqueeze(0)  # (1, F)

            # Forward — need tensor outputs, not numpy
            if x.dim() == 2:
                _x = x.unsqueeze(1)
            B, T, _ = _x.shape
            toks = torch.cat([self._model.mod_proj(_x[:, t, :]) for t in range(T)], dim=1)
            cls  = self._model.cls_token.expand(B, -1, -1)
            enc  = self._model.encoder(torch.cat([cls, toks], dim=1))
            h    = enc[:, 0, :]

            head_map = {
                "stress":     self._model.head_stress,
                "engagement": self._model.head_engagement,
                "attention":  self._model.head_attention,
                "fatigue":    self._model.head_fatigue,
                "emotion":    self._model.head_emotion,
            }
            if head not in head_map:
                head = "stress"
            out_val = head_map[head](h).sum()

            self._model.zero_grad()
            out_val.backward()

            if self._activations is None or self._gradients is None:
                return np.zeros(FACE_DIM, dtype=np.float32)

            # Global-average-pool gradients over D_MODEL dim → scalar weight per channel
            weights = self._gradients.mean(axis=-1, keepdims=True)  # (B,1)
            cam     = (self._activations * weights).mean(axis=0)     # (64,)
            cam     = np.maximum(cam, 0)                             # ReLU

            # Project 64-dim cam back to 12 face AUs via weight matrix of face linear
            face_W  = self._model.mod_proj.projs["face"].weight.detach().cpu().numpy()  # (64,12)
            au_saliency = np.abs(face_W.T @ cam)    # (12,)

        finally:
            self._remove_hooks()
            self._model.eval()

        # Normalise
        _max = au_saliency.max()
        if _max > 1e-8:
            au_saliency = au_saliency / _max
        return au_saliency.astype(np.float32)
