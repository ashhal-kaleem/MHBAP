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

    Uses forward activations from the face projection linear layer, weighted by
    the gradient of the chosen head output w.r.t. those activations.
    Falls back to weight-norm saliency if the gradient path yields zeros.
    """
    def __init__(self, model: TCMT) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("GradCAM requires PyTorch")
        self._model = model
        self._activations: Optional[np.ndarray] = None
        self._gradients:   Optional[np.ndarray] = None
        self._hook_handles = []

    def _get_linear_layer(self):
        proj = self._model.mod_proj.projs["face"]
        if hasattr(proj, "__getitem__"):
            return proj[0]  # nn.Sequential(nn.Linear, nn.LayerNorm)[0] -> nn.Linear
        return proj

    def _register_hooks(self) -> None:
        face_layer = self._get_linear_layer()

        def fwd_hook(module, inp, out):
            self._activations = out.detach().cpu().numpy()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach().cpu().numpy()

        self._hook_handles.append(face_layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(
            face_layer.register_full_backward_hook(bwd_hook)
        )

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def explain(
        self,
        feature_vector: np.ndarray,
        head: str = "stress",
    ) -> np.ndarray:
        """
        Compute GradCAM saliency over the 12 face AU features.
        Returns np.ndarray shape (12,) in [0, 1].
        """
        import torch

        self._model.train()
        self._model.zero_grad()
        self._register_hooks()
        self._activations = None
        self._gradients   = None

        try:
            x = torch.tensor(feature_vector, dtype=torch.float32,
                             requires_grad=True).unsqueeze(0)  # (1, F)

            _x = x.unsqueeze(1)          # (1, 1, F)
            B, T, _ = _x.shape
            toks = torch.cat(
                [self._model.mod_proj(_x[:, t, :]) for t in range(T)], dim=1
            )
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
            chosen_head = head_map.get(head, self._model.head_stress)
            out_val = chosen_head(h).sum()

            self._model.zero_grad()
            out_val.backward()

            face_layer = self._get_linear_layer()
            face_W = face_layer.weight.detach().cpu().numpy()  # (D_MODEL, 12)

            if (self._activations is not None and self._gradients is not None):
                # Standard GradCAM: weight activations by mean gradient
                weights = self._gradients.mean(axis=-1, keepdims=True)  # (B,1)
                cam     = (self._activations * weights).sum(axis=0)     # (D_MODEL,)
                cam_relu = np.maximum(cam, 0)
                au_sal  = np.abs(face_W.T @ cam_relu)   # (12,)

                if au_sal.max() < 1e-8:
                    # Fallback: gradient magnitudes projected through weight matrix
                    grad_mag = np.abs(self._gradients).mean(axis=0)  # (D_MODEL,)
                    au_sal   = np.abs(face_W.T @ grad_mag)
            else:
                au_sal = np.zeros(FACE_DIM, dtype=np.float32)

            # Final fallback: weight row norms (always non-zero after training)
            if au_sal.max() < 1e-8:
                au_sal = np.abs(face_W).mean(axis=0)   # (12,)

        finally:
            self._remove_hooks()
            self._model.eval()

        _max = au_sal.max()
        if _max > 1e-8:
            au_sal = au_sal / _max
        return au_sal.astype(np.float32)
