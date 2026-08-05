"""
shap_explainer.py — Integrated-Gradients attribution for MHBAP (Phase E).

Replaces the Gradient×Input proxy with captum.attr.IntegratedGradients,
which satisfies completeness + sensitivity axioms required for research-grade XAI.

Fallback chain:
  1. captum IntegratedGradients  (preferred — axiomatically correct)
  2. Vanilla Gradient×Input      (if captum unavailable)
  3. Feature-magnitude heuristic (if torch unavailable)
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import numpy as np

from ml.fusion.FeatureVector import MODALITY_KEYS
from ml.fusion.FeatureUtils import modality_slice
from ml.fusion.Tcmt import TCMT, _TORCH_AVAILABLE

logger = logging.getLogger(__name__)

TARGET_HEADS = ["stress", "engagement", "attention", "fatigue"]

# Check captum availability once at import
_CAPTUM_AVAILABLE = False
if _TORCH_AVAILABLE:
    try:
        from captum.attr import IntegratedGradients as _IG
        _CAPTUM_AVAILABLE = True
    except ImportError:
        pass


def _head_forward_factory(model: TCMT, head: str):
    """Return a callable (x: Tensor) -> scalar Tensor for one head."""
    import torch

    head_map = {
        "stress":     model.head_stress,
        "engagement": model.head_engagement,
        "attention":  model.head_attention,
        "fatigue":    model.head_fatigue,
    }

    def _forward(x: "torch.Tensor") -> "torch.Tensor":
        # x: (B, F)  — captum passes batched input
        if x.dim() == 2:
            _x = x.unsqueeze(1)
        else:
            _x = x
        B, T, _ = _x.shape
        toks = torch.cat([model.mod_proj(_x[:, t, :]) for t in range(T)], dim=1)
        cls  = model.cls_token.expand(B, -1, -1)
        enc  = model.encoder(torch.cat([cls, toks], dim=1))
        h    = enc[:, 0, :]
        return torch.sigmoid(head_map[head](h)).squeeze(-1)   # (B,)

    return _forward


class SHAPExplainer:
    """
    Modality-level attribution using Integrated Gradients (captum).

    Returns
    -------
    Dict[str, Dict[str, float]]
        {head: {modality: weight_0_to_1}}
    """
    def __init__(self, model: TCMT) -> None:
        self._model = model

    def explain(
        self,
        feature_vector: np.ndarray,
        target_heads: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        heads = target_heads or TARGET_HEADS
        if _CAPTUM_AVAILABLE:
            return self._ig_explain(feature_vector, heads)
        if _TORCH_AVAILABLE:
            return self._grad_explain(feature_vector, heads)
        return self._heuristic_explain(feature_vector, heads)

    # ------------------------------------------------------------------ #
    def _ig_explain(
        self, vec: np.ndarray, heads: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """True Integrated Gradients via captum — 50 Riemann steps."""
        import torch
        from captum.attr import IntegratedGradients

        result: Dict[str, Dict[str, float]] = {}
        baseline = torch.zeros(1, len(vec), dtype=torch.float32)
        inp      = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

        self._model.eval()
        for head in heads:
            try:
                fwd  = _head_forward_factory(self._model, head)
                ig   = IntegratedGradients(fwd)
                attr = ig.attribute(inp, baselines=baseline, n_steps=50)
                attr_np = attr.squeeze(0).detach().numpy()   # (F,)
                result[head] = self._aggregate(np.abs(attr_np))
            except Exception as e:
                logger.warning("IG failed for head %s: %s — falling back", head, e)
                result[head] = self._heuristic_explain(vec, [head])[head]
        return result

    def _grad_explain(
        self, vec: np.ndarray, heads: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Gradient × Input fallback when captum unavailable."""
        import torch
        result: Dict[str, Dict[str, float]] = {}
        self._model.eval()
        for head in heads:
            try:
                leaf = torch.tensor(vec, dtype=torch.float32, requires_grad=True)
                fwd  = _head_forward_factory(self._model, head)
                val  = fwd(leaf.unsqueeze(0)).sum()
                self._model.zero_grad()
                if leaf.grad is not None:
                    leaf.grad.zero_()
                val.backward()
                if leaf.grad is None:
                    result[head] = self._heuristic_explain(vec, [head])[head]
                    continue
                attr = np.abs(leaf.grad.detach().numpy() * vec)
                result[head] = self._aggregate(attr)
            except Exception as e:
                logger.warning("Grad explain failed for %s: %s", head, e)
                result[head] = self._heuristic_explain(vec, [head])[head]
        return result

    def _heuristic_explain(
        self, vec: np.ndarray, heads: List[str]
    ) -> Dict[str, Dict[str, float]]:
        attr = np.abs(vec)
        contrib = self._aggregate(attr)
        return {h: contrib for h in heads}

    def _aggregate(self, attr: np.ndarray) -> Dict[str, float]:
        raw: Dict[str, float] = {}
        for mod in MODALITY_KEYS:
            s, e = modality_slice(mod)
            raw[mod] = float(attr[s:e].sum())
        total = sum(raw.values()) or 1e-8
        return {mod: round(v / total, 4) for mod, v in raw.items()}
