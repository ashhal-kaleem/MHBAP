"""
shap_explainer.py — gradient-based SHAP proxy for MHBAP.

When PyTorch is available, uses vanilla gradients w.r.t. input features
as a cheap SHAP approximation (gradient × input, a.k.a. Gradient×Input).
Aggregates per-feature attributions into per-modality contributions.

When torch is absent, falls back to a heuristic (feature magnitude).
"""
from __future__ import annotations
import logging
from typing import Dict, Optional
import numpy as np
from ml.fusion.feature_vector import MODALITY_KEYS
from ml.fusion.feature_utils import modality_slice
from ml.fusion.tcmt import TCMT, _TORCH_AVAILABLE

logger = logging.getLogger(__name__)

TARGET_HEADS = ["stress", "engagement", "attention", "fatigue"]


class SHAPExplainer:
    """
    Computes modality-level attribution weights for each prediction head.

    Returns
    -------
    Dict[str, Dict[str, float]]
        {head: {modality: weight_0_to_1}}
    """
    def __init__(self, model: TCMT) -> None:
        self._model = model
        self._torch = _TORCH_AVAILABLE

    # ------------------------------------------------------------------
    def explain(
        self,
        feature_vector: np.ndarray,          # (58,)
        target_heads: Optional[list] = None,
    ) -> Dict[str, Dict[str, float]]:
        heads = target_heads or TARGET_HEADS
        if self._torch:
            return self._grad_explain(feature_vector, heads)
        return self._heuristic_explain(feature_vector, heads)

    def _grad_explain(
        self, vec: np.ndarray, heads: list
    ) -> Dict[str, Dict[str, float]]:
        import torch
        # leaf tensor keeps .grad; unsqueeze for model input
        leaf = torch.tensor(vec, dtype=torch.float32, requires_grad=True)
        leaf.retain_grad()
        x = leaf.unsqueeze(0)   # (1, 57)
        out = self._model(x)
        result: Dict[str, Dict[str, float]] = {}
        for head in heads:
            if head not in out:
                continue
            head_val = out[head]
            # TCMT numpy path returns numpy scalars/arrays, not tensors
            import torch as _torch
            if not isinstance(head_val, _torch.Tensor):
                result[head] = self._heuristic_explain(vec, [head])[head]
                continue
            loss = head_val.sum()
            self._model.zero_grad()
            if leaf.grad is not None:
                leaf.grad.zero_()
            loss.backward(retain_graph=True)
            if leaf.grad is None:
                result[head] = self._heuristic_explain(vec, [head])[head]
                continue
            grad = leaf.grad.detach().numpy()
            attr = np.abs(grad * vec)
            result[head] = self._aggregate(attr)
        return result

    def _heuristic_explain(
        self, vec: np.ndarray, heads: list
    ) -> Dict[str, Dict[str, float]]:
        """Feature-magnitude heuristic — same for all heads."""
        attr = np.abs(vec)
        contrib = self._aggregate(attr)
        return {h: contrib for h in heads}

    def _aggregate(self, attr: np.ndarray) -> Dict[str, float]:
        """Sum attributions within each modality slice, then normalise."""
        raw: Dict[str, float] = {}
        for mod in MODALITY_KEYS:
            start, end = modality_slice(mod)
            raw[mod] = float(attr[start:end].sum())
        total = sum(raw.values()) or 1e-8
        return {mod: round(v / total, 4) for mod, v in raw.items()}
