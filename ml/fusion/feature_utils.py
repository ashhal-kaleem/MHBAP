"""
feature_vector.py helpers — build/slice the 58-dim fusion vector.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from ml.fusion.feature_vector import MODALITY_KEYS, FEATURE_DIM

# Precompute slice boundaries
_SLICES: Dict[str, Tuple[int, int]] = {}
_offset = 0
for _mod, _keys in MODALITY_KEYS.items():
    _SLICES[_mod] = (_offset, _offset + len(_keys))
    _offset += len(_keys)


def dicts_to_vector(
    feature_dicts: Dict[str, Dict[str, float]],
    missing_fill: float = 0.0,
) -> np.ndarray:
    """Assemble modality feature dicts → float32 vector of length FEATURE_DIM.

    Missing modalities are filled with `missing_fill`.
    Missing individual keys within a present modality are also filled.
    """
    vec = np.full(FEATURE_DIM, missing_fill, dtype=np.float32)
    for mod, keys in MODALITY_KEYS.items():
        src = feature_dicts.get(mod, {})
        start, end = _SLICES[mod]
        for i, key in enumerate(keys):
            vec[start + i] = float(src.get(key, missing_fill))
    return vec


def modality_slice(modality: str) -> Tuple[int, int]:
    """Return (start, end) index range for a modality in the 58-dim vector."""
    return _SLICES[modality]


def vector_to_modality_dict(vec: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Inverse of dicts_to_vector — useful for debugging."""
    result: Dict[str, Dict[str, float]] = {}
    for mod, keys in MODALITY_KEYS.items():
        start, end = _SLICES[mod]
        result[mod] = {k: float(vec[start + i]) for i, k in enumerate(keys)}
    return result
