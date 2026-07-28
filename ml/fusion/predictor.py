"""
predictor.py — high-level inference wrapper around TCMT.

Maintains a ring-buffer of T=8 feature vectors for temporal context.
Accepts raw modality feature dicts from Phase 4 pipelines,
returns a PredictionResult with scores + emotion label.
"""
from __future__ import annotations
import collections
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

from ml.fusion.feature_utils import dicts_to_vector
from ml.fusion.feature_vector import FEATURE_DIM
from ml.fusion.tcmt import TCMT, EMOTION_CLASSES, _TORCH_AVAILABLE

logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    "neutral","happy","sad","angry","surprised","fearful","disgusted","contemptuous"
]
T_STEPS = 8                        # temporal context window
WEIGHTS_PATH = Path("ml/models/weights/tcmt.pt")


@dataclass
class PredictionResult:
    emotion: str
    emotion_scores: Dict[str, float]
    stress: float          # 0–10
    engagement: float      # 0–1
    attention: float       # 0–1
    fatigue: float         # 0–1
    timestamp: float = field(default_factory=time.time)
    feature_vector: Optional[np.ndarray] = field(default=None, repr=False)


class BehaviourPredictor:
    """
    Wraps TCMT for real-time inference.

    Usage
    -----
    predictor = BehaviourPredictor()
    result = predictor.predict({"face": {...}, "gaze": {...}, ...})
    """

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        self._model = TCMT()
        self._buffer: Deque[np.ndarray] = collections.deque(maxlen=T_STEPS)
        self._torch = _TORCH_AVAILABLE

        if self._torch:
            import torch
            self._torch_mod = torch
            self._model.eval()
            wp = weights_path or WEIGHTS_PATH
            if wp.exists():
                self._model.load_state_dict(torch.load(wp, map_location="cpu"))
                logger.info("TCMT weights loaded from %s", wp)
            else:
                logger.warning("No weights at %s — using random init", wp)

    # ------------------------------------------------------------------
    def predict(self, feature_dicts: Dict[str, Dict[str, float]]) -> PredictionResult:
        vec = dicts_to_vector(feature_dicts)
        self._buffer.append(vec)

        # Pad buffer to T_STEPS with zeros if needed
        buf_len = len(self._buffer)
        pad_n   = T_STEPS - buf_len
        arr = np.stack(
            [np.zeros(FEATURE_DIM, dtype=np.float32)] * pad_n
            + list(self._buffer),
            axis=0,
        )  # (T, F)

        if self._torch:
            import torch
            with torch.no_grad():
                x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1,T,F)
                out = self._model(x)
            emotion_logits = out["emotion_logits"][0].numpy()
            stress     = float(out["stress"][0, 0])
            engagement = float(out["engagement"][0, 0])
            attention  = float(out["attention"][0, 0])
            fatigue    = float(out["fatigue"][0, 0])
        else:
            out = self._model(arr[np.newaxis])                            # (1,T,F)
            emotion_logits = out["emotion_logits"][0]
            stress     = float(out["stress"][0, 0])
            engagement = float(out["engagement"][0, 0])
            attention  = float(out["attention"][0, 0])
            fatigue    = float(out["fatigue"][0, 0])

        # Softmax emotion scores
        exp = np.exp(emotion_logits - emotion_logits.max())
        probs = exp / exp.sum()
        top_idx  = int(np.argmax(probs))

        return PredictionResult(
            emotion=EMOTION_LABELS[top_idx],
            emotion_scores={EMOTION_LABELS[i]: float(probs[i]) for i in range(EMOTION_CLASSES)},
            stress=round(stress, 3),
            engagement=round(engagement, 3),
            attention=round(attention, 3),
            fatigue=round(fatigue, 3),
            feature_vector=vec,
        )
