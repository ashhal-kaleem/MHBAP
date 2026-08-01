"""
predictor.py — high-level inference wrapper for MHBAP.

Architecture (Phase A production update)
-----------------------------------------
Emotion classification:   EmotionRecognizer (EfficientNet-B0, AffectNet-8 pretrained)
Behavioural indices:      TCMT (Temporal Cross-Modal Transformer)
  stress / engagement / attention / fatigue all come from the TCMT heads

The split is intentional: EmotionRecognizer operates on the raw BGR face
frame and produces state-of-the-art AffectNet-8 classification; the TCMT
fuses all five modality feature streams (face AU proxies, gaze, pose, voice,
HCI) and specialises in the *continuous* behavioural indices that require
temporal context across 8-second windows.

When the EmotionRecognizer checkpoint is absent (first run before download,
or CI environments without ML deps), it falls back to TCMT logits so the
rest of the system keeps functioning.
"""
from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np

from ml.fusion.feature_utils import dicts_to_vector
from ml.fusion.feature_vector import FEATURE_DIM
from ml.fusion.tcmt import TCMT, EMOTION_CLASSES, _TORCH_AVAILABLE

logger = logging.getLogger(__name__)

# AffectNet-8 labels (must match EmotionRecognizer.EMOTION_LABELS)
EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]

T_STEPS      = 8
WEIGHTS_PATH = Path(__file__).parent.parent / "models" / "weights" / "tcmt.pt"


@dataclass
class PredictionResult:
    emotion: str
    emotion_scores: Dict[str, float]
    stress: float          # 0-1
    engagement: float      # 0-1
    attention: float       # 0-1
    fatigue: float         # 0-1
    valence: float = 0.0   # continuous VA from pretrained recognizer
    arousal: float = 0.5
    timestamp: float = field(default_factory=time.time)
    feature_vector: Optional[np.ndarray] = field(default=None, repr=False)
    emotion_source: str = "tcmt"   # "pretrained" | "tcmt"


class BehaviourPredictor:
    """
    Fuses real emotion recognition (pretrained EfficientNet-B0) with
    TCMT-based continuous behavioural indices.

    Usage
    -----
    predictor = BehaviourPredictor()
    result = predictor.predict(
        feature_dicts={"face": {...}, "gaze": {...}, ...},
        bgr_frame=frame,   # optional; enables real emotion recognition
    )
    """

    def __init__(self, weights_path: Optional[Path] = None, device: str = "cpu") -> None:
        self._tcmt   = TCMT()
        self._buffer: Deque[np.ndarray] = collections.deque(maxlen=T_STEPS)
        self._torch  = _TORCH_AVAILABLE
        self._device = device

        if self._torch:
            import torch
            self._tcmt.eval()
            wp = weights_path or WEIGHTS_PATH
            # Fallback: trainer saves as tcmt_trained.pt; accept either name.
            if not wp.exists():
                alt = wp.parent / "tcmt_trained.pt"
                if alt.exists():
                    wp = alt
            if wp.exists():
                raw = torch.load(str(wp), map_location="cpu", weights_only=True)
                # train_tcmt.py wraps weights in {"state_dict": ..., "test_metrics": ...}.
                # Handle both the wrapper-dict format and a bare state_dict for
                # backward-compatibility with any legacy checkpoints saved directly.
                state_dict = (
                    raw["state_dict"]
                    if isinstance(raw, dict) and "state_dict" in raw
                    else raw
                )
                self._tcmt.load_state_dict(state_dict)
                logger.info("TCMT weights loaded from %s", wp)
            else:
                logger.info(
                    "No TCMT weights at %s — behavioural indices use random-init TCMT; "
                    "emotion uses pretrained EmotionRecognizer.", wp
                )

        self._emotion_rec = None
        try:
            from ml.models.emotion_recognizer import EmotionRecognizer
            self._emotion_rec = EmotionRecognizer(device=device, auto_download=True)
            if self._emotion_rec.is_ready:
                logger.info("EmotionRecognizer ready — pretrained AffectNet-8 active")
            else:
                logger.warning("EmotionRecognizer not ready — using TCMT emotion logits")
        except Exception as exc:
            logger.warning("Could not load EmotionRecognizer: %s", exc)

    @property
    def _model(self):
        """Expose TCMT for SHAPExplainer compatibility."""
        return self._tcmt

    def predict(
        self,
        feature_dicts: Dict[str, Dict[str, float]],
        bgr_frame: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        """
        Parameters
        ----------
        feature_dicts : {modality: {feature: value}} from capture pipelines
        bgr_frame     : raw BGR frame for EmotionRecognizer (optional)
        """
        vec = dicts_to_vector(feature_dicts)
        self._buffer.append(vec)

        pad_n = T_STEPS - len(self._buffer)
        arr   = np.stack(
            [np.zeros(FEATURE_DIM, dtype=np.float32)] * pad_n + list(self._buffer), axis=0
        )  # (T, F)

        if self._torch:
            import torch
            with torch.no_grad():
                x   = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
                out = self._tcmt(x)
            tcmt_logits = np.array(out["emotion_logits"][0])
            stress     = float(out["stress"][0, 0]) / 10.0
            engagement = float(out["engagement"][0, 0])
            attention  = float(out["attention"][0, 0])
            fatigue    = float(out["fatigue"][0, 0])
        else:
            out = self._tcmt(arr[np.newaxis])
            tcmt_logits = np.array(out["emotion_logits"][0])
            stress     = float(out["stress"][0, 0]) / 10.0
            engagement = float(out["engagement"][0, 0])
            attention  = float(out["attention"][0, 0])
            fatigue    = float(out["fatigue"][0, 0])

        stress     = float(np.clip(stress,     0.0, 1.0))
        engagement = float(np.clip(engagement, 0.0, 1.0))
        attention  = float(np.clip(attention,  0.0, 1.0))
        fatigue    = float(np.clip(fatigue,    0.0, 1.0))

        # Emotion: pretrained recognizer if frame provided, else TCMT fallback
        valence = 0.0
        arousal = 0.5
        emotion_source = "tcmt"

        if (
            self._emotion_rec is not None
            and self._emotion_rec.is_ready
            and bgr_frame is not None
        ):
            er  = self._emotion_rec.predict(bgr_frame)
            emotion_label  = er["emotion_label"]
            emotion_scores = er["emotion_scores"]
            valence        = er["valence"]
            arousal        = er["arousal"]
            emotion_source = "pretrained"
        else:
            exp   = np.exp(tcmt_logits - tcmt_logits.max())
            probs = exp / exp.sum()
            top   = int(np.argmax(probs))
            emotion_label  = EMOTION_LABELS[top]
            emotion_scores = {EMOTION_LABELS[i]: round(float(probs[i]), 4) for i in range(EMOTION_CLASSES)}

        return PredictionResult(
            emotion        = emotion_label,
            emotion_scores = emotion_scores,
            stress         = round(stress,     3),
            engagement     = round(engagement, 3),
            attention      = round(attention,  3),
            fatigue        = round(fatigue,    3),
            valence        = round(valence,    4),
            arousal        = round(arousal,    4),
            feature_vector = vec,
            emotion_source = emotion_source,
        )
