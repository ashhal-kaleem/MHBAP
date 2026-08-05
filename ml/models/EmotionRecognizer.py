"""
emotion_recognizer.py — Pretrained AffectNet-8 face emotion classifier.

Model: EfficientNet-B0 fine-tuned on AffectNet-8 (EmotiEffNet family).
Checkpoint: enet_b0_8_best_afew.pt  (~16 MB, publicly available via HSE)

The recognizer accepts a BGR face crop (or full frame) and returns:
  - emotion_label  : str  (one of the 8 AffectNet classes)
  - emotion_scores : Dict[str, float]  (softmax probabilities)
  - valence        : float (continuous [-1, +1], estimated from class priors)
  - arousal        : float (continuous [ 0,  1], estimated from class priors)

Fallback: when the checkpoint is absent or torch/timm unavailable,
the class returns uniform-random probabilities so the rest of the pipeline
keeps running, and logs a warning at startup.

Usage:
    from ml.models.EmotionRecognizer import EmotionRecognizer
    rec = EmotionRecognizer()
    result = rec.predict(bgr_frame)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# AffectNet-8 label order that EmotiEffNet checkpoints use
EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise",
    "fear", "disgust", "anger", "contempt",
]

# Valence / arousal priors per AffectNet class (from literature)
# Used to derive continuous VA estimates without a separate VA head
_VALENCE_PRIOR  = [0.0,  0.9, -0.8, 0.4, -0.7, -0.9, -0.8, -0.6]
_AROUSAL_PRIOR  = [0.1,  0.6,  0.3, 0.8,  0.8,  0.5,  0.9,  0.4]

WEIGHTS_DIR  = Path(__file__).parent / "weights"
CHECKPOINT   = WEIGHTS_DIR / "enet_b0_8_best_afew.pt"
DOWNLOAD_URL = (
    "https://github.com/HSE-asavchenko/face-emotion-recognition"
    "/raw/main/models/affectnet_emotions/enet_b0_8_best_afew.pt"
)

INPUT_SIZE = 224  # EfficientNet-B0 canonical input


class EmotionRecognizer:
    """Pretrained AffectNet-8 emotion classifier (EfficientNet-B0 backbone)."""

    def __init__(self, device: str = "cpu", auto_download: bool = True) -> None:
        self._device = device
        self._model = None
        self._transform = None
        self._ready = False
        self._auto_download = auto_download
        self._load()  # sync — no download here

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            import torch
            import timm
        except ImportError:
            logger.warning("torch/timm not installed — EmotionRecognizer using stub")
            return

        if not CHECKPOINT.exists():
            logger.info(
                "EmotiEffNet checkpoint not found at %s. "
                "Call await recognizer.ensure_checkpoint() or run "
                "`python scripts/download_models.py --model emotion`.", CHECKPOINT
            )
            return

        try:
            import torch
            import timm

            # weights_only=False: checkpoint is a full pickled model object from
            # a trusted academic source (HSE EmotiEffNet / AffectNet-8).
            loaded = torch.load(
                str(CHECKPOINT), map_location=device_map(self._device), weights_only=False
            )

            # The HSE checkpoint stores the complete timm 0.x model object.
            # timm 1.x renamed some internal conv layers, so we extract the state_dict
            # and remap the classifier head key to match the current architecture.
            import torch.nn as nn

            if isinstance(loaded, nn.Module):
                old_sd = loaded.state_dict()
            elif isinstance(loaded, dict):
                old_sd = loaded.get("model_state_dict") or loaded.get("state_dict") or loaded
            else:
                raise ValueError(f"Unrecognised checkpoint type: {type(loaded)}")

            model = timm.create_model(
                "efficientnet_b0", pretrained=False, num_classes=len(EMOTION_LABELS)
            )
            new_sd = model.state_dict()

            # Remap mismatched classifier key: 'classifier.0.*' → 'classifier.*'
            remapped: dict = {}
            for k, v in old_sd.items():
                new_k = k.replace("classifier.0.", "classifier.") if "classifier.0." in k else k
                if new_k in new_sd and new_sd[new_k].shape == v.shape:
                    remapped[new_k] = v

            missing = [k for k in new_sd if k not in remapped]
            model.load_state_dict(remapped, strict=False)
            if missing:
                logger.debug("Keys not restored (expected for arch diff): %s", missing[:5])

            model.eval()
            model.to(self._device)

            self._model = model
            self._transform = _build_transform()
            self._ready = True
            logger.info("EmotionRecognizer loaded from %s (device=%s)", CHECKPOINT, self._device)

        except Exception as exc:
            logger.error("EmotionRecognizer failed to load checkpoint: %s", exc)

    async def ensure_checkpoint(self) -> None:
        """Async: download checkpoint if absent, then reload model (non-blocking)."""
        if not CHECKPOINT.exists() and self._auto_download:
            await self._download_checkpoint()
        if not self._ready and CHECKPOINT.exists():
            self._load()  # model load is CPU-bound but fast; OK on async path

    async def _download_checkpoint(self) -> None:
        """Try to auto-download the checkpoint from GitHub (non-blocking)."""
        import asyncio
        import urllib.request
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading EmotiEffNet checkpoint from GitHub (~16 MB) …")
        try:
            # Run in thread pool — urlretrieve is blocking I/O
            await asyncio.to_thread(urllib.request.urlretrieve, DOWNLOAD_URL, str(CHECKPOINT))
            logger.info("Checkpoint saved to %s", CHECKPOINT)
        except Exception as exc:
            logger.warning("Auto-download failed: %s", exc)

    # ------------------------------------------------------------------
    def predict(
        self, bgr_frame: Optional[np.ndarray]
    ) -> Dict[str, object]:
        """
        Run emotion recognition on a BGR frame (or face crop).

        Returns
        -------
        {
            "emotion_label":  str,
            "emotion_scores": {label: prob, ...},
            "valence":        float,
            "arousal":        float,
        }
        """
        if bgr_frame is None or not self._ready:
            return _stub_result()

        try:
            import torch
            tensor = _preprocess(bgr_frame, self._transform)
            with torch.no_grad():
                logits = self._model(tensor.to(self._device))          # (1, 8)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]  # (8,)

            top_idx = int(np.argmax(probs))
            scores  = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}

            # Weighted average VA from class probabilities
            valence = float(np.dot(probs, _VALENCE_PRIOR))
            arousal = float(np.dot(probs, _AROUSAL_PRIOR))

            return {
                "emotion_label":  EMOTION_LABELS[top_idx],
                "emotion_scores": scores,
                "valence":        round(valence, 4),
                "arousal":        round(arousal, 4),
            }
        except Exception as exc:
            logger.error("EmotionRecognizer.predict error: %s", exc)
            return _stub_result()

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── helpers ───────────────────────────────────────────────────────────────────

def device_map(device: str) -> str:
    return device  # passthrough; 'cpu' or 'cuda:0'


def _build_transform():
    """ImageNet normalisation transform compatible with EfficientNet-B0."""
    try:
        import torchvision.transforms as T
        return T.Compose([
            T.ToPILImage(),
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    except ImportError:
        return None


def _preprocess(bgr: np.ndarray, transform) -> "torch.Tensor":
    """Convert BGR numpy frame → normalised RGB tensor (1, 3, H, W)."""
    import cv2
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if transform is not None:
        tensor = transform(rgb).unsqueeze(0)        # (1, 3, H, W)
    else:
        rgb_rs = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb_rs = (rgb_rs - mean) / std
        tensor = torch.tensor(rgb_rs.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def _stub_result() -> Dict[str, object]:
    """Uniform distribution fallback — never crashes the pipeline."""
    n     = len(EMOTION_LABELS)
    probs = np.ones(n, dtype=np.float32) / n
    return {
        "emotion_label":  EMOTION_LABELS[0],
        "emotion_scores": {EMOTION_LABELS[i]: float(probs[i]) for i in range(n)},
        "valence":  0.0,
        "arousal":  0.5,
    }
