"""
Abstract base class for all MHBAP modality pipelines.

Every pipeline must implement:
  - extract(frame_or_data) → ModalityFeatures
  - warm_up()              → load model weights
  - teardown()             → release resources

This interface ensures:
  1. TCMT fusion can treat all pipelines uniformly.
  2. Missing-modality masking: pipeline.available → bool.
  3. Ablation studies can swap pipelines without touching fusion code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ModalityFeatures:
    """
    Standardised output of any modality pipeline.

    Attributes
    ----------
    modality : str
        Identifier (e.g. "face", "gaze", "hci").
    features : np.ndarray
        Raw feature vector, shape (D,).
    timestamp : float
        Unix timestamp of the source data.
    confidence : float
        Model confidence in [0, 1]. Used as fusion weight.
    available : bool
        False if source data was unavailable (camera blocked, mic off).
    meta : dict
        Optional extra data (e.g. bounding box, AU intensities).
    """
    modality: str
    features: np.ndarray
    timestamp: float
    confidence: float = 1.0
    available: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "features": self.features.tolist(),
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "available": self.available,
            "meta": self.meta,
        }


class BasePipeline(ABC):
    """Abstract modality pipeline."""

    name: str = "base"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._loaded = False

    @abstractmethod
    def warm_up(self) -> None:
        """Load model weights into memory. Called once at startup."""

    @abstractmethod
    def extract(self, data: Any) -> ModalityFeatures:
        """
        Extract features from raw sensor data.

        Parameters
        ----------
        data : Any
            Frame (np.ndarray HxWxC), audio chunk (np.ndarray), or dict.

        Returns
        -------
        ModalityFeatures
        """

    def teardown(self) -> None:
        """Release GPU memory and file handles. Override if needed."""
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}, loaded={self._loaded})"
