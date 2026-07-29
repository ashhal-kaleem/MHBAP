"""Pydantic schemas for the Prediction resource (multi-head output + XAI)."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionCreate(BaseModel):
    session_id: uuid.UUID
    emotion_label: str
    emotion_scores: Dict[str, float]
    stress: float = Field(ge=0.0, le=1.0)
    engagement: float = Field(ge=0.0, le=1.0)
    attention: float = Field(ge=0.0, le=1.0)
    fatigue: float = Field(ge=0.0, le=1.0)
    shap_weights: Dict[str, float] = Field(default_factory=dict)
    explanation_text: str = ""


class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    time: datetime
    emotion_label: str
    emotion_scores: Dict[str, float]
    stress: float
    engagement: float
    attention: float
    fatigue: float
    shap_weights: Dict[str, float]
    explanation_text: str


class ModalityTrend(BaseModel):
    """Time-ordered modality weight for one head — used in XAI timeline charts."""
    time: datetime
    weights: Dict[str, float]     # {modality: normalised_weight}


class XAISummary(BaseModel):
    """
    Session-level XAI aggregation.
    avg_weights: averaged SHAP weights per prediction head.
    trends:      per-head time series of modality weights (for trend charts).
    dominant_modality: highest average contributor across all heads.
    """
    session_id: uuid.UUID
    prediction_count: int
    avg_weights: Dict[str, Dict[str, float]]  # {head: {modality: avg_weight}}
    trends: Dict[str, List[ModalityTrend]]    # {head: [ModalityTrend]}
    dominant_modality: Optional[str]
