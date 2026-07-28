"""Pydantic schemas for the Prediction resource (multi-head output + XAI)."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict

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
