"""Pydantic schemas for the Session resource."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SessionCreate(BaseModel):
    user_id: uuid.UUID
    context: str = "unspecified"
    consent_recording: bool = False
    faces_blurred: bool = False
    session_metadata: Optional[dict] = None


class SessionUpdate(BaseModel):
    status: Optional[str] = None
    ended_at: Optional[datetime] = None


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    context: str
    status: str
    consent_recording: bool
    faces_blurred: bool
    session_metadata: Optional[dict]
    started_at: datetime
    ended_at: Optional[datetime]


class SessionStats(BaseModel):
    """Aggregated stats for a single session — returned by GET /sessions/{id}/stats."""
    session_id: uuid.UUID
    prediction_count: int
    duration_seconds: Optional[float]   # None if session still active or ended_at missing
    avg_stress: Optional[float]
    avg_engagement: Optional[float]
    avg_attention: Optional[float]
    avg_fatigue: Optional[float]
    dominant_emotion: Optional[str]


class SessionContextUpdate(BaseModel):
    """Payload for PATCH /sessions/{id}/context — updates the human-readable label."""
    context: str
