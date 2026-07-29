"""
analytics.py — Pydantic schemas for Phase 10 cross-session analytics.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class MetricTimePoint(BaseModel):
    """One (time, value) point in a metric trend line."""
    time: datetime
    value: float


class MetricTimeSeries(BaseModel):
    """All trend points for a single metric across sessions."""
    metric: str
    points: List[MetricTimePoint]


class EmotionBreakdown(BaseModel):
    """Aggregate emotion label distribution across all predictions."""
    counts: Dict[str, int]    # {"neutral": 42, "happy": 18, ...}
    total: int


class SessionSummary(BaseModel):
    """Per-session summary row in the analytics response."""
    session_id: uuid.UUID
    context: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    duration_seconds: Optional[float]
    prediction_count: int
    avg_stress: Optional[float]
    avg_engagement: Optional[float]
    avg_attention: Optional[float]
    avg_fatigue: Optional[float]
    dominant_emotion: Optional[str]


class UserAnalytics(BaseModel):
    """
    Top-level cross-session analytics payload for one user.
    Returned by GET /api/v1/analytics/user/{user_id}
    """
    user_id: uuid.UUID
    session_count: int
    total_duration_seconds: float
    sessions: List[SessionSummary]
    metric_trends: Dict[str, MetricTimeSeries]  # keyed by metric name
    emotion_breakdown: EmotionBreakdown
