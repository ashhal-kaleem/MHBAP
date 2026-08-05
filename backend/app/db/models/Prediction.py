"""
Prediction — one row per second of inference output per session.
TimescaleDB hypertable, same composite-PK pattern as ModalityFeature.

emotion_label / emotion_scores: categorical (AffectNet-8) + softmax
stress / engagement / attention / fatigue: continuous [0, 1] regression heads
shap_weights: per-modality contribution, e.g. {"voice": 0.41, "face": 0.12, ...}
explanation_text: NL sentence generated from top-k SHAP contributors (Phase 8)
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, JSON, String, UUID, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.Base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4)
    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), primary_key=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id"), primary_key=True, index=True
    )

    emotion_label: Mapped[str] = mapped_column(String(32))
    emotion_scores: Mapped[dict] = mapped_column(JSON)

    stress: Mapped[float] = mapped_column(Float)
    engagement: Mapped[float] = mapped_column(Float)
    attention: Mapped[float] = mapped_column(Float)
    fatigue: Mapped[float] = mapped_column(Float)

    shap_weights: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation_text: Mapped[str] = mapped_column(String(512), default="")

    session: Mapped["Session"] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_session_time", "session_id", "time"),
    )
