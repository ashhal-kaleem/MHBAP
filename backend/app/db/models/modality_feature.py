"""
ModalityFeature — one row per (session, modality, timestamp).

TimescaleDB hypertable: partitioned by `time` in the migration.
The primary key MUST include the partitioning column, so it is
a composite (id, time) key rather than id alone.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class ModalityFeature(Base):
    __tablename__ = "modality_features"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4)
    time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), primary_key=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("sessions.id"), primary_key=True, index=True
    )
    modality: Mapped[str] = mapped_column(String(32), primary_key=True)
    # one of: face, gaze, head_pose, body_pose, voice, hci

    feature_vector: Mapped[dict] = mapped_column(JSONB)
    quality: Mapped[float] = mapped_column(Float, default=1.0)  # 0=missing/occluded

    session: Mapped["Session"] = relationship(back_populates="features")

    __table_args__ = (
        Index("ix_modality_features_session_time", "session_id", "time"),
    )
