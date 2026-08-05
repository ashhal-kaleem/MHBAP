"""
Session model — one row per recording session (e.g. one lecture,
one coding task). Named session_model.py to avoid colliding with
db/session.py (the SQLAlchemy engine module).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.Base import Base


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    context: Mapped[str] = mapped_column(String(64), default="unspecified")
    status: Mapped[str] = mapped_column(String(16), default="active")

    consent_recording: Mapped[bool] = mapped_column(default=False)
    faces_blurred: Mapped[bool] = mapped_column(default=False)
    session_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    ended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user: Mapped["User"] = relationship(back_populates="sessions")
    features: Mapped[list["ModalityFeature"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
