"""
Session service — CRUD logic for recording sessions.
Kept separate from the API layer so it's independently unit-testable
and reusable from the CLI (seed command, Phase 10) and WebSocket handler.
"""
from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.prediction import Prediction
from app.db.models.session_model import Session
from app.schemas.session import SessionCreate, SessionContextUpdate, SessionStats, SessionUpdate


async def create_session(db: AsyncSession, data: SessionCreate) -> Session:
    session = Session(**data.model_dump())
    db.add(session)  # db.add is synchronous in SQLAlchemy 2.x
    await db.commit()
    await db.refresh(session)
    return session


async def get_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
    result = await db.execute(select(Session).where(Session.id == session_id))
    return result.scalar_one_or_none()


async def list_sessions_for_user(
    db: AsyncSession, user_id: uuid.UUID, limit: int = 50
) -> list[Session]:
    result = await db.execute(
        select(Session)
        .where(Session.user_id == user_id)
        .order_by(Session.started_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def update_session(
    db: AsyncSession, session_id: uuid.UUID, data: SessionUpdate
) -> Session | None:
    session = await get_session(db, session_id)
    if session is None:
        return None
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(session, field, value)
    await db.commit()
    await db.refresh(session)
    return session


async def end_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
    """Convenience wrapper — marks a session completed with a server timestamp."""
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if session is None:
        return None
    session.status = "completed"
    session.ended_at = datetime.now(timezone.utc)
    await db.commit()
    return session


async def delete_session(db: AsyncSession, session_id: uuid.UUID) -> bool:
    """Hard-delete a session (cascades to predictions via FK).  Returns True if deleted."""
    session = await get_session(db, session_id)
    if session is None:
        return False
    await db.delete(session)
    await db.commit()
    return True


async def update_session_context(
    db: AsyncSession, session_id: uuid.UUID, data: SessionContextUpdate
) -> Session | None:
    """Update the human-readable context label only."""
    session = await get_session(db, session_id)
    if session is None:
        return None
    session.context = data.context
    await db.commit()
    await db.refresh(session)
    return session


async def get_session_stats(db: AsyncSession, session_id: uuid.UUID) -> SessionStats | None:
    """Return aggregated analytics for a session.  Returns None if session not found."""
    session = await get_session(db, session_id)
    if session is None:
        return None

    # Aggregate numerics in the DB — one round-trip
    agg = await db.execute(
        select(
            func.count().label("n"),
            func.avg(Prediction.stress).label("stress"),
            func.avg(Prediction.engagement).label("engagement"),
            func.avg(Prediction.attention).label("attention"),
            func.avg(Prediction.fatigue).label("fatigue"),
        ).where(Prediction.session_id == session_id)
    )
    row = agg.one()

    # Dominant emotion: fetch labels then tally in Python (avoids dialect-specific mode())
    labels_res = await db.execute(
        select(Prediction.emotion_label).where(Prediction.session_id == session_id)
    )
    labels = [r[0] for r in labels_res.all()]
    dominant = Counter(labels).most_common(1)[0][0] if labels else None

    duration: float | None = None
    if session.ended_at and session.started_at:
        duration = (session.ended_at - session.started_at).total_seconds()

    return SessionStats(
        session_id=session_id,
        prediction_count=row.n or 0,
        duration_seconds=duration,
        avg_stress=float(row.stress) if row.stress is not None else None,
        avg_engagement=float(row.engagement) if row.engagement is not None else None,
        avg_attention=float(row.attention) if row.attention is not None else None,
        avg_fatigue=float(row.fatigue) if row.fatigue is not None else None,
        dominant_emotion=dominant,
    )
