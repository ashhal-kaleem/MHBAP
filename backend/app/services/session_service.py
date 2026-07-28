"""
Session service — CRUD logic for recording sessions.
Kept separate from the API layer so it's independently unit-testable
and reusable from the CLI (seed command, Phase 10) and WebSocket handler.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.session_model import Session
from backend.app.schemas.session import SessionCreate, SessionUpdate


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
    return await update_session(
        db, session_id, SessionUpdate(status="completed", ended_at=datetime.now(timezone.utc))
    )
