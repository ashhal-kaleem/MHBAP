"""
Prediction service — write path for the inference pipeline (Phase 5+)
and read path for the dashboard (Phase 9) and exports (Phase 10).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.prediction import Prediction
from backend.app.schemas.prediction import PredictionCreate


async def create_prediction(db: AsyncSession, data: PredictionCreate) -> Prediction:
    prediction = Prediction(**data.model_dump())
    db.add(prediction)  # db.add is synchronous in SQLAlchemy 2.x
    await db.commit()
    await db.refresh(prediction)
    return prediction


async def list_predictions_for_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    since: datetime | None = None,
    limit: int = 1000,
) -> list[Prediction]:
    """
    Returns predictions in chronological order. `since` lets the dashboard
    poll incrementally instead of re-fetching the whole session each time.
    """
    stmt = select(Prediction).where(Prediction.session_id == session_id)
    if since is not None:
        stmt = stmt.where(Prediction.time > since)
    stmt = stmt.order_by(Prediction.time.asc()).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def latest_prediction(db: AsyncSession, session_id: uuid.UUID) -> Prediction | None:
    result = await db.execute(
        select(Prediction)
        .where(Prediction.session_id == session_id)
        .order_by(Prediction.time.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()
