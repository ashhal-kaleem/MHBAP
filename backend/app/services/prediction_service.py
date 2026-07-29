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


async def get_xai_summary(db: AsyncSession, session_id: uuid.UUID):
    """
    Compute session-level XAI summary from stored shap_weights.
    Returns XAISummary or None if session has no predictions.
    """
    from collections import defaultdict
    from backend.app.schemas.prediction import ModalityTrend, XAISummary

    predictions = await list_predictions_for_session(db, session_id)
    if not predictions:
        return None

    HEADS = ["stress", "engagement", "attention", "fatigue"]
    # Accumulate per-head weights
    acc: dict = defaultdict(lambda: defaultdict(list))   # head -> modality -> [weights]
    trends: dict = defaultdict(list)                     # head -> [ModalityTrend]

    for pred in predictions:
        weights = pred.shap_weights or {}
        # shap_weights may be flat {modality: w} or nested {head: {modality: w}}
        # Try nested first
        if weights and isinstance(next(iter(weights.values())), dict):
            for head in HEADS:
                hw = weights.get(head, {})
                for mod, w in hw.items():
                    acc[head][mod].append(w)
                trends[head].append(ModalityTrend(time=pred.time, weights=hw))
        else:
            # Flat — apply to all heads
            for head in HEADS:
                for mod, w in weights.items():
                    acc[head][mod].append(w)
                trends[head].append(ModalityTrend(time=pred.time, weights=weights))

    avg_weights: dict = {}
    for head in HEADS:
        modalities = acc[head]
        avg_weights[head] = {
            mod: round(sum(vs) / len(vs), 4) for mod, vs in modalities.items()
        } if modalities else {}

    # Dominant modality = highest avg weight across all heads combined
    combined: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    for head_weights in avg_weights.values():
        for mod, w in head_weights.items():
            combined[mod] += w
            counts[mod] += 1
    dominant = max(combined, key=combined.get) if combined else None

    return XAISummary(
        session_id=session_id,
        prediction_count=len(predictions),
        avg_weights=avg_weights,
        trends={h: trends[h] for h in HEADS},
        dominant_modality=dominant,
    )
