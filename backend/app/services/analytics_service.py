"""
analytics_service.py — cross-session aggregation for Phase 10.

Provides:
  get_user_analytics(db, user_id)  → UserAnalytics
  export_user_csv(db, user_id)     → CSV string (all sessions + predictions)
"""
from __future__ import annotations

import io
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.prediction import Prediction
from app.db.models.session_model import Session
from app.schemas.analytics import (
    EmotionBreakdown,
    MetricTimeSeries,
    MetricTimePoint,
    SessionSummary,
    UserAnalytics,
)


async def get_user_analytics(db: AsyncSession, user_id: uuid.UUID) -> UserAnalytics:
    """
    Compute cross-session analytics for a user.
    - Per-session summary (avg metrics, dominant emotion, duration)
    - Global metric time-series (one point per session, using session start_time)
    - Emotion breakdown across all sessions
    """
    # 1. All sessions for user
    sess_result = await db.execute(
        select(Session)
        .where(Session.user_id == user_id)
        .order_by(Session.started_at.asc())
    )
    sessions = list(sess_result.scalars().all())

    if not sessions:
        return UserAnalytics(
            user_id=user_id,
            session_count=0,
            total_duration_seconds=0.0,
            sessions=[],
            metric_trends={m: MetricTimeSeries(metric=m, points=[])
                           for m in ("stress", "engagement", "attention", "fatigue")},
            emotion_breakdown=EmotionBreakdown(counts={}, total=0),
        )

    session_ids = [s.id for s in sessions]

    # 2. Aggregate per session from predictions table (one query)
    agg_result = await db.execute(
        select(
            Prediction.session_id,
            func.count().label("n"),
            func.avg(Prediction.stress).label("avg_stress"),
            func.avg(Prediction.engagement).label("avg_engagement"),
            func.avg(Prediction.attention).label("avg_attention"),
            func.avg(Prediction.fatigue).label("avg_fatigue"),
        )
        .where(Prediction.session_id.in_(session_ids))
        .group_by(Prediction.session_id)
    )
    agg_by_sid = {row.session_id: row for row in agg_result.all()}

    # 3. Emotion labels per session
    label_result = await db.execute(
        select(Prediction.session_id, Prediction.emotion_label)
        .where(Prediction.session_id.in_(session_ids))
    )
    labels_by_sid: dict = defaultdict(list)
    all_labels: list = []
    for row in label_result.all():
        labels_by_sid[row.session_id].append(row.emotion_label)
        all_labels.append(row.emotion_label)

    # 4. Build per-session summaries + time-series points
    metric_points: dict = defaultdict(list)
    session_summaries: list[SessionSummary] = []
    total_dur = 0.0

    for s in sessions:
        agg = agg_by_sid.get(s.id)
        labels = labels_by_sid.get(s.id, [])
        dominant = Counter(labels).most_common(1)[0][0] if labels else None
        dur = (s.ended_at - s.started_at).total_seconds() if s.ended_at and s.started_at else None
        if dur:
            total_dur += dur

        session_summaries.append(SessionSummary(
            session_id=s.id,
            context=s.context,
            started_at=s.started_at,
            ended_at=s.ended_at,
            duration_seconds=dur,
            prediction_count=agg.n if agg else 0,
            avg_stress=float(agg.avg_stress) if agg and agg.avg_stress is not None else None,
            avg_engagement=float(agg.avg_engagement) if agg and agg.avg_engagement is not None else None,
            avg_attention=float(agg.avg_attention) if agg and agg.avg_attention is not None else None,
            avg_fatigue=float(agg.avg_fatigue) if agg and agg.avg_fatigue is not None else None,
            dominant_emotion=dominant,
        ))

        if agg:
            ts = s.started_at
            for metric, val in (
                ("stress",     agg.avg_stress),
                ("engagement", agg.avg_engagement),
                ("attention",  agg.avg_attention),
                ("fatigue",    agg.avg_fatigue),
            ):
                if val is not None:
                    metric_points[metric].append(MetricTimePoint(time=ts, value=float(val)))

    metric_trends = {
        m: MetricTimeSeries(metric=m, points=metric_points.get(m, []))
        for m in ("stress", "engagement", "attention", "fatigue")
    }

    emotion_counts = dict(Counter(all_labels))

    return UserAnalytics(
        user_id=user_id,
        session_count=len(sessions),
        total_duration_seconds=total_dur,
        sessions=session_summaries,
        metric_trends=metric_trends,
        emotion_breakdown=EmotionBreakdown(counts=emotion_counts, total=len(all_labels)),
    )


async def export_user_csv_stream(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> "AsyncGenerator[str, None]":
    """
    Async generator that yields CSV lines one at a time.
    Uses yield_per(500) to avoid loading all predictions into RAM.
    Suitable for large research sessions (thousands of predictions).
    """
    from typing import AsyncGenerator

    sess_result = await db.execute(
        select(Session).where(Session.user_id == user_id).order_by(Session.started_at.asc())
    )
    sessions = list(sess_result.scalars().all())
    context_map = {s.id: (s.context or "") for s in sessions}
    started_map = {s.id: s.started_at for s in sessions}
    session_ids = [s.id for s in sessions]

    # Header
    yield ("session_id,context,session_started_at,time,emotion_label,"
           "stress,engagement,attention,fatigue,explanation_text\n")

    if not session_ids:
        return

    # Stream predictions in batches of 500 rows — never loads all into RAM
    stmt = (
        select(Prediction)
        .where(Prediction.session_id.in_(session_ids))
        .order_by(Prediction.session_id, Prediction.time)
        .execution_options(yield_per=500)
    )
    result = await db.stream(stmt)
    async for partition in result.partitions(500):
        for row in partition:
            p = row[0]
            ctx = context_map.get(p.session_id, "").replace('"', "'")
            exp = (p.explanation_text or "").replace('"', "'")
            started = started_map.get(p.session_id, "")
            yield (
                f'{p.session_id},"{ctx}",{started},{p.time.isoformat()},'
                f'{p.emotion_label},{p.stress:.4f},{p.engagement:.4f},'
                f'{p.attention:.4f},{p.fatigue:.4f},"{exp}"\n'
            )


# Keep legacy name for backward compatibility — delegates to streaming version
async def export_user_csv(db: AsyncSession, user_id: uuid.UUID) -> str:
    """Legacy: returns full CSV as string. Use export_user_csv_stream for large exports."""
    lines = []
    async for line in export_user_csv_stream(db, user_id):
        lines.append(line)
    return "".join(lines)
