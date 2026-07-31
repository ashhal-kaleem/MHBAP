"""
analytics.py — Cross-session analytics endpoints (Phase 10).

GET  /api/v1/analytics/user/{user_id}            → UserAnalytics JSON
GET  /api/v1/analytics/user/{user_id}/export/csv → full prediction dump CSV
GET  /api/v1/sessions/{session_id}/export/json   → single-session JSON export
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.analytics import UserAnalytics
from app.services import analytics_service

router = APIRouter()


@router.get("/user/{user_id}", response_model=UserAnalytics)
async def user_analytics(
    user_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> UserAnalytics:
    """
    Cross-session behavioural analytics for a user.
    Returns per-session summaries, metric trend lines (stress/engagement/
    attention/fatigue over time), and emotion label distribution.
    """
    return await analytics_service.get_user_analytics(db, user_id)


@router.get("/user/{user_id}/export/csv")
async def export_user_csv(
    user_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> StreamingResponse:
    """
    Download all predictions across all sessions for a user as a single CSV.
    Suitable for offline analysis / research publication.
    """
    csv_content = await analytics_service.export_user_csv(db, user_id)
    filename = f"mhbap_user_{user_id}_all_sessions.csv"
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
