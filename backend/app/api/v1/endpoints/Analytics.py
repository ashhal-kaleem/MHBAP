"""
analytics.py — Cross-session analytics endpoints (Phase 10).

GET  /api/v1/Analytics/User/{user_id}            → UserAnalytics JSON
GET  /api/v1/Analytics/User/{user_id}/export/csv → full prediction dump CSV
GET  /api/v1/Sessions/{session_id}/export/json   → single-session JSON export
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.Dependencies import get_current_user
from app.db.Session import get_db
from app.schemas.Analytics import UserAnalytics
from app.services import AnalyticsService as analytics_service

router = APIRouter()


@router.get("/User/{user_id}", response_model=UserAnalytics)
@router.get("/user/{user_id}", response_model=UserAnalytics, include_in_schema=False)
async def user_analytics(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> UserAnalytics:
    """
    Cross-session behavioural analytics for a user.
    Returns per-session summaries, metric trend lines (stress/engagement/
    attention/fatigue over time), and emotion label distribution.
    """
    if str(user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    return await analytics_service.get_user_analytics(db, user_id)


@router.get("/User/{user_id}/export/csv")
@router.get("/user/{user_id}/export/csv", include_in_schema=False)
async def export_user_csv(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> StreamingResponse:
    """
    Download all predictions across all sessions for a user as a single CSV.
    Streams rows directly from DB — no full dataset loaded into RAM.
    """
    if str(user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    filename = f"mhbap_user_{user_id}_all_sessions.csv"
    return StreamingResponse(
        analytics_service.export_user_csv_stream(db, user_id),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
