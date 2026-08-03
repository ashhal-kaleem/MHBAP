"""
Prediction endpoints.
POST /api/v1/predictions/                 write one prediction row (inference service)
GET  /api/v1/predictions/session/{id}      list predictions for a session (dashboard poll)
GET  /api/v1/predictions/session/{id}/latest  most recent prediction only
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.session import get_db
from app.schemas.prediction import PredictionCreate, PredictionRead, XAISummary
from app.services import prediction_service, session_service

router = APIRouter()


@router.post("/", response_model=PredictionRead, status_code=201)
async def create_prediction(
    data: PredictionCreate,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> PredictionRead:
    session = await session_service.get_session(db, data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    prediction = await prediction_service.create_prediction(db, data)
    return PredictionRead.model_validate(prediction)


@router.get("/session/{session_id}", response_model=list[PredictionRead])
async def list_session_predictions(
    session_id: uuid.UUID,
    since: Optional[datetime] = None,
    limit: int = 1000,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> list[PredictionRead]:
    session = await session_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    predictions = await prediction_service.list_predictions_for_session(
        db, session_id, since, limit
    )
    return [PredictionRead.model_validate(p) for p in predictions]


@router.get("/session/{session_id}/latest", response_model=PredictionRead)
async def latest_session_prediction(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> PredictionRead:
    session = await session_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    prediction = await prediction_service.latest_prediction(db, session_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="No predictions yet for this session")
    return PredictionRead.model_validate(prediction)


@router.get("/session/{session_id}/xai", response_model=XAISummary)
async def session_xai_summary(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> XAISummary:
    """
    Aggregate SHAP weights across all predictions for a session.
    Returns per-head average modality contributions + time-series trends.
    """
    session = await session_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    summary = await prediction_service.get_xai_summary(db, session_id)
    if summary is None:
        return XAISummary(
            session_id=session_id,
            prediction_count=0,
            avg_weights={},
            trends={"stress": [], "engagement": [], "attention": [], "fatigue": []},
            dominant_modality=None,
        )
    return summary
