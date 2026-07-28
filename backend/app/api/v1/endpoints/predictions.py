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

from backend.app.db.session import get_db
from backend.app.schemas.prediction import PredictionCreate, PredictionRead
from backend.app.services import prediction_service

router = APIRouter()


@router.post("/", response_model=PredictionRead, status_code=201)
async def create_prediction(
    data: PredictionCreate, db: AsyncSession = Depends(get_db)
) -> PredictionRead:
    prediction = await prediction_service.create_prediction(db, data)
    return PredictionRead.model_validate(prediction)


@router.get("/session/{session_id}", response_model=list[PredictionRead])
async def list_session_predictions(
    session_id: uuid.UUID,
    since: Optional[datetime] = None,
    limit: int = 1000,
    db: AsyncSession = Depends(get_db),
) -> list[PredictionRead]:
    predictions = await prediction_service.list_predictions_for_session(
        db, session_id, since, limit
    )
    return [PredictionRead.model_validate(p) for p in predictions]


@router.get("/session/{session_id}/latest", response_model=PredictionRead)
async def latest_session_prediction(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> PredictionRead:
    prediction = await prediction_service.latest_prediction(db, session_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="No predictions yet for this session")
    return PredictionRead.model_validate(prediction)
