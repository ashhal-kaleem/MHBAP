"""
Session endpoints.
POST   /api/v1/Sessions/            create a new recording session
GET    /api/v1/Sessions/{id}        fetch one session
GET    /api/v1/Sessions/User/{uid}  list a user's sessions
PATCH  /api/v1/Sessions/{id}        update status / end session
POST   /api/v1/Sessions/{id}/end    convenience: mark completed now
"""
from __future__ import annotations

import io
import csv
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.Dependencies import get_current_user
from app.db.Session import get_db
from app.schemas.Session import (
    SessionContextUpdate,
    SessionCreate,
    SessionRead,
    SessionStats,
    SessionUpdate,
)
from app.services import session_service

router = APIRouter()


@router.post("/", response_model=SessionRead, status_code=201)
async def create_session(
    data: SessionCreate,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> SessionRead:
    if str(data.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    session = await session_service.create_session(db, data)
    return SessionRead.model_validate(session)


@router.get("/{session_id}", response_model=SessionRead)
async def read_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> SessionRead:
    session = await session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    return SessionRead.model_validate(session)


@router.get("/User/{user_id}", response_model=list[SessionRead])
async def list_user_sessions(
    user_id: uuid.UUID,
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> list[SessionRead]:
    if str(user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    sessions = await session_service.list_sessions_for_user(db, user_id, limit)
    return [SessionRead.model_validate(s) for s in sessions]


@router.patch("/{session_id}", response_model=SessionRead)
async def patch_session(
    session_id: uuid.UUID,
    data: SessionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> SessionRead:
    existing_session = await session_service.get_session(db, session_id)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(existing_session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    session = await session_service.update_session(db, session_id, data)
    return SessionRead.model_validate(session)


@router.post("/{session_id}/end", response_model=SessionRead)
async def end_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> SessionRead:
    existing_session = await session_service.get_session(db, session_id)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(existing_session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    session = await session_service.end_session(db, session_id)
    return SessionRead.model_validate(session)


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> None:
    existing_session = await session_service.get_session(db, session_id)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(existing_session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    await session_service.delete_session(db, session_id)


@router.patch("/{session_id}/context", response_model=SessionRead)
async def update_context(
    session_id: uuid.UUID,
    data: SessionContextUpdate,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> SessionRead:
    existing_session = await session_service.get_session(db, session_id)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(existing_session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    session = await session_service.update_session_context(db, session_id, data)
    return SessionRead.model_validate(session)


@router.get("/{session_id}/stats", response_model=SessionStats)
async def session_stats(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> SessionStats:
    existing_session = await session_service.get_session(db, session_id)
    if existing_session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(existing_session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    stats = await session_service.get_session_stats(db, session_id)
    if stats is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return stats


@router.get("/{session_id}/export/json")
async def export_session_json(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
):
    """Export all predictions for a session as a JSON file."""
    import json as _json
    from app.services.PredictionService import list_predictions_for_session
    from app.schemas.Prediction import PredictionRead

    session = await session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")

    predictions = await list_predictions_for_session(db, session_id)
    payload = {
        "session_id": str(session_id),
        "context": session.context,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "predictions": [
            PredictionRead.model_validate(p).model_dump(mode="json") for p in predictions
        ],
    }
    content = _json.dumps(payload, indent=2, default=str)
    filename = f"mhbap_session_{session_id}.json"
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{session_id}/export/csv")
async def export_session_csv(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> StreamingResponse:
    """Stream predictions for a session as a CSV file for offline analysis."""
    from app.services.PredictionService import list_predictions_for_session  # local import avoids circular

    session = await session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if str(session.user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")

    predictions = await list_predictions_for_session(db, session_id)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["time", "session_id", "emotion_label", "stress", "engagement", "attention", "fatigue", "explanation_text"])
    for p in predictions:
        writer.writerow([
            p.time.isoformat(),
            str(p.session_id),
            p.emotion_label,
            f"{p.stress:.4f}",
            f"{p.engagement:.4f}",
            f"{p.attention:.4f}",
            f"{p.fatigue:.4f}",
            p.explanation_text or "",
        ])
    buf.seek(0)

    filename = f"mhbap_session_{session_id}.csv"
    return StreamingResponse(
        iter([buf.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
