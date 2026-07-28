"""
Session endpoints.
POST   /api/v1/sessions/            create a new recording session
GET    /api/v1/sessions/{id}        fetch one session
GET    /api/v1/sessions/user/{uid}  list a user's sessions
PATCH  /api/v1/sessions/{id}        update status / end session
POST   /api/v1/sessions/{id}/end    convenience: mark completed now
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import get_db
from backend.app.schemas.session import SessionCreate, SessionRead, SessionUpdate
from backend.app.services import session_service

router = APIRouter()


@router.post("/", response_model=SessionRead, status_code=201)
async def create_session(data: SessionCreate, db: AsyncSession = Depends(get_db)) -> SessionRead:
    session = await session_service.create_session(db, data)
    return SessionRead.model_validate(session)


@router.get("/{session_id}", response_model=SessionRead)
async def read_session(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> SessionRead:
    session = await session_service.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionRead.model_validate(session)


@router.get("/user/{user_id}", response_model=list[SessionRead])
async def list_user_sessions(
    user_id: uuid.UUID, limit: int = 50, db: AsyncSession = Depends(get_db)
) -> list[SessionRead]:
    sessions = await session_service.list_sessions_for_user(db, user_id, limit)
    return [SessionRead.model_validate(s) for s in sessions]


@router.patch("/{session_id}", response_model=SessionRead)
async def patch_session(
    session_id: uuid.UUID, data: SessionUpdate, db: AsyncSession = Depends(get_db)
) -> SessionRead:
    session = await session_service.update_session(db, session_id, data)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionRead.model_validate(session)


@router.post("/{session_id}/end", response_model=SessionRead)
async def end_session(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> SessionRead:
    session = await session_service.end_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionRead.model_validate(session)
