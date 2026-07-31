"""
User endpoints — minimal, no auth (Phase 2 scope).
POST /api/v1/users/       create a user (researcher/educator/participant)
GET  /api/v1/users/{id}   fetch a user
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.user import UserCreate, UserRead
from app.services import user_service

router = APIRouter()


@router.post("/", response_model=UserRead, status_code=201)
async def create_user(data: UserCreate, db: AsyncSession = Depends(get_db)) -> UserRead:
    user = await user_service.create_user(db, data)
    return UserRead.model_validate(user)


@router.get("/{user_id}", response_model=UserRead)
async def read_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> UserRead:
    user = await user_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserRead.model_validate(user)
