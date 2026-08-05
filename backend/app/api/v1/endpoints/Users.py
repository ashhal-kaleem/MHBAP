"""
User endpoints.
POST /api/v1/users/       create a participant account (unauthenticated — registration path)
GET  /api/v1/users/{id}   fetch own profile (requires auth)
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.Dependencies import get_current_user
from app.db.Session import get_db
from app.schemas.User import UserCreate, UserRead
from app.services import user_service

router = APIRouter()


@router.post("/", response_model=UserRead, status_code=201)
async def create_user(data: UserCreate, db: AsyncSession = Depends(get_db)) -> UserRead:
    """
    Create a participant account.  No auth required (this IS the registration path
    for guest/participant flows).  Role is always server-assigned to 'participant'.
    """
    try:
        user = await user_service.create_user(db, data)
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="A user with this email or username already exists.",
        )
    return UserRead.model_validate(user)


@router.get("/{user_id}", response_model=UserRead)
async def read_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user),
) -> UserRead:
    """Fetch own profile. Users can only read their own data."""
    if str(user_id) != current_user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this resource")
    user = await user_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserRead.model_validate(user)

