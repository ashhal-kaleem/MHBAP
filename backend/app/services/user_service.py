"""
user_service.py — User CRUD backed by the PostgreSQL users table.

All functions accept an async SQLAlchemy session and return ORM objects
(or None when the record is not found).  Password hashing / verification
is delegated to core.security so the service stays persistence-only.
"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.db.models.user import User
from app.schemas.user import UserCreate


async def create_user(db: AsyncSession, data: UserCreate) -> User:
    """Create a user record (legacy path — role and is_active server-assigned)."""
    user = User(
        username=data.username,
        email=data.email,
        display_name=data.display_name or data.username,
        hashed_password="",  # no password on legacy path
        role="participant",  # always server-assigned, never from caller
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def create_user_with_password(
    db: AsyncSession,
    email: str,
    password: str,
    display_name: str = "",
    role: str = "participant",
) -> User:
    """Register a new user, storing a bcrypt hash of the password."""
    username = email.split("@")[0]
    user = User(
        id=uuid.uuid4(),
        username=username,
        email=email,
        hashed_password=hash_password(password),
        display_name=display_name or username,
        role=role,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def deactivate_user(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    """Soft-delete: flip is_active to False."""
    user = await get_user(db, user_id)
    if user:
        user.is_active = False
        await db.commit()
        await db.refresh(user)
    return user
