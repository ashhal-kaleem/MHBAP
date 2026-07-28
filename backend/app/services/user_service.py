"""User service — minimal CRUD. Full auth is out of scope for MHBAP
research platform (Phase 2); sessions are created against existing
user records seeded via the CLI or created here directly."""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.user import User
from backend.app.schemas.user import UserCreate


async def create_user(db: AsyncSession, data: UserCreate) -> User:
    user = User(**data.model_dump())
    db.add(user)  # db.add is synchronous in SQLAlchemy 2.x
    await db.commit()
    await db.refresh(user)
    return user


async def get_user(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
