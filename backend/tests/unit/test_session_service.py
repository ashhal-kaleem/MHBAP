"""
Unit tests for session_service — AsyncSession is mocked, so these
run without any database and exercise only our query-building logic.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from backend.app.schemas.session import SessionCreate, SessionUpdate
from backend.app.services import session_service


@pytest.mark.asyncio
async def test_create_session_adds_commits_refreshes() -> None:
    db = AsyncMock()
    db.add = MagicMock()  # db.add is sync in SQLAlchemy 2.x
    data = SessionCreate(user_id=uuid4(), context="lecture")

    result = await session_service.create_session(db, data)

    db.add.assert_called_once()
    db.commit.assert_awaited_once()
    db.refresh.assert_awaited_once()
    assert result.context == "lecture"


@pytest.mark.asyncio
async def test_get_session_returns_none_when_missing() -> None:
    db = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None
    db.execute.return_value = scalar_result

    result = await session_service.get_session(db, uuid4())

    db.execute.assert_awaited_once()
    assert result is None


@pytest.mark.asyncio
async def test_end_session_sets_completed_status() -> None:
    db = AsyncMock()
    existing = MagicMock(id=uuid4(), status="active", ended_at=None)
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = existing
    db.execute.return_value = scalar_result

    result = await session_service.end_session(db, existing.id)

    assert result.status == "completed"
    assert result.ended_at is not None
    db.commit.assert_awaited_once()
