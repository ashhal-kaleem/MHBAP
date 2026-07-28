"""Unit tests for user_service — mocked AsyncSession, no DB needed."""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from backend.app.schemas.user import UserCreate
from backend.app.services import user_service


@pytest.mark.asyncio
async def test_create_user_adds_commits_refreshes() -> None:
    db = AsyncMock()
    db.add = MagicMock()  # db.add is sync in SQLAlchemy 2.x
    data = UserCreate(username="testuser", email="test@example.com")

    result = await user_service.create_user(db, data)

    db.add.assert_called_once()
    db.commit.assert_awaited_once()
    db.refresh.assert_awaited_once()
    assert result.username == "testuser"


@pytest.mark.asyncio
async def test_get_user_returns_none_when_missing() -> None:
    db = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None
    db.execute.return_value = scalar_result

    result = await user_service.get_user(db, uuid4())

    db.execute.assert_awaited_once()
    assert result is None


@pytest.mark.asyncio
async def test_get_user_returns_record_when_found() -> None:
    db = AsyncMock()
    uid = uuid4()
    fake_user = MagicMock(id=uid, username="alice", email="alice@example.com")
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = fake_user
    db.execute.return_value = scalar_result

    result = await user_service.get_user(db, uid)

    assert result is fake_user
    assert result.email == "alice@example.com"
