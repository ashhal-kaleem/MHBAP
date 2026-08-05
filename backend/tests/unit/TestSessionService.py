"""
Unit tests for session_service — AsyncSession is mocked, so these
run without any database and exercise only our query-building logic.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.schemas.Session import SessionCreate, SessionContextUpdate, SessionUpdate
from app.services import session_service


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


@pytest.mark.asyncio
async def test_delete_session_returns_true_when_found() -> None:
    db = AsyncMock()
    existing = MagicMock(id=uuid4())
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = existing
    db.execute.return_value = scalar_result
    db.delete = AsyncMock()

    result = await session_service.delete_session(db, existing.id)

    db.delete.assert_awaited_once_with(existing)
    db.commit.assert_awaited_once()
    assert result is True


@pytest.mark.asyncio
async def test_delete_session_returns_false_when_not_found() -> None:
    db = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None
    db.execute.return_value = scalar_result

    result = await session_service.delete_session(db, uuid4())

    assert result is False
    db.delete.assert_not_called()


@pytest.mark.asyncio
async def test_update_session_context() -> None:
    db = AsyncMock()
    session_id = uuid4()
    existing = MagicMock(id=session_id, context="old")
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = existing
    db.execute.return_value = scalar_result

    result = await session_service.update_session_context(
        db, session_id, SessionContextUpdate(context="exam")
    )

    assert existing.context == "exam"
    db.commit.assert_awaited_once()
    db.refresh.assert_awaited_once()
    assert result is existing


@pytest.mark.asyncio
async def test_update_session_context_missing_returns_none() -> None:
    db = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None
    db.execute.return_value = scalar_result

    result = await session_service.update_session_context(
        db, uuid4(), SessionContextUpdate(context="exam")
    )

    assert result is None
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_get_session_stats_returns_none_when_session_missing() -> None:
    db = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None
    db.execute.return_value = scalar_result

    result = await session_service.get_session_stats(db, uuid4())

    assert result is None


@pytest.mark.asyncio
async def test_get_session_stats_zero_predictions() -> None:
    """Stats should return 0-count, None averages when no predictions exist."""
    db = AsyncMock()
    session_id = uuid4()

    session_mock = MagicMock(
        id=session_id,
        started_at=datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2025, 1, 1, 10, 30, tzinfo=timezone.utc),
    )

    # First execute → get_session; second → aggregate; third → labels
    scalar_session = MagicMock()
    scalar_session.scalar_one_or_none.return_value = session_mock

    agg_row = MagicMock(n=0, stress=None, engagement=None, attention=None, fatigue=None)
    agg_result = MagicMock()
    agg_result.one.return_value = agg_row

    labels_result = MagicMock()
    labels_result.all.return_value = []

    db.execute.side_effect = [scalar_session, agg_result, labels_result]

    stats = await session_service.get_session_stats(db, session_id)

    assert stats is not None
    assert stats.prediction_count == 0
    assert stats.dominant_emotion is None
    assert stats.duration_seconds == pytest.approx(1800.0)
