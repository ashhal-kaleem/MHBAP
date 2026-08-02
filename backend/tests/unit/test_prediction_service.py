"""Unit tests for prediction_service — mocked AsyncSession, no DB needed."""
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.schemas.prediction import PredictionCreate
from app.services import prediction_service


def _sample_prediction_data(**overrides) -> PredictionCreate:
    base = dict(
        session_id=uuid4(),
        emotion_label="neutral",
        emotion_scores={"neutral": 0.7, "focused": 0.3},
        stress=0.2,
        engagement=0.8,
        attention=0.75,
        fatigue=0.1,
        shap_weights={"face": 0.4, "voice": 0.3, "hci": 0.3},
        explanation_text="High engagement driven mainly by face and voice signals.",
    )
    base.update(overrides)
    return PredictionCreate(**base)


@pytest.mark.asyncio
async def test_create_prediction_adds_commits_refreshes() -> None:
    db = AsyncMock()
    db.add = MagicMock()  # db.add is sync in SQLAlchemy 2.x
    data = _sample_prediction_data()

    result = await prediction_service.create_prediction(db, data)

    db.add.assert_called_once()
    db.commit.assert_awaited_once()
    db.refresh.assert_awaited_once()
    assert result.emotion_label == "neutral"
    assert result.stress == 0.2


@pytest.mark.asyncio
async def test_prediction_score_bounds_are_enforced_by_schema() -> None:
    with pytest.raises(ValueError):
        _sample_prediction_data(stress=1.5)  # out of [0, 1] range


@pytest.mark.asyncio
async def test_latest_prediction_returns_none_when_empty() -> None:
    db = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None
    db.execute.return_value = scalar_result

    result = await prediction_service.latest_prediction(db, uuid4())

    assert result is None
