"""
Unit tests for ml.xai — NL explainer + SHAP aggregation logic in prediction_service.
"""
from __future__ import annotations
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from ml.xai.nl_explainer import generate_explanation


def _pred(**kwargs):
    defaults = dict(stress=0.5, engagement=0.6, attention=0.55, fatigue=0.3, emotion="neutral")
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestNLExplainer:
    def test_returns_string(self):
        r = generate_explanation(_pred())
        assert isinstance(r, str) and len(r) > 10

    def test_no_shap_returns_single_sentence(self):
        r = generate_explanation(_pred(), shap_weights=None)
        # no second sentence about modality attribution
        assert "primarily driven" not in r

    def test_with_shap_mentions_top_modality(self):
        weights = {"face": 0.60, "gaze": 0.20, "pose": 0.10, "voice": 0.05, "hci": 0.05}
        r = generate_explanation(_pred(), shap_weights=weights, head="stress")
        assert "facial expression" in r
        assert "60%" in r

    def test_secondary_driver_mentioned(self):
        weights = {"face": 0.50, "gaze": 0.30, "voice": 0.20}
        r = generate_explanation(_pred(), shap_weights=weights, head="engagement")
        assert "Secondary" in r

    def test_high_stress_advisory(self):
        weights = {"face": 0.80, "gaze": 0.20}
        r = generate_explanation(_pred(stress=0.90), shap_weights=weights, head="stress")
        assert "break" in r.lower() or "stress" in r.lower()

    def test_high_fatigue_advisory(self):
        weights = {"hci": 0.70, "voice": 0.30}
        r = generate_explanation(_pred(fatigue=0.85), shap_weights=weights, head="fatigue")
        assert "fatigue" in r.lower() or "impaired" in r.lower()

    def test_low_engagement_advisory(self):
        weights = {"gaze": 0.60, "pose": 0.40}
        r = generate_explanation(_pred(engagement=0.20), shap_weights=weights, head="engagement")
        assert "distraction" in r.lower() or "engagement" in r.lower()

    def test_emotion_adjective_in_output(self):
        r = generate_explanation(_pred(emotion="angry"), shap_weights=None)
        assert "agitated" in r

    def test_percentage_display(self):
        weights = {"voice": 0.55, "face": 0.45}
        r = generate_explanation(_pred(), shap_weights=weights, head="attention")
        assert "55%" in r


class TestXAIServiceFunction:
    """Unit-tests for prediction_service.get_xai_summary (mocked DB)."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_predictions(self):
        from app.services.prediction_service import get_xai_summary

        db = AsyncMock()
        # list_predictions_for_session internally does db.execute
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        db.execute.return_value = result_mock

        result = await get_xai_summary(db, uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_summary_with_flat_weights(self):
        from app.services.prediction_service import get_xai_summary

        session_id = uuid4()
        t = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        pred = MagicMock(
            session_id=session_id,
            time=t,
            shap_weights={"face": 0.6, "gaze": 0.2, "pose": 0.1, "voice": 0.05, "hci": 0.05},
        )
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = [pred]
        db = AsyncMock()
        db.execute.return_value = result_mock

        summary = await get_xai_summary(db, session_id)
        assert summary is not None
        assert summary.prediction_count == 1
        assert summary.dominant_modality == "face"
        # All four heads should have weights
        assert set(summary.avg_weights.keys()) >= {"stress", "engagement", "attention", "fatigue"}

    @pytest.mark.asyncio
    async def test_summary_dominant_modality_correct(self):
        from app.services.prediction_service import get_xai_summary

        session_id = uuid4()
        t = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)

        def make_pred(w):
            return MagicMock(session_id=session_id, time=t, shap_weights=w)

        preds = [
            make_pred({"hci": 0.80, "face": 0.20}),
            make_pred({"hci": 0.70, "face": 0.30}),
        ]
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = preds
        db = AsyncMock()
        db.execute.return_value = result_mock

        summary = await get_xai_summary(db, session_id)
        assert summary.dominant_modality == "hci"
