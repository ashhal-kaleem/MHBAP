"""
test_session_runner_inference.py
End-to-end inference path tests covering the Phase A production fix:
  session_runner._tick() must forward bgr_frame to BehaviourPredictor.predict()
  so that when EmotionRecognizer.is_ready == True the emotion_source is "pretrained".

These tests do NOT require a live camera, microphone, database, or Redis.
All capture/pipeline dependencies are mocked.  The EmotionRecognizer is either
tested against the real checkpoint (when present) or with is_ready=False (stub
mode).  Both modes validate the correct routing logic.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

CHECKPOINT = Path("ml/models/weights/enet_b0_8_best_afew.pt")
HAS_CHECKPOINT = CHECKPOINT.exists()

_FEATURE_DICTS = {
    "face":  {},
    "gaze":  {},
    "pose":  {},
    "voice": {},
    "hci":   {},
}

_FAKE_FRAME = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


# ── BehaviourPredictor routing ────────────────────────────────────────────────

class TestBehaviourPredictorFrameRouting:
    """Verify the predict() signature accepts bgr_frame and routes correctly."""

    def setup_method(self):
        from ml.fusion.predictor import BehaviourPredictor
        self.predictor = BehaviourPredictor(device="cpu")

    def test_predict_without_frame_uses_tcmt(self):
        result = self.predictor.predict(_FEATURE_DICTS, bgr_frame=None)
        assert result.emotion_source == "tcmt", (
            "Without a frame the fallback must be 'tcmt', got: " + result.emotion_source
        )

    def test_predict_none_bgr_uses_tcmt(self):
        """Passing bgr_frame=None explicitly must still fall through to TCMT."""
        result = self.predictor.predict(_FEATURE_DICTS, bgr_frame=None)
        assert result.emotion_source == "tcmt"

    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint absent — pretrained path not reachable")
    def test_predict_with_frame_uses_pretrained(self):
        result = self.predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        assert result.emotion_source == "pretrained", (
            "With a frame and a loaded checkpoint, emotion_source must be 'pretrained', "
            "got: " + result.emotion_source
        )

    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint absent")
    def test_pretrained_scores_sum_to_one(self):
        result = self.predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        total = sum(result.emotion_scores.values())
        assert abs(total - 1.0) < 1e-3

    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint absent")
    def test_pretrained_valence_in_range(self):
        result = self.predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        assert -1.0 <= result.valence <= 1.0

    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint absent")
    def test_pretrained_arousal_in_range(self):
        result = self.predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        assert 0.0 <= result.arousal <= 1.0

    def test_result_has_emotion_source_field(self):
        result = self.predictor.predict(_FEATURE_DICTS)
        assert hasattr(result, "emotion_source")
        assert result.emotion_source in ("tcmt", "pretrained")

    def test_continuous_indices_always_from_tcmt(self):
        """stress/engagement/attention/fatigue come from TCMT regardless of emotion path."""
        result = self.predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME if HAS_CHECKPOINT else None)
        assert 0.0 <= result.stress <= 1.0
        assert 0.0 <= result.engagement <= 1.0
        assert 0.0 <= result.attention <= 1.0
        assert 0.0 <= result.fatigue <= 1.0


# ── Mock-based routing test (no checkpoint needed) ────────────────────────────

class TestFrameRoutingWithMockedRecognizer:
    """
    Injects a mock EmotionRecognizer with is_ready=True so the pretrained path
    is exercised even in CI where the checkpoint is absent.
    """

    def _make_predictor_with_mock_recognizer(self):
        from ml.fusion.predictor import BehaviourPredictor
        predictor = BehaviourPredictor(device="cpu")

        mock_rec = MagicMock()
        type(mock_rec).is_ready = PropertyMock(return_value=True)
        mock_rec.predict.return_value = {
            "emotion_label":  "happy",
            "emotion_scores": {
                "neutral": 0.05, "happy": 0.80, "sad": 0.03, "surprise": 0.04,
                "fear": 0.02, "disgust": 0.02, "anger": 0.02, "contempt": 0.02,
            },
            "valence":  0.75,
            "arousal":  0.60,
        }
        predictor._emotion_rec = mock_rec
        return predictor, mock_rec

    def test_frame_passed_triggers_pretrained_path(self):
        predictor, mock_rec = self._make_predictor_with_mock_recognizer()
        result = predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        assert result.emotion_source == "pretrained"
        mock_rec.predict.assert_called_once()

    def test_recognizer_receives_the_exact_frame(self):
        predictor, mock_rec = self._make_predictor_with_mock_recognizer()
        predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        call_args = mock_rec.predict.call_args
        passed_frame = call_args[0][0] if call_args[0] else call_args[1].get("bgr_frame")
        assert np.array_equal(passed_frame, _FAKE_FRAME)

    def test_no_frame_does_not_call_recognizer(self):
        predictor, mock_rec = self._make_predictor_with_mock_recognizer()
        result = predictor.predict(_FEATURE_DICTS, bgr_frame=None)
        mock_rec.predict.assert_not_called()
        assert result.emotion_source == "tcmt"

    def test_pretrained_label_propagated_to_result(self):
        predictor, _ = self._make_predictor_with_mock_recognizer()
        result = predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        assert result.emotion == "happy"

    def test_pretrained_scores_propagated(self):
        predictor, _ = self._make_predictor_with_mock_recognizer()
        result = predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        assert result.emotion_scores["happy"] == pytest.approx(0.80)

    def test_pretrained_valence_propagated(self):
        predictor, _ = self._make_predictor_with_mock_recognizer()
        result = predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        assert result.valence == pytest.approx(0.75)

    def test_pretrained_arousal_propagated(self):
        predictor, _ = self._make_predictor_with_mock_recognizer()
        result = predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        assert result.arousal == pytest.approx(0.60)

    def test_continuous_indices_still_from_tcmt(self):
        """Even in pretrained emotion path, stress etc. come from TCMT heads."""
        predictor, _ = self._make_predictor_with_mock_recognizer()
        result = predictor.predict(_FEATURE_DICTS, bgr_frame=_FAKE_FRAME)
        assert 0.0 <= result.stress <= 1.0
        assert 0.0 <= result.engagement <= 1.0


# ── SessionRunner tick integration ────────────────────────────────────────────

class TestSessionRunnerPassesFrameToPredictor:
    """
    Verify that SessionRunner._tick() forwards the camera frame to
    BehaviourPredictor.predict() as bgr_frame.

    All heavy dependencies (camera, mic, HCI, pipelines, DB writer, bus) are
    mocked so this test runs with zero infrastructure.

    SessionRunner is imported under a comprehensive sys.modules patch to
    avoid SQLAlchemy MetaData conflicts that occur when the test session has
    already loaded the FastAPI app (which registers the same ORM tables).
    """

    def _make_runner(self):
        import asyncio
        import sys
        import types

        # Stub out the DB-touching modules so SessionRunner can be imported
        # fresh without hitting SQLAlchemy's MetaData re-registration error.
        _stubs = {}
        for mod_name in list(sys.modules):
            if "data_writer" in mod_name or (
                "app.db" in mod_name and "model" in mod_name
            ):
                _stubs[mod_name] = sys.modules.pop(mod_name)

        # Also provide a lightweight DataWriter stub if not already patched
        fake_dw_mod = types.ModuleType("ml.data_writer")
        class _FakeDataWriter:
            async def start(self): pass
            async def stop(self): pass
            async def write(self, *a, **kw): pass
        fake_dw_mod.DataWriter = _FakeDataWriter
        sys.modules["ml.data_writer"] = fake_dw_mod

        # Pop session_runner so it re-imports cleanly
        sys.modules.pop("ml.session_runner", None)

        from ml.session_runner import SessionRunner

        # Restore everything we yanked
        for k, v in _stubs.items():
            sys.modules[k] = v

        runner = SessionRunner.__new__(SessionRunner)
        runner.session_id = uuid.uuid4()
        runner._fps = 15
        runner._stop = asyncio.Event()

        # Capture mocks
        runner._cam = MagicMock()
        runner._cam.get_frame.return_value = _FAKE_FRAME
        runner._mic = MagicMock()
        runner._mic.get_chunk.return_value = np.zeros(1024, dtype=np.float32)
        runner._hci = MagicMock()
        runner._hci.get_events.return_value = []

        # Pipeline mocks — return empty feature dicts
        for attr in ("_face", "_gaze", "_pose", "_voice", "_hci_pipe"):
            m = MagicMock()
            m.process.return_value = {}
            setattr(runner, attr, m)

        # Writer mock
        runner._writer = MagicMock()
        runner._writer.write = AsyncMock()

        # XAI mocks
        runner._explainer = MagicMock()
        runner._explainer.explain.return_value = {}
        runner.latest_shap = {}
        runner.latest_explanation = ""
        runner.latest_prediction = None

        # BehaviourPredictor mock — we'll assert on it
        mock_predictor = MagicMock()
        from ml.fusion.predictor import PredictionResult
        mock_predictor.predict.return_value = PredictionResult(
            emotion="neutral",
            emotion_scores={"neutral": 1.0},
            stress=0.3, engagement=0.6, attention=0.5, fatigue=0.2,
            emotion_source="pretrained",
        )
        runner._predictor = mock_predictor
        runner._predictor._model = MagicMock()

        return runner, mock_predictor

    @pytest.mark.asyncio
    async def test_tick_passes_frame_as_bgr_frame(self):
        """The core regression test: _tick() must call predict(…, bgr_frame=frame)."""
        with patch("app.core.stream_bus.publish", return_value=None), \
             patch("ml.xai.nl_explainer.generate_explanation", return_value=""):
            runner, mock_predictor = self._make_runner()
            await runner._tick()

        call_kwargs = mock_predictor.predict.call_args
        # bgr_frame should appear as keyword arg
        assert "bgr_frame" in call_kwargs.kwargs, (
            "_tick() did not pass bgr_frame keyword to predict(). "
            "The production bug has been reintroduced."
        )

    @pytest.mark.asyncio
    async def test_tick_passes_correct_frame(self):
        """bgr_frame passed to predict() must be the frame from CameraCapture."""
        with patch("app.core.stream_bus.publish", return_value=None), \
             patch("ml.xai.nl_explainer.generate_explanation", return_value=""):
            runner, mock_predictor = self._make_runner()
            await runner._tick()

        passed_frame = mock_predictor.predict.call_args.kwargs["bgr_frame"]
        assert np.array_equal(passed_frame, _FAKE_FRAME), (
            "bgr_frame passed to predict() does not match the camera frame."
        )

    @pytest.mark.asyncio
    async def test_tick_passes_feature_dicts(self):
        """feature_dicts positional arg must also be present."""
        with patch("app.core.stream_bus.publish", return_value=None), \
             patch("ml.xai.nl_explainer.generate_explanation", return_value=""):
            runner, mock_predictor = self._make_runner()
            await runner._tick()

        args = mock_predictor.predict.call_args
        feature_dicts = args.args[0] if args.args else args.kwargs.get("feature_dicts")
        assert isinstance(feature_dicts, dict)
        assert set(feature_dicts.keys()) == {"face", "gaze", "pose", "voice", "hci"}

    @pytest.mark.asyncio
    async def test_tick_predict_called_once_per_tick(self):
        with patch("app.core.stream_bus.publish", return_value=None), \
             patch("ml.xai.nl_explainer.generate_explanation", return_value=""):
            runner, mock_predictor = self._make_runner()
            await runner._tick()

        mock_predictor.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_stores_latest_prediction(self):
        with patch("app.core.stream_bus.publish", return_value=None), \
             patch("ml.xai.nl_explainer.generate_explanation", return_value=""):
            runner, _ = self._make_runner()
            await runner._tick()

        assert runner.latest_prediction is not None
        assert runner.latest_prediction.emotion_source == "pretrained"
