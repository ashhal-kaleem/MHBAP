"""
Regression test for SessionRunner diagnostic logging.
"""
import asyncio
import unittest
from unittest.mock import MagicMock, patch
import uuid
from ml.SessionRunner import SessionRunner
from ml.fusion.Predictor import PredictionResult

class DummyWriter:
    async def write(self, *args, **kwargs):
        pass

class TestSessionRunnerLogging(unittest.IsolatedAsyncioTestCase):
    async def test_diagnostic_logging_no_crash_empty_scores(self):
        """
        Verify that _tick does not crash with an UnboundLocalError
        when diagnostic logging fires, even if emotion_scores is empty.
        """
        runner = SessionRunner(session_id=uuid.uuid4(), fps=15)
        
        # Mock components to avoid heavy setup
        runner._cam = MagicMock()
        runner._cam.get_frame.return_value = None
        runner._face = MagicMock()
        runner._face.process.return_value = {}
        runner._gaze = MagicMock()
        runner._gaze.process.return_value = {}
        runner._pose = MagicMock()
        runner._pose.process.return_value = {}
        runner._voice = MagicMock()
        runner._voice.process.return_value = {}
        runner._hci_pipe = MagicMock()
        runner._hci_pipe.process.return_value = {}
        
        runner._mic = MagicMock()
        runner._mic.get_chunk.return_value = None
        runner._hci = MagicMock()
        runner._hci.drain.return_value = ([], [])
        
        runner._writer = DummyWriter()
        
        # Mock Predictor to return a PredictionResult with empty emotion_scores
        pred_mock = MagicMock()
        pred_mock.predict.return_value = PredictionResult(
            emotion="neutral",
            emotion_scores={},  # Empty scores
            stress=0.5,
            engagement=0.5,
            attention=0.5,
            fatigue=0.5,
            feature_vector=None
        )
        runner._predictor = pred_mock
        
        # We need to mock _bus_publish to avoid actually connecting to Redis
        with patch("ml.SessionRunner._bus_publish") as mock_publish:
            # First tick: self._tick_n = 1 (<= 30) -> triggers the diagnostic log
            try:
                await runner._tick()
            except Exception as e:
                self.fail(f"_tick() raised an exception: {e}")

    async def test_diagnostic_logging_no_crash_with_scores(self):
        """
        Verify that _tick does not crash with an UnboundLocalError
        when diagnostic logging fires with valid emotion_scores.
        """
        runner = SessionRunner(session_id=uuid.uuid4(), fps=15)
        
        runner._cam = MagicMock()
        runner._cam.get_frame.return_value = None
        runner._face = MagicMock()
        runner._face.process.return_value = {}
        runner._gaze = MagicMock()
        runner._gaze.process.return_value = {}
        runner._pose = MagicMock()
        runner._pose.process.return_value = {}
        runner._voice = MagicMock()
        runner._voice.process.return_value = {}
        runner._hci_pipe = MagicMock()
        runner._hci_pipe.process.return_value = {}
        
        runner._mic = MagicMock()
        runner._mic.get_chunk.return_value = None
        runner._hci = MagicMock()
        runner._hci.drain.return_value = ([], [])
        
        runner._writer = DummyWriter()
        
        pred_mock = MagicMock()
        pred_mock.predict.return_value = PredictionResult(
            emotion="happy",
            emotion_scores={"happy": 0.9, "sad": 0.1},
            stress=0.5,
            engagement=0.5,
            attention=0.5,
            fatigue=0.5,
            feature_vector=None
        )
        runner._predictor = pred_mock
        
        with patch("ml.SessionRunner._bus_publish") as mock_publish:
            try:
                await runner._tick()
            except Exception as e:
                self.fail(f"_tick() raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
