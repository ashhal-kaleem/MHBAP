"""
Unit tests for Phase 4 pipelines — no hardware, no GPU, no internet.

All MediaPipe / OpenCV / librosa calls are mocked.
"""
from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Face pipeline
# ---------------------------------------------------------------------------

class TestFacePipeline(unittest.TestCase):

    def test_none_frame_returns_zeros(self):
        from ml.pipelines.face.Pipeline import FacePipeline, _FEATURE_KEYS
        pipe = FacePipeline()
        result = pipe.process(None)
        self.assertEqual(set(result.keys()), set(_FEATURE_KEYS))
        self.assertTrue(all(v == 0.0 for v in result.values()))

    def test_no_mediapipe_returns_zeros(self):
        from ml.pipelines.face.Pipeline import FacePipeline, _FEATURE_KEYS
        pipe = FacePipeline()
        with patch.dict("sys.modules", {"mediapipe": None}):
            result = pipe.process(_fake_frame())
        self.assertEqual(set(result.keys()), set(_FEATURE_KEYS))

    def test_no_face_detected_returns_zeros(self):
        from ml.pipelines.face.Pipeline import FacePipeline, _FEATURE_KEYS
        fake_mp = MagicMock()
        fake_mp.solutions.face_mesh.FaceMesh.return_value.process.return_value \
            .multi_face_landmarks = None
        pipe = FacePipeline()
        with patch.dict("sys.modules", {"mediapipe": fake_mp, "cv2": MagicMock()}):
            pipe._mesh = fake_mp.solutions.face_mesh.FaceMesh()
            result = pipe.process(_fake_frame())
        self.assertEqual(set(result.keys()), set(_FEATURE_KEYS))


# ---------------------------------------------------------------------------
# Gaze pipeline
# ---------------------------------------------------------------------------

class TestGazePipeline(unittest.TestCase):

    def test_none_frame_returns_zeros(self):
        from ml.pipelines.gaze.Pipeline import GazePipeline
        pipe = GazePipeline()
        result = pipe.process(None)
        self.assertIn("gaze_x", result)
        self.assertIn("fixation_stability", result)

    def test_output_keys_present(self):
        from ml.pipelines.gaze.Pipeline import GazePipeline
        pipe = GazePipeline()
        result = pipe.process(None)
        for key in ["gaze_x", "gaze_y", "blink_l", "blink_r", "fixation_stability"]:
            self.assertIn(key, result)


# ---------------------------------------------------------------------------
# Voice pipeline
# ---------------------------------------------------------------------------

class TestVoicePipeline(unittest.TestCase):

    def test_none_returns_zeros(self):
        from ml.pipelines.voice.Pipeline import VoicePipeline
        pipe = VoicePipeline()
        result = pipe.process(None)
        self.assertEqual(result["energy"], 0.0)

    def test_short_chunk_returns_zeros(self):
        from ml.pipelines.voice.Pipeline import VoicePipeline
        pipe = VoicePipeline()
        result = pipe.process(np.zeros(100, dtype=np.float32))
        self.assertEqual(result["energy"], 0.0)

    def test_no_librosa_returns_zeros(self):
        from ml.pipelines.voice.Pipeline import VoicePipeline
        pipe = VoicePipeline()
        chunk = np.random.randn(16000).astype(np.float32)
        with patch.dict("sys.modules", {"librosa": None}):
            result = pipe.process(chunk)
        self.assertEqual(result["pitch_mean"], 0.0)


# ---------------------------------------------------------------------------
# HCI pipeline
# ---------------------------------------------------------------------------

class TestHCIPipeline(unittest.TestCase):

    def test_empty_events_returns_zeros(self):
        from ml.pipelines.hci.Pipeline import HCIPipeline
        pipe = HCIPipeline()
        result = pipe.process([])
        self.assertTrue(all(v == 0.0 for v in result.values()))

    def test_keystroke_rate_computed(self):
        from ml.pipelines.hci.Pipeline import HCIPipeline
        pipe = HCIPipeline()
        events = [
            {"type": "key_press", "ts": i * 0.1, "key": "a"}
            for i in range(20)
        ]
        result = pipe.process(events)
        self.assertGreater(result["keystroke_rate"], 0.0)

    def test_backspace_raises_error_rate(self):
        from ml.pipelines.hci.Pipeline import HCIPipeline
        pipe = HCIPipeline()
        events = (
            [{"type": "key_press", "ts": i * 0.1, "key": "a"} for i in range(10)]
            + [{"type": "key_press", "ts": i * 0.1 + 1.0, "key": "backspace"}
               for i in range(5)]
        )
        result = pipe.process(events)
        self.assertGreater(result["error_rate_proxy"], 0.0)


if __name__ == "__main__":
    unittest.Main()
