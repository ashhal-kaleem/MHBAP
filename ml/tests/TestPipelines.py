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
        # Mock the Tasks API: _mesh.detect(mp_image) → result with face_landmarks=[]
        fake_result = MagicMock()
        fake_result.face_landmarks = []
        fake_landmarker = MagicMock()
        fake_landmarker.detect.return_value = fake_result

        # Also mock mp.Image so cvtColor + Image() don't need real mediapipe
        fake_mp = MagicMock()
        fake_mp.ImageFormat.SRGB = 0
        fake_mp.Image.return_value = MagicMock()

        pipe = FacePipeline()
        pipe._mesh = fake_landmarker
        pipe._mp_available = True
        
        # Assume it had a bbox from a previous frame
        pipe.last_face_bbox = (0.1, 0.1, 0.5, 0.5)

        with patch.dict("sys.modules", {"mediapipe": fake_mp, "cv2": MagicMock()}):
            result = pipe.process(_fake_frame())
            
        self.assertEqual(set(result.keys()), set(_FEATURE_KEYS))
        self.assertTrue(all(v == 0.0 for v in result.values()))
        self.assertIsNone(pipe.last_face_bbox)

    def test_face_detected_sets_bbox(self):
        from ml.pipelines.face.Pipeline import FacePipeline
        
        # Create 468 fake landmarks with known min/max x,y
        class FakeLandmark:
            def __init__(self, x, y, z=0):
                self.x = x
                self.y = y
                self.z = z
        
        lms = [FakeLandmark(0.5, 0.5) for _ in range(468)]
        # Min/max boundaries
        lms[0] = FakeLandmark(0.1, 0.2)
        lms[100] = FakeLandmark(0.9, 0.8)
        
        fake_result = MagicMock()
        fake_result.face_landmarks = [lms]
        
        fake_landmarker = MagicMock()
        fake_landmarker.detect.return_value = fake_result

        fake_mp = MagicMock()
        fake_mp.ImageFormat.SRGB = 0
        fake_mp.Image.return_value = MagicMock()

        pipe = FacePipeline()
        pipe._mesh = fake_landmarker
        pipe._mp_available = True

        with patch.dict("sys.modules", {"mediapipe": fake_mp, "cv2": MagicMock()}):
            result = pipe.process(_fake_frame())

        # Should have found the min/max from our injected fake landmarks
        self.assertIsNotNone(pipe.last_face_bbox)
        self.assertEqual(pipe.last_face_bbox, (0.1, 0.2, 0.9, 0.8))




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
    unittest.main()
