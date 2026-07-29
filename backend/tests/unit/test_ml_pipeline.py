"""
tests/unit/test_ml_pipeline.py
Tests for TCMT fusion model, feature utils, and feature vector layout.
All tests use the numpy stub fallback — no PyTorch required.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── TCMT ─────────────────────────────────────────────────────────────────────

class TestTCMT:
    def setup_method(self):
        from ml.fusion.tcmt import TCMT, EMOTION_CLASSES, FEATURE_DIM
        self.TCMT = TCMT
        self.EMOTION_CLASSES = EMOTION_CLASSES
        self.FEATURE_DIM = FEATURE_DIM

    def _x(self, B=2, T=5):
        return np.random.rand(B, T, self.FEATURE_DIM).astype(np.float32)

    def test_instantiation(self):
        assert self.TCMT() is not None

    def test_output_keys(self):
        out = self.TCMT()(self._x())
        assert set(out.keys()) == {"emotion_logits", "stress", "engagement", "attention", "fatigue"}

    def test_emotion_logits_n_classes(self):
        out = self.TCMT()(self._x(B=3))
        assert np.array(out["emotion_logits"]).shape[-1] == self.EMOTION_CLASSES

    def test_stress_range(self):
        v = np.array(self.TCMT()(self._x(B=4))["stress"]).flatten()
        assert np.all(v >= 0) and np.all(v <= 10)

    def test_engagement_range(self):
        v = np.array(self.TCMT()(self._x())["engagement"]).flatten()
        assert np.all(v >= 0) and np.all(v <= 1)

    def test_attention_range(self):
        v = np.array(self.TCMT()(self._x())["attention"]).flatten()
        assert np.all(v >= 0) and np.all(v <= 1)

    def test_fatigue_range(self):
        v = np.array(self.TCMT()(self._x())["fatigue"]).flatten()
        assert np.all(v >= 0) and np.all(v <= 1)

    def test_single_sample(self):
        out = self.TCMT()(self._x(B=1, T=1))
        assert "emotion_logits" in out

    def test_zero_input_no_nan(self):
        x = np.zeros((2, 4, self.FEATURE_DIM), dtype=np.float32)
        out = self.TCMT()(x)
        for v in out.values():
            assert not np.any(np.isnan(np.array(v)))

    def test_feature_dim_matches_layout(self):
        from ml.fusion.feature_vector import MODALITY_KEYS
        expected = sum(len(v) for v in MODALITY_KEYS.values())
        assert self.FEATURE_DIM == expected


# ── Feature vector layout ────────────────────────────────────────────────────

class TestFeatureVectorLayout:
    def setup_method(self):
        from ml.fusion.feature_vector import MODALITY_KEYS, FEATURE_DIM
        self.MODALITY_KEYS = MODALITY_KEYS
        self.FEATURE_DIM = FEATURE_DIM

    def test_total_dim_consistent(self):
        total = sum(len(v) for v in self.MODALITY_KEYS.values())
        assert total == self.FEATURE_DIM

    def test_modalities_present(self):
        assert set(self.MODALITY_KEYS.keys()) == {"face", "gaze", "pose", "voice", "hci"}

    def test_face_12_dims(self):
        assert len(self.MODALITY_KEYS["face"]) == 12

    def test_gaze_5_dims(self):
        assert len(self.MODALITY_KEYS["gaze"]) == 5

    def test_voice_dims_gte_13(self):
        # 13 MFCCs + pitch + energy + extras
        assert len(self.MODALITY_KEYS["voice"]) >= 13

    def test_hci_10_dims(self):
        assert len(self.MODALITY_KEYS["hci"]) == 10


# ── Feature utils ─────────────────────────────────────────────────────────────

class TestFeatureUtils:
    def setup_method(self):
        from ml.fusion.feature_utils import dicts_to_vector, vector_to_modality_dict, modality_slice
        from ml.fusion.feature_vector import FEATURE_DIM
        self.dicts_to_vector = dicts_to_vector
        self.vector_to_modality_dict = vector_to_modality_dict
        self.modality_slice = modality_slice
        self.FEATURE_DIM = FEATURE_DIM

    def test_empty_dicts_gives_zeros(self):
        vec = self.dicts_to_vector({})
        assert vec.shape == (self.FEATURE_DIM,)
        assert np.all(vec == 0)

    def test_hci_values_round_trip(self):
        hci = {"keystroke_rate": 3.5, "mouse_speed": 1.2}
        vec = self.dicts_to_vector({"hci": hci})
        back = self.vector_to_modality_dict(vec)
        assert abs(back["hci"]["keystroke_rate"] - 3.5) < 1e-5

    def test_missing_fill_applied(self):
        vec = self.dicts_to_vector({}, missing_fill=-1.0)
        assert np.all(vec == -1.0)

    def test_modality_slice_hci(self):
        start, end = self.modality_slice("hci")
        assert end - start == 10

    def test_modality_slice_face(self):
        start, end = self.modality_slice("face")
        assert start == 0 and end == 12

    def test_output_dtype_float32(self):
        vec = self.dicts_to_vector({"voice": {"pitch_mean": 120.0}})
        assert vec.dtype == np.float32
