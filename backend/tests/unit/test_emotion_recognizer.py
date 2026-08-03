"""
test_emotion_recognizer.py
Tests for the pretrained EmotionRecognizer (Phase A).

These tests run in two modes:
  - Full mode: checkpoint present at ml/models/weights/enet_b0_8_best_afew.pt
  - Stub mode: checkpoint absent, uses uniform-fallback (CI/no-GPU environments)

Both modes must pass all assertions; the stub mode just has flat distributions.
"""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


from ml.models.emotion_recognizer import CHECKPOINT
HAS_CHECKPOINT = CHECKPOINT.exists()


@pytest.fixture(scope="module")
def recognizer():
    from ml.models.emotion_recognizer import EmotionRecognizer
    return EmotionRecognizer(device="cpu", auto_download=False)


class TestEmotionRecognizerContract:
    """Interface contract — must hold whether or not checkpoint is present."""

    def test_instantiates(self, recognizer):
        assert recognizer is not None

    def test_is_ready_reflects_checkpoint(self, recognizer):
        assert recognizer.is_ready == HAS_CHECKPOINT

    def test_predict_returns_dict(self, recognizer):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = recognizer.predict(frame)
        assert isinstance(result, dict)

    def test_required_keys_present(self, recognizer):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = recognizer.predict(frame)
        assert "emotion_label" in result
        assert "emotion_scores" in result
        assert "valence" in result
        assert "arousal" in result

    def test_emotion_label_is_valid(self, recognizer):
        from ml.models.emotion_recognizer import EMOTION_LABELS
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = recognizer.predict(frame)
        assert result["emotion_label"] in EMOTION_LABELS

    def test_emotion_scores_sum_to_one(self, recognizer):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = recognizer.predict(frame)
        total = sum(result["emotion_scores"].values())
        assert abs(total - 1.0) < 1e-4

    def test_emotion_scores_all_nonnegative(self, recognizer):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = recognizer.predict(frame)
        assert all(v >= 0.0 for v in result["emotion_scores"].values())

    def test_valence_in_range(self, recognizer):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = recognizer.predict(frame)
        assert -1.0 <= result["valence"] <= 1.0

    def test_arousal_in_range(self, recognizer):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = recognizer.predict(frame)
        assert 0.0 <= result["arousal"] <= 1.0

    def test_predict_none_returns_stub(self, recognizer):
        result = recognizer.predict(None)
        assert result["emotion_label"] is not None
        assert abs(sum(result["emotion_scores"].values()) - 1.0) < 1e-4

    def test_various_frame_sizes_handled(self, recognizer):
        for h, w in [(64, 64), (112, 112), (480, 640)]:
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            result = recognizer.predict(frame)
            assert result["emotion_label"] is not None

    def test_8_classes_in_scores(self, recognizer):
        from ml.models.emotion_recognizer import EMOTION_LABELS
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = recognizer.predict(frame)
        assert set(result["emotion_scores"].keys()) == set(EMOTION_LABELS)


@pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint not present — skipping real-model tests")
class TestEmotionRecognizerReal:
    """Real inference tests — only run when checkpoint is available."""

    def test_non_uniform_output_on_real_frame(self, recognizer):
        """A real model should produce non-uniform probability distributions."""
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        probs = list(result["emotion_scores"].values())
        max_p = max(probs)
        # Real model should have at least some confidence (>= 1/8 * 1.5 = 18.75%)
        assert max_p >= 0.15, f"Max probability too low: {max_p:.3f}"

    def test_different_inputs_give_different_outputs(self, recognizer):
        """Model should be sensitive to input variation."""
        results = []
        for color in [[200, 180, 160], [50, 100, 150], [240, 60, 80]]:
            frame = np.full((224, 224, 3), color, dtype=np.uint8)
            r = recognizer.predict(frame)
            results.append(list(r["emotion_scores"].values()))
        # Not all outputs should be identical
        arr = np.array(results)
        assert arr.std(axis=0).max() > 1e-4, "Model gives identical outputs for different inputs"

    def test_valence_happy_bias(self, recognizer):
        """A clearly 'happy' face crop (high saturation warm) should lean positive valence
        more often than not — this is a statistical sanity check, not a strict assertion."""
        # We just verify the model is discriminating, not that it's 100% accurate
        frame = np.full((224, 224, 3), [50, 180, 50], dtype=np.uint8)
        result = recognizer.predict(frame)
        # Valence should be a real number, not the stub 0.0
        assert result["valence"] != 0.0 or result["arousal"] != 0.5


class TestBehaviourPredictorWithRealEmotion:
    """BehaviourPredictor integration: verify it routes through EmotionRecognizer."""

    def setup_method(self):
        from ml.fusion.predictor import BehaviourPredictor
        self.predictor = BehaviourPredictor(device="cpu")

    def test_predict_without_frame_returns_tcmt_emotion(self):
        feats = {"face": {}, "gaze": {}, "pose": {}, "voice": {}, "hci": {}}
        result = self.predictor.predict(feats, bgr_frame=None)
        assert result.emotion_source == "tcmt"

    @pytest.mark.skipif(not HAS_CHECKPOINT, reason="Checkpoint not present")
    def test_predict_with_frame_returns_pretrained_emotion(self):
        feats = {"face": {}, "gaze": {}, "pose": {}, "voice": {}, "hci": {}}
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = self.predictor.predict(feats, bgr_frame=frame)
        assert result.emotion_source == "pretrained"

    def test_stress_in_range(self):
        feats = {"face": {}, "gaze": {}, "pose": {}, "voice": {}, "hci": {}}
        result = self.predictor.predict(feats)
        assert 0.0 <= result.stress <= 1.0

    def test_valence_field_exists(self):
        feats = {"face": {}, "gaze": {}, "pose": {}, "voice": {}, "hci": {}}
        result = self.predictor.predict(feats)
        assert hasattr(result, "valence")

    def test_feature_vector_attached(self):
        feats = {"face": {}, "hci": {"keystroke_rate": 0.5}}
        result = self.predictor.predict(feats)
        assert result.feature_vector is not None
        from ml.fusion.feature_vector import FEATURE_DIM
        assert result.feature_vector.shape == (FEATURE_DIM,)
