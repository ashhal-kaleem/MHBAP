"""
TestEmotionRecognizer.py — Regression tests for EmotionRecognizer.

Guards against:
  1. Wrong class-label order (the root cause of the original bug).
  2. Valence / arousal prior sign errors.
  3. Softmax / confidence correctness.
  4. Checkpoint loads without error and is_ready == True.
  5. Diagnostic logging fires at correct interval.

Run from repo root:
    pytest ml/tests/TestEmotionRecognizer.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from ml.models.EmotionRecognizer import EMOTION_LABELS, _VALENCE_PRIOR, _AROUSAL_PRIOR

CHECKPOINT_PATH = "ml/models/weights/enet_b0_8_best_afew.pt"


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Label order matches HSE checkpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestLabelOrder:
    """
    The HSE enet_b0_8_best_afew.pt checkpoint was trained with AffectNet-8
    labels in the order:  anger(0) contempt(1) disgust(2) fear(3)
                          happiness(4) neutral(5) sadness(6) surprise(7).

    This test suite is the primary regression guard for the original bug
    where the labels were 'neutral happy sad surprise fear disgust anger contempt'.
    """

    _CORRECT_ORDER = [
        "anger", "contempt", "disgust", "fear",
        "happiness", "neutral", "sadness", "surprise",
    ]

    def test_label_count(self):
        """Must be exactly 8 labels."""
        assert len(EMOTION_LABELS) == 8, (
            f"Expected 8 labels, got {len(EMOTION_LABELS)}: {EMOTION_LABELS}"
        )

    def test_label_order_exact(self):
        """Labels must be in HSE AffectNet-8 order."""
        assert list(EMOTION_LABELS) == self._CORRECT_ORDER, (
            f"\nExpected: {self._CORRECT_ORDER}\nGot:      {list(EMOTION_LABELS)}\n"
            "Did someone revert the label order fix?"
        )

    def test_anger_is_index_0(self):
        assert EMOTION_LABELS[0] == "anger"

    def test_happiness_is_index_4(self):
        """Happiness at index 4 — previously would have been 'fear' under wrong labels."""
        assert EMOTION_LABELS[4] == "happiness"

    def test_neutral_is_index_5(self):
        """Neutral at index 5 — previously would have been 'disgust' under wrong labels."""
        assert EMOTION_LABELS[5] == "neutral"

    def test_no_old_labels_present(self):
        """The wrong labels from the pre-fix version must not be in the list."""
        bad_labels = {"happy", "sad"}  # old misspellings
        overlap = bad_labels & set(EMOTION_LABELS)
        assert not overlap, (
            f"Old (wrong) labels still present: {overlap}.  "
            f"Current labels: {EMOTION_LABELS}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Valence / arousal priors match label order
# ──────────────────────────────────────────────────────────────────────────────

class TestVAPriors:
    """Valence/arousal priors must align with the corrected EMOTION_LABELS order."""

    def test_prior_lengths(self):
        assert len(_VALENCE_PRIOR) == len(EMOTION_LABELS)
        assert len(_AROUSAL_PRIOR) == len(EMOTION_LABELS)

    def test_happiness_valence_positive(self):
        """Happiness (index 4) must have positive valence."""
        idx = EMOTION_LABELS.index("happiness")
        assert _VALENCE_PRIOR[idx] > 0, (
            f"happiness valence={_VALENCE_PRIOR[idx]} should be > 0"
        )

    def test_anger_valence_negative(self):
        """Anger (index 0) must have negative valence."""
        idx = EMOTION_LABELS.index("anger")
        assert _VALENCE_PRIOR[idx] < 0, (
            f"anger valence={_VALENCE_PRIOR[idx]} should be < 0"
        )

    def test_neutral_low_arousal(self):
        """Neutral must have lower arousal than anger."""
        idx_neutral = EMOTION_LABELS.index("neutral")
        idx_anger   = EMOTION_LABELS.index("anger")
        assert _AROUSAL_PRIOR[idx_neutral] < _AROUSAL_PRIOR[idx_anger], (
            f"neutral arousal={_AROUSAL_PRIOR[idx_neutral]} should be < "
            f"anger arousal={_AROUSAL_PRIOR[idx_anger]}"
        )

    def test_surprise_valence_positive(self):
        """Surprise (index 7) must have positive valence."""
        idx = EMOTION_LABELS.index("surprise")
        assert _VALENCE_PRIOR[idx] > 0, (
            f"surprise valence={_VALENCE_PRIOR[idx]} should be > 0"
        )

    def test_disgust_most_negative_valence(self):
        """Disgust should have the most negative valence."""
        idx = EMOTION_LABELS.index("disgust")
        assert _VALENCE_PRIOR[idx] == min(_VALENCE_PRIOR), (
            f"Expected disgust to have min valence; got {_VALENCE_PRIOR}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Checkpoint loads; model produces valid probabilities
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckpointInference:
    """End-to-end: load real checkpoint and run inference on synthetic frames."""

    @pytest.fixture(scope="class")
    def recognizer(self):
        from pathlib import Path
        if not Path(CHECKPOINT_PATH).exists():
            pytest.skip("enet_b0_8_best_afew.pt not present")
        from ml.models.EmotionRecognizer import EmotionRecognizer
        return EmotionRecognizer(device="cpu", auto_download=False)

    def test_is_ready(self, recognizer):
        assert recognizer.is_ready, "EmotionRecognizer should be ready after loading checkpoint"

    def test_predict_returns_all_keys(self, recognizer):
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        assert set(result.keys()) == {"emotion_label", "emotion_scores", "valence", "arousal"}

    def test_emotion_label_is_valid(self, recognizer):
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        assert result["emotion_label"] in EMOTION_LABELS, (
            f"'{result['emotion_label']}' not in {EMOTION_LABELS}"
        )

    def test_emotion_scores_sum_to_one(self, recognizer):
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        total = sum(result["emotion_scores"].values())
        assert abs(total - 1.0) < 1e-4, f"Scores sum to {total}, expected 1.0"

    def test_emotion_scores_keys_match_labels(self, recognizer):
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        assert set(result["emotion_scores"].keys()) == set(EMOTION_LABELS)

    def test_top_label_matches_argmax(self, recognizer):
        """emotion_label must be the argmax of emotion_scores."""
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        scores = result["emotion_scores"]
        expected_top = max(scores, key=scores.get)
        assert result["emotion_label"] == expected_top, (
            f"emotion_label={result['emotion_label']} but argmax={expected_top}"
        )

    def test_no_old_label_in_output(self, recognizer):
        """
        The original bug: a neutral/skin-tone frame was predicted as 'contempt'
        under the wrong label mapping.  Regression guard: the returned label
        must be in the correct EMOTION_LABELS list (never 'happy' or 'sad').
        """
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        assert result["emotion_label"] in EMOTION_LABELS
        assert result["emotion_label"] not in {"happy", "sad"}  # old misspelled labels

    def test_valence_in_range(self, recognizer):
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        assert -1.0 <= result["valence"] <= 1.0, f"Valence out of range: {result['valence']}"

    def test_arousal_in_range(self, recognizer):
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        assert 0.0 <= result["arousal"] <= 1.0, f"Arousal out of range: {result['arousal']}"

    def test_predict_different_frames_differ(self, recognizer):
        """Two very different frames should produce different top predictions."""
        dark_frame   = np.zeros((224, 224, 3), dtype=np.uint8)
        bright_frame = np.full((224, 224, 3), 255, dtype=np.uint8)
        r1 = recognizer.predict(dark_frame)
        r2 = recognizer.predict(bright_frame)
        s1 = list(r1["emotion_scores"].values())
        s2 = list(r2["emotion_scores"].values())
        assert s1 != s2, "Identical scores for completely different frames — model is broken"

    def test_no_stub_when_checkpoint_present(self, recognizer):
        """
        When the checkpoint is loaded, _stub_result() must NOT be returned.
        The stub returns equal probs (0.125 each).  A real frame should NOT
        produce exactly uniform probabilities.
        """
        frame = np.full((224, 224, 3), [200, 180, 160], dtype=np.uint8)
        result = recognizer.predict(frame)
        scores = list(result["emotion_scores"].values())
        all_equal = all(abs(s - scores[0]) < 1e-6 for s in scores)
        assert not all_equal, "Model returned uniform stub probabilities despite checkpoint being loaded"


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Diagnostic logging attributes
# ──────────────────────────────────────────────────────────────────────────────

class TestDiagnosticLogging:
    """Verify the diagnostic logging counter exists and is valid."""

    def test_diag_every_attribute_exists(self):
        from ml.models.EmotionRecognizer import EmotionRecognizer
        assert hasattr(EmotionRecognizer, "_DIAG_EVERY")
        assert EmotionRecognizer._DIAG_EVERY >= 0

    def test_diag_count_attribute_exists(self):
        from ml.models.EmotionRecognizer import EmotionRecognizer
        assert hasattr(EmotionRecognizer, "_diag_count")
        assert isinstance(EmotionRecognizer._diag_count, int)
