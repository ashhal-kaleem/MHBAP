"""
test_checkpoint_loading.py — Regression tests for TCMT checkpoint load path.

Covers:
  1. Trainer always saves the wrapper-dict format {"state_dict": ..., "test_metrics": ...}.
  2. Predictor correctly unwraps and loads a wrapper-dict checkpoint.
  3. Predictor is backward-compatible with a bare state_dict (legacy format).
  4. Real on-disk checkpoint (tcmt_trained.pt) has the expected top-level keys.
  5. Predictor discovers tcmt_trained.pt via the fallback path when tcmt.pt is absent.
  6. End-to-end: predictor loaded from real checkpoint produces valid PredictionResult.

Run from repo root:
    pytest ml/tests/TestCheckpointLoading.py -v
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

REAL_CKPT = Path("ml/models/weights/tcmt_trained.pt")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _zero_feature_dicts():
    from ml.fusion.FeatureVector import MODALITY_KEYS
    return {mod: {k: 0.0 for k in keys} for mod, keys in MODALITY_KEYS.items()}


def _fresh_state_dict():
    """Return a random-init TCMT state_dict (no file I/O)."""
    from ml.fusion.Tcmt import TCMT
    return TCMT().state_dict()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Trainer save format
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainerSaveFormat:
    """Verify TCMT_Train_Colab.ipynb produces the wrapper-dict format."""

    def test_real_checkpoint_is_wrapper_dict(self):
        """The on-disk file must be a dict, not a bare state_dict."""
        if not REAL_CKPT.exists():
            pytest.skip("tcmt_trained.pt not present — run training first")
        import torch
        ckpt = torch.load(str(REAL_CKPT), map_location="cpu", weights_only=True)
        assert isinstance(ckpt, dict), "Checkpoint must be a dict"
        assert "state_dict" in ckpt, (
            f"Expected 'state_dict' key; got keys: {list(ckpt.keys())}"
        )

    def test_real_checkpoint_extra_keys(self):
        """Wrapper dict should carry metadata keys alongside state_dict."""
        if not REAL_CKPT.exists():
            pytest.skip("tcmt_trained.pt not present — run training first")
        import torch
        ckpt = torch.load(str(REAL_CKPT), map_location="cpu", weights_only=True)
        # At least one metadata key must exist beyond state_dict
        meta_keys = {k for k in ckpt if k != "state_dict"}
        assert meta_keys, f"No metadata keys found; ckpt keys: {list(ckpt.keys())}"

    def test_real_state_dict_has_expected_layers(self):
        """state_dict inside wrapper must contain TCMT layer names."""
        if not REAL_CKPT.exists():
            pytest.skip("tcmt_trained.pt not present — run training first")
        import torch
        ckpt = torch.load(str(REAL_CKPT), map_location="cpu", weights_only=True)
        sd = ckpt["state_dict"]
        has_emo_head = "head_emotion.weight" in sd or "head_emotion.0.weight" in sd
        assert has_emo_head, f"Missing head_emotion weight in state_dict: {list(sd.keys())}"
        assert "head_stress.weight" in sd
        assert "cls_token" in sd


# ──────────────────────────────────────────────────────────────────────────────
# 2. Predictor loads wrapper-dict format correctly
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictorLoadsWrapperDict:
    """Regression: predictor must unwrap {"state_dict": ...} before load_state_dict."""

    def _save_wrapper_ckpt(self, tmp_dir: Path) -> Path:
        import torch
        path = tmp_dir / "wrapper.pt"
        torch.save(
            {"state_dict": _fresh_state_dict(), "test_metrics": {"emotion": {"accuracy": 0.5}}},
            str(path),
        )
        return path

    def test_no_runtime_error_on_wrapper_dict(self, tmp_path):
        """load_state_dict must not raise when checkpoint is a wrapper dict."""
        ckpt_path = self._save_wrapper_ckpt(tmp_path)
        from ml.fusion.Predictor import BehaviourPredictor
        # Should not raise RuntimeError (unexpected keys: state_dict, test_metrics, ...)
        pred = BehaviourPredictor(weights_path=ckpt_path)
        assert pred is not None

    def test_predict_works_after_wrapper_load(self, tmp_path):
        """After loading wrapper-dict weights, predict() must return valid result."""
        ckpt_path = self._save_wrapper_ckpt(tmp_path)
        from ml.fusion.Predictor import BehaviourPredictor, PredictionResult
        pred = BehaviourPredictor(weights_path=ckpt_path)
        result = pred.predict(_zero_feature_dicts())
        assert isinstance(result, PredictionResult)
        assert 0.0 <= result.stress <= 1.0
        assert 0.0 <= result.engagement <= 1.0
        assert 0.0 <= result.attention <= 1.0
        assert 0.0 <= result.fatigue <= 1.0
        assert abs(sum(result.emotion_scores.values()) - 1.0) < 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# 3. Backward-compat: bare state_dict (legacy format)
# ──────────────────────────────────────────────────────────────────────────────

class TestPredictorLoadsBareSateDict:
    """Predictor must also accept a raw state_dict (no wrapper dict)."""

    def _save_bare_ckpt(self, tmp_dir: Path) -> Path:
        import torch
        path = tmp_dir / "bare.pt"
        torch.save(_fresh_state_dict(), str(path))
        return path

    def test_no_runtime_error_on_bare_state_dict(self, tmp_path):
        ckpt_path = self._save_bare_ckpt(tmp_path)
        from ml.fusion.Predictor import BehaviourPredictor
        pred = BehaviourPredictor(weights_path=ckpt_path)
        assert pred is not None

    def test_predict_works_after_bare_load(self, tmp_path):
        ckpt_path = self._save_bare_ckpt(tmp_path)
        from ml.fusion.Predictor import BehaviourPredictor, PredictionResult
        pred = BehaviourPredictor(weights_path=ckpt_path)
        result = pred.predict(_zero_feature_dicts())
        assert isinstance(result, PredictionResult)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Filename fallback: tcmt.pt absent → tcmt_trained.pt discovered automatically
# ──────────────────────────────────────────────────────────────────────────────

class TestFilenameResolutionFallback:
    """Predictor must auto-discover tcmt_trained.pt when tcmt.pt is absent."""

    def test_fallback_to_trained_pt(self, tmp_path):
        import torch
        # Place a wrapper-dict checkpoint named tcmt_trained.pt only
        trained_path = tmp_path / "tcmt_trained.pt"
        torch.save({"state_dict": _fresh_state_dict(), "test_metrics": {}}, str(trained_path))

        # Pass the non-existent tcmt.pt path — predictor must fall back
        from ml.fusion.Predictor import BehaviourPredictor, PredictionResult
        missing_path = tmp_path / "tcmt.pt"
        pred = BehaviourPredictor(weights_path=missing_path)
        result = pred.predict(_zero_feature_dicts())
        assert isinstance(result, PredictionResult)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Real checkpoint end-to-end
# ──────────────────────────────────────────────────────────────────────────────

class TestRealCheckpointEndToEnd:
    """Load the actual tcmt_trained.pt and run one inference."""

    def test_real_checkpoint_full_roundtrip(self):
        if not REAL_CKPT.exists():
            pytest.skip("tcmt_trained.pt not present — run training first")
        from ml.fusion.Predictor import BehaviourPredictor, PredictionResult
        pred = BehaviourPredictor(weights_path=REAL_CKPT)
        result = pred.predict(_zero_feature_dicts())
        assert isinstance(result, PredictionResult)
        assert result.emotion in [
            "neutral", "happy", "sad", "surprise",
            "fear", "disgust", "anger", "contempt",
        ]
        assert 0.0 <= result.stress <= 1.0
        total_prob = sum(result.emotion_scores.values())
        assert abs(total_prob - 1.0) < 1e-3, f"Probs sum to {total_prob}"

    def test_real_checkpoint_print_keys(self):
        """Prove the checkpoint structure by printing all top-level keys."""
        if not REAL_CKPT.exists():
            pytest.skip("tcmt_trained.pt not present — run training first")
        import torch
        ckpt = torch.load(str(REAL_CKPT), map_location="cpu", weights_only=True)
        print("\n--- Checkpoint top-level keys ---")
        for k, v in ckpt.items():
            if k == "state_dict":
                print(f"  state_dict  →  {len(v)} tensors, "
                      f"e.g. {list(v.keys())[:4]}")
            else:
                print(f"  {k}  →  {type(v).__name__}: {v!r}"[:120])
        assert "state_dict" in ckpt


# ──────────────────────────────────────────────────────────────────────────────
# 6. Regression: trained weights actually change inference vs random-init
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainedWeightsActuallyUsed:
    """
    Prove that loading the on-disk checkpoint changes the model's parameters
    relative to a random-init TCMT, and that this difference is reflected in
    inference outputs.

    These are the key regression tests that guard against silent fallbacks
    where the predictor silently uses random weights despite a valid checkpoint
    being present.
    """

    def test_trained_state_dict_differs_from_random_init(self):
        """At least one parameter tensor must differ between trained and random TCMT."""
        if not REAL_CKPT.exists():
            pytest.skip("tcmt_trained.pt not present — run training first")
        import torch
        from ml.fusion.Tcmt import TCMT

        random_sd  = TCMT().state_dict()
        ckpt       = torch.load(str(REAL_CKPT), map_location="cpu", weights_only=True)
        trained_sd = ckpt["state_dict"]

        assert set(random_sd.keys()) == set(trained_sd.keys()), (
            "Trained state_dict keys don't match TCMT architecture"
        )
        any_diff = any(
            not torch.equal(random_sd[k], trained_sd[k])
            for k in random_sd
        )
        assert any_diff, (
            "Trained checkpoint is identical to a fresh random-init TCMT — "
            "weights were never updated by training or the wrong file was saved."
        )

    def test_predictor_loaded_from_checkpoint_differs_from_random_init(self):
        """
        A predictor loaded from the real checkpoint must produce outputs that
        differ from a fresh random-init predictor on at least one inference.

        This test constructs a non-trivial (non-zero) feature input so that
        both models have something to respond to, then verifies their outputs
        differ — proving the checkpoint weights are actually in use.
        """
        if not REAL_CKPT.exists():
            pytest.skip("tcmt_trained.pt not present — run training first")

        import torch
        from ml.fusion.FeatureVector import MODALITY_KEYS
        from ml.fusion.Predictor import BehaviourPredictor

        # Non-trivial input: all features set to 1.0
        feature_dicts = {mod: {k: 1.0 for k in keys} for mod, keys in MODALITY_KEYS.items()}

        # Predictor from trained checkpoint
        pred_trained = BehaviourPredictor(weights_path=REAL_CKPT)

        # Predictor with a fresh random-init TCMT (save a random-init checkpoint)
        import tempfile, pathlib
        from ml.fusion.Tcmt import TCMT
        with tempfile.TemporaryDirectory() as td:
            rand_path = pathlib.Path(td) / "rand.pt"
            torch.save({"state_dict": TCMT().state_dict(), "test_metrics": {}}, str(rand_path))
            pred_random = BehaviourPredictor(weights_path=rand_path)

        result_trained = pred_trained.predict(feature_dicts)
        result_random  = pred_random.predict(feature_dicts)

        # At least one continuous output must differ
        diffs = {
            "stress":     abs(result_trained.stress     - result_random.stress),
            "engagement": abs(result_trained.engagement - result_random.engagement),
            "attention":  abs(result_trained.attention  - result_random.attention),
            "fatigue":    abs(result_trained.fatigue    - result_random.fatigue),
        }
        assert any(d > 1e-4 for d in diffs.values()), (
            f"Trained and random-init predictors produced identical outputs — "
            f"the checkpoint was not actually loaded.\n  diffs={diffs}"
        )

    def test_predictor_weights_match_checkpoint_tensors(self):
        """
        After loading, every parameter in self._tcmt must equal the corresponding
        tensor in the checkpoint file — a direct tensor-equality check.
        """
        if not REAL_CKPT.exists():
            pytest.skip("tcmt_trained.pt not present — run training first")

        import torch
        from ml.fusion.Predictor import BehaviourPredictor

        pred = BehaviourPredictor(weights_path=REAL_CKPT)
        ckpt = torch.load(str(REAL_CKPT), map_location="cpu", weights_only=True)
        trained_sd = ckpt["state_dict"]
        live_sd    = pred._tcmt.state_dict()

        mismatched = [
            k for k in trained_sd
            if not torch.equal(trained_sd[k].cpu(), live_sd[k].cpu())
        ]
        assert not mismatched, (
            f"Parameters differ between checkpoint and loaded model: {mismatched[:5]}"
        )
