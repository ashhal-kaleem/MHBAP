"""Unit tests for Phase 5 — fusion + XAI. No GPU, no trained weights."""
from __future__ import annotations
import unittest
import numpy as np


class TestFeatureVector(unittest.TestCase):

    def test_dim_is_57(self):
        from ml.fusion.feature_vector import FEATURE_DIM
        self.assertEqual(FEATURE_DIM, 57)

    def test_dicts_to_vector_zeros(self):
        from ml.fusion.feature_utils import dicts_to_vector
        from ml.fusion.feature_vector import FEATURE_DIM
        vec = dicts_to_vector({})
        self.assertEqual(vec.shape[0], FEATURE_DIM)
        self.assertTrue(np.all(vec == 0.0))

    def test_dicts_to_vector_face_keys(self):
        from ml.fusion.feature_utils import dicts_to_vector
        from ml.fusion.feature_vector import FACE_KEYS
        face = {k: 1.0 for k in FACE_KEYS}
        vec = dicts_to_vector({"face": face})
        self.assertTrue(np.all(vec[:12] == 1.0))
        self.assertTrue(np.all(vec[12:] == 0.0))

    def test_modality_slice_coverage(self):
        from ml.fusion.feature_utils import modality_slice
        from ml.fusion.feature_vector import MODALITY_KEYS, FEATURE_DIM
        ends = []
        for mod in MODALITY_KEYS:
            s, e = modality_slice(mod)
            ends.append(e)
        self.assertEqual(max(ends), FEATURE_DIM)

    def test_round_trip(self):
        from ml.fusion.feature_utils import dicts_to_vector, vector_to_modality_dict
        from ml.fusion.feature_vector import MODALITY_KEYS
        original = {
            mod: {k: float(i) for i, k in enumerate(keys)}
            for mod, keys in MODALITY_KEYS.items()
        }
        vec = dicts_to_vector(original)
        recovered = vector_to_modality_dict(vec)
        for mod, keys_dict in original.items():
            for k, v in keys_dict.items():
                self.assertAlmostEqual(recovered[mod][k], v, places=5)


class TestPredictor(unittest.TestCase):

    def _zeros(self):
        from ml.fusion.feature_vector import MODALITY_KEYS
        return {mod: {k: 0.0 for k in keys} for mod, keys in MODALITY_KEYS.items()}

    def test_predict_returns_result(self):
        from ml.fusion.predictor import BehaviourPredictor, PredictionResult
        p = BehaviourPredictor()
        result = p.predict(self._zeros())
        self.assertIsInstance(result, PredictionResult)

    def test_emotion_label_valid(self):
        from ml.fusion.predictor import BehaviourPredictor, EMOTION_LABELS
        p = BehaviourPredictor()
        result = p.predict(self._zeros())
        self.assertIn(result.emotion, EMOTION_LABELS)

    def test_scores_sum_to_one(self):
        from ml.fusion.predictor import BehaviourPredictor
        p = BehaviourPredictor()
        result = p.predict(self._zeros())
        total = sum(result.emotion_scores.values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_scalar_ranges(self):
        from ml.fusion.predictor import BehaviourPredictor
        p = BehaviourPredictor()
        result = p.predict(self._zeros())
        self.assertGreaterEqual(result.stress, 0.0)
        self.assertLessEqual(result.stress, 10.0)
        for attr in ("engagement", "attention", "fatigue"):
            v = getattr(result, attr)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_buffer_accumulates(self):
        from ml.fusion.predictor import BehaviourPredictor
        p = BehaviourPredictor()
        for _ in range(10):
            r = p.predict(self._zeros())
        self.assertIsNotNone(r)


class TestSHAP(unittest.TestCase):

    def _predictor_and_vec(self):
        from ml.fusion.predictor import BehaviourPredictor
        from ml.fusion.feature_vector import MODALITY_KEYS
        p = BehaviourPredictor()
        dicts = {mod: {k: 0.1 for k in keys} for mod, keys in MODALITY_KEYS.items()}
        result = p.predict(dicts)
        return p, result.feature_vector

    def test_shap_keys(self):
        from ml.xai.shap_explainer import SHAPExplainer
        from ml.fusion.feature_vector import MODALITY_KEYS
        p, vec = self._predictor_and_vec()
        exp = SHAPExplainer(p._model)
        shap = exp.explain(vec, target_heads=["stress"])
        self.assertIn("stress", shap)
        for mod in MODALITY_KEYS:
            self.assertIn(mod, shap["stress"])

    def test_shap_sums_to_one(self):
        from ml.xai.shap_explainer import SHAPExplainer
        p, vec = self._predictor_and_vec()
        exp = SHAPExplainer(p._model)
        shap = exp.explain(vec, target_heads=["stress"])
        total = sum(shap["stress"].values())
        self.assertAlmostEqual(total, 1.0, places=3)


class TestNLExplainer(unittest.TestCase):

    def test_returns_string(self):
        from ml.fusion.predictor import BehaviourPredictor
        from ml.xai.nl_explainer import generate_explanation
        from ml.fusion.feature_vector import MODALITY_KEYS
        p = BehaviourPredictor()
        dicts = {mod: {k: 0.0 for k in keys} for mod, keys in MODALITY_KEYS.items()}
        result = p.predict(dicts)
        shap = {"face": 0.4, "gaze": 0.2, "pose": 0.1, "voice": 0.2, "hci": 0.1}
        explanation = generate_explanation(result, shap)
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 20)


if __name__ == "__main__":
    unittest.main()
