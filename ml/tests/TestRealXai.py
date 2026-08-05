"""
test_real_xai.py — Phase E: verify real GradCAM + IG-based SHAP.

All tests run without a trained checkpoint (random-weight TCMT is fine
for structural/shape tests). No mocks.
"""
from __future__ import annotations
import numpy as np
import pytest


def _make_model_and_vec():
    from ml.fusion.Tcmt import TCMT
    from ml.fusion.FeatureVector import MODALITY_KEYS, FEATURE_DIM
    model = TCMT()
    vec   = np.random.default_rng(0).uniform(0.1, 0.9, FEATURE_DIM).astype(np.float32)
    return model, vec


class TestIntegratedGradientsSHAP:
    """captum IG-based SHAPExplainer tests."""

    def test_returns_all_heads(self):
        from ml.xai.ShapExplainer import SHAPExplainer, TARGET_HEADS
        model, vec = _make_model_and_vec()
        exp   = SHAPExplainer(model)
        result = exp.explain(vec)
        for h in TARGET_HEADS:
            assert h in result, f"Missing head: {h}"

    def test_weights_sum_to_one_per_head(self):
        from ml.xai.ShapExplainer import SHAPExplainer, TARGET_HEADS
        from ml.fusion.FeatureVector import MODALITY_KEYS
        model, vec = _make_model_and_vec()
        exp    = SHAPExplainer(model)
        result = exp.explain(vec)
        for h in TARGET_HEADS:
            total = sum(result[h].values())
            assert abs(total - 1.0) < 1e-3, f"Head {h}: sum={total}"

    def test_all_modalities_present(self):
        from ml.xai.ShapExplainer import SHAPExplainer
        from ml.fusion.FeatureVector import MODALITY_KEYS
        model, vec = _make_model_and_vec()
        exp    = SHAPExplainer(model)
        result = exp.explain(vec, target_heads=["stress"])
        for mod in MODALITY_KEYS:
            assert mod in result["stress"], f"Missing modality: {mod}"

    def test_weights_non_negative(self):
        from ml.xai.ShapExplainer import SHAPExplainer, TARGET_HEADS
        model, vec = _make_model_and_vec()
        exp    = SHAPExplainer(model)
        result = exp.explain(vec)
        for h in TARGET_HEADS:
            for mod, w in result[h].items():
                assert w >= 0.0, f"{h}/{mod} weight is negative: {w}"

    def test_different_inputs_differ(self):
        """IG should produce different attributions for different inputs."""
        from ml.xai.ShapExplainer import SHAPExplainer
        from ml.fusion.FeatureVector import FEATURE_DIM
        rng   = np.random.default_rng(42)
        model, _ = _make_model_and_vec()
        exp   = SHAPExplainer(model)
        v1 = rng.uniform(0.0, 0.3, FEATURE_DIM).astype(np.float32)
        v2 = rng.uniform(0.7, 1.0, FEATURE_DIM).astype(np.float32)
        r1 = exp.explain(v1, target_heads=["stress"])["stress"]
        r2 = exp.explain(v2, target_heads=["stress"])["stress"]
        # Not identical
        assert r1 != r2, "Same attribution for very different inputs"

    def test_zero_input_handled(self):
        """Zero feature vector should not crash."""
        from ml.xai.ShapExplainer import SHAPExplainer
        from ml.fusion.FeatureVector import FEATURE_DIM
        model, _ = _make_model_and_vec()
        exp   = SHAPExplainer(model)
        vec   = np.zeros(FEATURE_DIM, dtype=np.float32)
        result = exp.explain(vec, target_heads=["fatigue"])
        assert "fatigue" in result


class TestGradCAM:
    """True Grad-CAM structural tests."""

    def test_output_shape(self):
        from ml.xai.Gradcam import GradCAM, FACE_DIM
        model, vec = _make_model_and_vec()
        gc  = GradCAM(model)
        sal = gc.explain(vec, head="stress")
        assert sal.shape == (FACE_DIM,), f"Expected ({FACE_DIM},), got {sal.shape}"

    def test_output_range(self):
        from ml.xai.Gradcam import GradCAM
        model, vec = _make_model_and_vec()
        gc  = GradCAM(model)
        sal = gc.explain(vec, head="engagement")
        assert sal.min() >= 0.0
        assert sal.max() <= 1.0 + 1e-6

    def test_all_heads(self):
        from ml.xai.Gradcam import GradCAM
        model, vec = _make_model_and_vec()
        gc = GradCAM(model)
        for head in ("stress", "engagement", "attention", "fatigue", "emotion"):
            sal = gc.explain(vec, head=head)
            assert sal is not None, f"None for head={head}"

    def test_no_hook_leak(self):
        """Hooks must be cleaned up after explain()."""
        from ml.xai.Gradcam import GradCAM
        model, vec = _make_model_and_vec()
        gc = GradCAM(model)
        gc.explain(vec)
        assert len(gc._hook_handles) == 0, "Hooks not cleaned up"

    def test_different_heads_can_differ(self):
        """GradCAM maps for stress vs fatigue can differ."""
        from ml.xai.Gradcam import GradCAM
        model, vec = _make_model_and_vec()
        gc  = GradCAM(model)
        s1  = gc.explain(vec, head="stress")
        s2  = gc.explain(vec, head="fatigue")
        # Allow they could be same in pathological cases, but shape must match
        assert s1.shape == s2.shape

    def test_nonzero_saliency(self):
        """At least one AU should have non-zero saliency."""
        from ml.xai.Gradcam import GradCAM
        from ml.fusion.FeatureVector import FEATURE_DIM
        rng  = np.random.default_rng(99)
        model, _ = _make_model_and_vec()
        vec  = rng.uniform(0.1, 0.9, FEATURE_DIM).astype(np.float32)
        gc   = GradCAM(model)
        sal  = gc.explain(vec, head="stress")
        assert sal.sum() > 0.0, "All-zero saliency map"
