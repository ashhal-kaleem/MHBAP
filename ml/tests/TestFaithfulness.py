"""Tests for ml/xai/Faithfulness.py (no dataset needed)."""
import numpy as np
import torch

from ml.fusion.VideoTcmt import VideoTCMT
from ml.xai import Faithfulness as FX


def _setup(B=4, T=8):
    torch.manual_seed(0)
    m = VideoTCMT(max_t=T, d_model=32, n_heads=4, n_layers=1, ffn_dim=64).eval()
    x = torch.randn(B, T, 28)
    v = torch.ones(B, T, dtype=torch.bool)
    v[0, -2:] = False
    return m, x, v


def test_ig_completeness():
    m, x, v = _setup()
    lvl = FX.predicted_level(m, x, v, 0)
    a = FX.integrated_gradients(m, x, v, 0, lvl, steps=64)
    assert a.shape == x.shape
    assert FX.completeness_gap(m, x, v, 0, lvl, a).max() < 0.02


def test_deletion_curve_endpoints():
    m, x, v = _setup()
    lvl = FX.predicted_level(m, x, v, 1)
    a = FX.gradient_x_input(m, x, v, 1, lvl)
    c = FX.deletion_insertion(m, x, v, 1, lvl, a, n_steps=5)
    assert c["deletion"].shape == (4, 6)
    # step 0 of deletion = full input, last step of insertion = full input
    np.testing.assert_allclose(c["deletion"][:, 0], c["insertion"][:, -1], atol=1e-5)
    np.testing.assert_allclose(c["deletion"][:, -1], c["insertion"][:, 0], atol=1e-5)


def test_randomise_changes_every_parameter():
    m, _, _ = _setup()
    r = FX.randomise_model(m)
    for (n1, p1), (_, p2) in zip(m.named_parameters(), r.named_parameters()):
        if p1.dim() >= 2:   # weight matrices / embeddings (norms reset to 1/0 by design)
            assert not torch.allclose(p1, p2), n1


def test_spearman_and_auc():
    a = np.arange(10.0)
    assert abs(FX.spearman(a, a) - 1) < 1e-9
    assert abs(FX.spearman(a, -a) + 1) < 1e-9
    assert abs(FX.auc(np.array([[1.0, 1.0, 1.0]]))[0] - 1.0) < 1e-9


def test_modality_shares_sum_to_one():
    m, x, v = _setup()
    from ml.training.DaiseeDataset import VISUAL_SLICES
    sh = FX.modality_shares(torch.randn_like(x), VISUAL_SLICES)
    np.testing.assert_allclose(sum(sh.values()), 1.0, atol=1e-5)


def test_face_au_saliency_is_deterministic():
    """Regression: saliency used to run in train() mode, so dropout made it random."""
    from ml.fusion.Tcmt import TCMT
    from ml.fusion.FeatureVector import FEATURE_DIM
    from ml.xai.Gradcam import FaceAUSaliency
    torch.manual_seed(0)
    vec = np.random.default_rng(0).uniform(0.1, 0.9, FEATURE_DIM).astype(np.float32)
    sal = FaceAUSaliency(TCMT())
    np.testing.assert_allclose(sal.explain(vec, "engagement"), sal.explain(vec, "engagement"))


def test_ig_alias_backward_compatible():
    from ml.xai.ShapExplainer import SHAPExplainer
    from ml.xai.IntegratedGradients import IntegratedGradientsExplainer
    assert SHAPExplainer is IntegratedGradientsExplainer
