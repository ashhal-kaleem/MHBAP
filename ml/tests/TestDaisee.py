"""Tests for the DAiSEE benchmark pipeline (no dataset or MediaPipe needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.training.DaiseeDataset import (
    VISUAL_DIM, apply_normaliser, class_weights, fit_normaliser,
    read_labels, sample_frame_indices, synthetic_split,
)

torch = pytest.importorskip("torch")


def test_read_labels_handles_trailing_space_header(tmp_path: Path):
    p = tmp_path / "TrainLabels.csv"
    p.write_text("ClipID,Boredom,Engagement,Confusion,Frustration \n"
                 "1100011002.avi,0,2,0,0\n1100011003.mp4,1,3,0,1\n")
    labels = read_labels(p)
    assert labels["1100011002"].tolist() == [0, 2, 0, 0]
    assert labels["1100011003"].tolist() == [1, 3, 0, 1]


def test_frame_indices_uniform_and_in_range():
    idx = sample_frame_indices(300, 32)
    assert len(idx) == 32 and idx.min() >= 0 and idx.max() < 300
    assert np.all(np.diff(idx) > 0)


def test_normaliser_uses_valid_frames_and_zeroes_invalid():
    d = synthetic_split(20, t=8)
    d["valid"][:, -2:] = False
    mu, sd = fit_normaliser(d["X"], d["valid"])
    Z = apply_normaliser(d["X"], d["valid"], mu, sd)
    assert Z.shape == d["X"].shape
    assert np.all(Z[:, -2:] == 0)


def test_class_weights_upweight_rare_levels():
    y = np.array([[0, 0, 0, 0]] * 9 + [[1, 1, 1, 1]])
    w = class_weights(y)
    assert w[0, 1] > w[0, 0]


def test_video_tcmt_shapes_and_ablations():
    from ml.fusion.VideoTcmt import VideoTCMT
    x = torch.randn(3, 16, VISUAL_DIM)
    v = torch.ones(3, 16, dtype=torch.bool)
    v[0, 5:] = False
    for temporal in (True, False):
        m = VideoTCMT(max_t=16, temporal=temporal)
        assert m(x, v).shape == (3, 4, 4)
        assert m(x, v, drop_modalities=["pose"]).shape == (3, 4, 4)
    assert VideoTCMT(modalities=["gaze"], max_t=16)(x, v).shape == (3, 4, 4)


def test_temporal_model_is_order_sensitive():
    """Original TCMT had no time embedding; VideoTCMT must see frame order."""
    from ml.fusion.VideoTcmt import VideoTCMT
    torch.manual_seed(0)
    m = VideoTCMT(max_t=8).eval()
    x = torch.randn(1, 8, VISUAL_DIM)
    assert not torch.allclose(m(x), m(x.flip(1)), atol=1e-5)


def test_train_eval_smoke():
    from scripts.daisee.train_eval import majority_baseline, train_one
    data = {"train": synthetic_split(64, t=8, seed=0),
            "val": synthetic_split(32, t=8, seed=1),
            "test": synthetic_split(32, t=8, seed=2)}
    r = train_one(data, ["face", "gaze", "pose"], True, seed=0,
                  epochs=2, lr=1e-3, bs=16, device="cpu")
    assert set(r["test"]) == {"Boredom", "Engagement", "Confusion", "Frustration"}
    assert 0.0 <= r["test"]["Engagement"]["accuracy"] <= 1.0
    assert "Engagement" in majority_baseline(data)
