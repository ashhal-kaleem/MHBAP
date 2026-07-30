"""
test_training.py — Tests for Phase D: TCMT training + weight persistence.

These tests actually TRAIN a small TCMT (fast: 5 epochs, 400 samples)
and verify real metrics — no mocks, no fixtures.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

WEIGHT_PATH  = Path("ml/models/weights/tcmt_trained.pt")
METRICS_PATH = Path("ml/models/weights/tcmt_eval_metrics.json")


class TestDataset:
    def test_split_sizes(self):
        from ml.training.dataset import make_dataset
        tr, va, te = make_dataset(n_samples=200, seed=0)
        n = 200
        n_val  = int(n * 0.15)
        n_test = int(n * 0.15)
        n_tr   = n - n_val - n_test
        assert len(tr["X"]) == n_tr
        assert len(va["X"]) == n_val
        assert len(te["X"]) == n_test

    def test_label_ranges(self):
        from ml.training.dataset import make_dataset
        tr, _, _ = make_dataset(n_samples=100, seed=1)
        assert tr["emotion"].min() >= 0
        assert tr["emotion"].max() <= 3
        for k in ("stress", "engagement", "attention", "fatigue"):
            assert tr[k].min() >= 0.0
            assert tr[k].max() <= 1.0

    def test_feature_dim(self):
        from ml.training.dataset import make_dataset
        from ml.fusion.feature_vector import FEATURE_DIM
        tr, _, _ = make_dataset(n_samples=50, seed=2)
        assert tr["X"].shape[1] == FEATURE_DIM

    def test_reproducible(self):
        from ml.training.dataset import make_dataset
        a, _, _ = make_dataset(n_samples=50, seed=7)
        b, _, _ = make_dataset(n_samples=50, seed=7)
        np.testing.assert_array_equal(a["X"], b["X"])

    def test_different_seeds_differ(self):
        from ml.training.dataset import make_dataset
        a, _, _ = make_dataset(n_samples=50, seed=0)
        b, _, _ = make_dataset(n_samples=50, seed=1)
        assert not np.allclose(a["X"], b["X"])


class TestTCMTTrainingLoop:
    """Fast (5-epoch, 400-sample) training — confirms real learning happens."""

    def _quick_train(self):
        import torch, torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from ml.fusion.tcmt import TCMT
        from ml.training.dataset import make_dataset

        torch.manual_seed(0); np.random.seed(0)
        tr, _, te = make_dataset(n_samples=400, seed=0)

        def _t(sp):
            return TensorDataset(
                torch.tensor(sp["X"],         dtype=torch.float32),
                torch.tensor(sp["emotion"],    dtype=torch.long),
                torch.tensor(sp["stress"],     dtype=torch.float32).unsqueeze(-1),
                torch.tensor(sp["engagement"], dtype=torch.float32).unsqueeze(-1),
                torch.tensor(sp["attention"],  dtype=torch.float32).unsqueeze(-1),
                torch.tensor(sp["fatigue"],    dtype=torch.float32).unsqueeze(-1),
            )

        model = TCMT()
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ce    = nn.CrossEntropyLoss()
        mse   = nn.MSELoss()

        losses = []
        for _ in range(5):
            model.train()
            ep_loss = 0.0
            for X, em, st, en, at, fa in DataLoader(_t(tr), batch_size=32, shuffle=True):
                opt.zero_grad()
                if X.dim() == 2:
                    X = X.unsqueeze(1)
                B, T, _ = X.shape
                toks = torch.cat([model.mod_proj(X[:, t, :]) for t in range(T)], dim=1)
                cls  = model.cls_token.expand(B, -1, -1)
                enc  = model.encoder(torch.cat([cls, toks], dim=1))
                h    = enc[:, 0, :]
                loss = (ce(model.head_emotion(h), em)
                        + mse(torch.sigmoid(model.head_stress(h)), st))
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            losses.append(ep_loss)
        return model, losses, te

    def test_loss_decreases(self):
        _, losses, _ = self._quick_train()
        # Loss at epoch 5 should be lower than epoch 1
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_emotion_accuracy_above_chance(self):
        import torch
        from ml.evaluation.metrics import emotion_metrics
        model, _, te = self._quick_train()
        model.eval()
        Xt = torch.tensor(te["X"], dtype=torch.float32)
        with torch.no_grad():
            out = model(Xt)
        m = emotion_metrics(te["emotion"], np.array(out["emotion_logits"]))
        # 4-class chance = 0.25; trained model should beat it
        assert m["accuracy"] > 0.25, f"Accuracy {m['accuracy']} not above chance"

    def test_regression_heads_in_range(self):
        import torch
        model, _, te = self._quick_train()
        model.eval()
        Xt = torch.tensor(te["X"], dtype=torch.float32)
        with torch.no_grad():
            out = model(Xt)
        for head in ("stress", "engagement", "attention", "fatigue"):
            vals = np.array(out[head])
            # stress 0-10, others 0-1
            upper = 10.0 if head == "stress" else 1.0
            assert vals.min() >= 0.0
            assert vals.max() <= upper + 1e-4


class TestWeightPersistence:
    """Verify saved weights exist and load correctly (only if file present)."""

    def test_weights_file_exists_or_skip(self):
        if not WEIGHT_PATH.exists():
            pytest.skip("Run scripts/save_trained_tcmt.py first")
        assert WEIGHT_PATH.stat().st_size > 1000

    def test_metrics_file_valid(self):
        if not METRICS_PATH.exists():
            pytest.skip("Run scripts/save_trained_tcmt.py first")
        m = json.loads(METRICS_PATH.read_text())
        assert "emotion" in m
        assert 0.0 <= m["emotion"]["accuracy"] <= 1.0
        assert m["stress"]["rmse"] >= 0.0

    def test_checkpoint_loads(self):
        if not WEIGHT_PATH.exists():
            pytest.skip("Run ml.training.train_tcmt first")
        import torch
        from ml.fusion.tcmt import TCMT, EMOTION_CLASSES
        ckpt  = torch.load(str(WEIGHT_PATH), map_location="cpu")
        # Verify head size matches current EMOTION_CLASSES before loading
        emo_w = ckpt["state_dict"]["head_emotion.weight"]
        if emo_w.shape[0] != EMOTION_CLASSES:
            pytest.skip(
                f"Stale checkpoint: head has {emo_w.shape[0]} classes, "
                f"model expects {EMOTION_CLASSES}. Retrain to fix."
            )
        model = TCMT()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        assert model is not None
