"""Unit tests for the evaluation package."""
import pytest
from app.evaluation.metrics import (
    precision_recall_f1,
    confusion_matrix,
    cohen_kappa,
    mae,
    rmse,
    compute_report,
)
from app.evaluation.ablation import run_ablation, MODALITIES
from app.evaluation.benchmark import run_benchmark


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestPrecisionRecallF1:
    def test_perfect(self):
        y = [0, 1, 2, 0, 1, 2]
        prec, rec, f1, sup = precision_recall_f1(y, y)
        for c in [0, 1, 2]:
            assert prec[c] == 1.0
            assert rec[c] == 1.0
            assert f1[c] == 1.0

    def test_all_wrong(self):
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        prec, rec, f1, sup = precision_recall_f1(y_true, y_pred, labels=[0, 1])
        assert rec[0] == 0.0
        assert prec[0] == 0.0

    def test_support_counts(self):
        y_true = [0, 0, 1, 2]
        y_pred = [0, 1, 1, 2]
        _, _, _, sup = precision_recall_f1(y_true, y_pred)
        assert sup[0] == 2
        assert sup[1] == 1
        assert sup[2] == 1


class TestConfusionMatrix:
    def test_shape(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_diagonal_perfect(self):
        y_true = [0, 1, 2]
        y_pred = [0, 1, 2]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        assert cm[0][0] == 1
        assert cm[1][1] == 1
        assert cm[2][2] == 1


class TestCohenKappa:
    def test_perfect(self):
        y = [0, 1, 2, 0, 1]
        assert cohen_kappa(y, y) == pytest.approx(1.0, abs=1e-6)

    def test_empty(self):
        assert cohen_kappa([], []) == 0.0

    def test_range(self):
        import random
        rng = random.Random(0)
        y_true = [rng.randint(0, 3) for _ in range(200)]
        y_pred = [rng.randint(0, 3) for _ in range(200)]
        k = cohen_kappa(y_true, y_pred)
        assert -1.0 <= k <= 1.0


class TestMAE_RMSE:
    def test_zero(self):
        y = [1.0, 2.0, 3.0]
        assert mae(y, y) == pytest.approx(0.0)
        assert rmse(y, y) == pytest.approx(0.0)

    def test_values(self):
        assert mae([0.0], [1.0]) == pytest.approx(1.0)
        assert rmse([0.0], [1.0]) == pytest.approx(1.0)

    def test_empty(self):
        assert mae([], []) == 0.0
        assert rmse([], []) == 0.0


class TestComputeReport:
    def test_structure(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]
        report = compute_report("test", y_true, y_pred, label_names={0: "a", 1: "b", 2: "c"})
        assert report.name == "test"
        assert 0.0 <= report.accuracy <= 1.0
        assert len(report.per_class) == 3
        assert len(report.confusion_matrix) == 3
        assert report.n_samples == 6

    def test_perfect_accuracy(self):
        y = list(range(10))
        report = compute_report("perfect", y, y)
        assert report.accuracy == pytest.approx(1.0)
        assert report.macro_f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ablation  (requires real test split cache + trained checkpoint)
# ---------------------------------------------------------------------------

import pytest
from pathlib import Path

_TEST_SPLIT = Path("ml/datasets/processed/eval_test_split.npz")
_CHECKPOINT  = Path("ml/models/weights/tcmt_trained.pt")
_REAL_EVAL_AVAILABLE = _TEST_SPLIT.exists() and _CHECKPOINT.exists()


class TestAblation:
    """Ablation tests over the real held-out test split.

    All tests skip gracefully when the data cache or checkpoint is absent
    (e.g. CI environments without ML weights).  Run
    ``python scripts/cache_eval_test_split.py`` to generate the cache.
    """

    @pytest.mark.skipif(not _REAL_EVAL_AVAILABLE,
                        reason="Real test split or checkpoint not found")
    def test_result_count(self):
        study = run_ablation(n_samples=100, seed=42, modality_subset=["facial", "audio"])
        # 2^2 - 1 = 3 non-empty subsets
        assert len(study.results) == 3

    @pytest.mark.skipif(not _REAL_EVAL_AVAILABLE,
                        reason="Real test split or checkpoint not found")
    def test_study_has_n_samples_and_seed(self):
        """AblationStudy must carry n_samples and seed for the endpoint schema."""
        study = run_ablation(n_samples=50, seed=42, modality_subset=["facial", "audio"])
        assert isinstance(study.n_samples, int) and study.n_samples > 0
        assert isinstance(study.seed, int)

    @pytest.mark.skipif(not _REAL_EVAL_AVAILABLE,
                        reason="Real test split or checkpoint not found")
    def test_dropped_modalities_complement(self):
        study = run_ablation(n_samples=100, seed=42, modality_subset=["facial", "audio"])
        for r in study.results:
            combined = set(r.active_modalities) | set(r.dropped_modalities)
            assert combined == {"facial", "audio"}

    @pytest.mark.skipif(not _REAL_EVAL_AVAILABLE,
                        reason="Real test split or checkpoint not found")
    def test_metrics_in_range(self):
        study = run_ablation(n_samples=100, seed=42, modality_subset=["facial", "audio"])
        for r in study.results:
            assert 0.0 <= r.report.accuracy <= 1.0
            assert 0.0 <= r.report.macro_f1 <= 1.0

    @pytest.mark.skipif(not _REAL_EVAL_AVAILABLE,
                        reason="Real test split or checkpoint not found")
    def test_fusion_beats_weakest(self):
        """Full fusion F1 should be comparable to or better than single-modality F1."""
        study = run_ablation(n_samples=500, seed=42, modality_subset=["facial", "audio"])
        full_result = next(
            r for r in study.results
            if set(r.active_modalities) == {"facial", "audio"}
        )
        single_results = [r for r in study.results if len(r.active_modalities) == 1]
        max_single = max(r.report.macro_f1 for r in single_results)
        assert full_result.report.macro_f1 >= max_single - 0.10

    def test_raises_when_cache_missing(self, tmp_path, monkeypatch):
        """Ensure RuntimeError (not a silent fallback) when cache is absent."""
        import app.evaluation.ablation as abl_mod
        monkeypatch.setattr(abl_mod, "TEST_SPLIT_PATH", tmp_path / "missing.npz")
        with pytest.raises(RuntimeError, match="Real evaluation test split not found"):
            run_ablation(n_samples=50, seed=42, modality_subset=["facial"])


# ---------------------------------------------------------------------------
# benchmark  (requires real test split cache + trained checkpoint)
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Benchmark tests over the real held-out test split.

    All tests skip gracefully when the data cache or checkpoint is absent.
    """

    @pytest.mark.skipif(not _REAL_EVAL_AVAILABLE,
                        reason="Real test split or checkpoint not found")
    def test_report_count(self):
        reports = run_benchmark(n_samples=100, seed=42)
        # 4 modalities + 1 fusion
        assert len(reports) == len(MODALITIES) + 1

    @pytest.mark.skipif(not _REAL_EVAL_AVAILABLE,
                        reason="Real test split or checkpoint not found")
    def test_fusion_exists(self):
        reports = run_benchmark(n_samples=100, seed=42)
        names = [r.name for r in reports]
        assert "fusion" in names

    @pytest.mark.skipif(not _REAL_EVAL_AVAILABLE,
                        reason="Real test split or checkpoint not found")
    def test_metrics_in_range(self):
        reports = run_benchmark(n_samples=100, seed=42)
        for r in reports:
            assert 0.0 <= r.accuracy <= 1.0
            assert 0.0 <= r.macro_f1 <= 1.0
            assert 0.0 <= r.weighted_f1 <= 1.0

    @pytest.mark.skipif(not _REAL_EVAL_AVAILABLE,
                        reason="Real test split or checkpoint not found")
    def test_fusion_beats_weakest(self):
        reports = run_benchmark(n_samples=100, seed=42)
        fusion = next(r for r in reports if r.name == "fusion")
        weakest = min((r for r in reports if r.name != "fusion"),
                      key=lambda r: r.macro_f1)
        assert fusion.macro_f1 >= weakest.macro_f1

    def test_raises_when_cache_missing(self, tmp_path, monkeypatch):
        """Ensure RuntimeError (not a silent fallback) when cache is absent."""
        import app.evaluation.benchmark as bm_mod
        monkeypatch.setattr(bm_mod, "TEST_SPLIT_PATH", tmp_path / "missing.npz")
        with pytest.raises(RuntimeError, match="Real evaluation test split not found"):
            run_benchmark(n_samples=50, seed=42)
