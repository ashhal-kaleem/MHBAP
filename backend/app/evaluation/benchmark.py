"""
benchmark.py — Real TCMT benchmark using the held-out real test split.

Data source
-----------
Loads ``ml/datasets/processed/eval_test_split.npz``, which is generated
by ``scripts/cache_eval_test_split.py``.  This file contains the exact same
test split that was produced by ``make_real_dataset(seed=42)`` during training
(deterministic shuffle → same rows), so metrics computed here are directly
comparable to the values stored in ``tcmt_eval_metrics.json``.

Raises ``RuntimeError`` explicitly when:
  - The cached test split is not found  (run scripts/cache_eval_test_split.py)
  - The TCMT checkpoint is not found    (run python -m ml.training.train_tcmt)

There is NO silent fallback to synthetic data.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import numpy as np

from backend.app.evaluation.metrics import EvaluationReport, compute_report

# ── paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent.parent

WEIGHT_PATH = _REPO_ROOT / "ml" / "models" / "weights" / "tcmt_trained.pt"
TEST_SPLIT_PATH = _REPO_ROOT / "ml" / "datasets" / "processed" / "eval_test_split.npz"

MODALITIES = ["facial", "audio", "physiological", "hci"]

# Map benchmark modality names → TCMT feature-vector modality keys
_MOD_MAP = {
    "facial":        ["face", "gaze"],
    "audio":         ["voice"],
    "physiological": ["pose"],
    "hci":           ["hci"],
}

EMOTION_LABELS = {0: "neutral", 1: "happy", 2: "sad", 3: "angry"}


# ── data / model loading ────────────────────────────────────────────────────────

def _load_real_test_data(n_samples: int) -> dict:
    """
    Load the real held-out test split from the cached .npz file.

    Raises
    ------
    RuntimeError
        If the cache file does not exist.  Run ``python scripts/cache_eval_test_split.py``
        to generate it from the same HuggingFace datasets used during training.
    """
    if not TEST_SPLIT_PATH.exists():
        raise RuntimeError(
            f"Real evaluation test split not found at:\n  {TEST_SPLIT_PATH}\n\n"
            "Generate it by running:\n"
            "    python scripts/cache_eval_test_split.py\n\n"
            "This script downloads FER2013 + WESAD from HuggingFace (same as training), "
            "splits with seed=42, and saves the test portion to the path above."
        )
    data = dict(np.load(str(TEST_SPLIT_PATH)))
    # Trim to requested n_samples (cached split may be larger)
    n = min(n_samples, len(data["X"]))
    return {k: v[:n] for k, v in data.items()}


def _load_model():
    """
    Load the trained TCMT checkpoint.

    Raises
    ------
    RuntimeError
        If the checkpoint file does not exist.
    """
    try:
        import torch
        from ml.fusion.tcmt import TCMT
    except ImportError as exc:
        raise RuntimeError(f"PyTorch / TCMT not importable: {exc}") from exc

    if not WEIGHT_PATH.exists():
        raise RuntimeError(
            f"TCMT checkpoint not found at:\n  {WEIGHT_PATH}\n\n"
            "Train the model first:\n"
            "    python -m ml.training.train_tcmt"
        )

    ckpt = torch.load(str(WEIGHT_PATH), map_location="cpu")
    state_dict = (
        ckpt["state_dict"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt
        else ckpt
    )
    model = TCMT()
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── inference with modality masking ────────────────────────────────────────────

def _infer_with_mask(model, X: np.ndarray, masked_mods: List[str]) -> np.ndarray:
    """Run TCMT inference with selected modalities zeroed out."""
    import torch
    from ml.fusion.feature_utils import modality_slice

    Xm = X.copy()
    for mod in masked_mods:
        s, e = modality_slice(mod)
        Xm[:, s:e] = 0.0

    Xt = torch.tensor(Xm, dtype=torch.float32)
    with torch.no_grad():
        out = model(Xt)
    logits = np.array(out["emotion_logits"])   # (N, 4)
    return logits.argmax(axis=1)


# ── public API ──────────────────────────────────────────────────────────────────

def run_benchmark(
    n_samples: int = 1000,
    seed: int = 42,
    modalities: Optional[List[str]] = None,
) -> List[EvaluationReport]:
    """
    Evaluate each modality independently (masking all others) plus the full
    fusion model, using the real held-out test split.

    Parameters
    ----------
    n_samples:
        Maximum number of test samples to use (capped by the cached split size).
    seed:
        Informational only — the test split is already fixed by training seed=42.
    modalities:
        Which modality subsets to benchmark.  Defaults to all four.

    Returns
    -------
    List of EvaluationReport (one per modality + one for "fusion").

    Raises
    ------
    RuntimeError
        If the test split cache or checkpoint is missing.
    """
    active = modalities or MODALITIES
    data = _load_real_test_data(n_samples)
    model = _load_model()

    X = data["X"]
    y_true = (data["emotion"] % 4).tolist()   # always in {0,1,2,3}

    reports: List[EvaluationReport] = []

    # Per-modality: zero out ALL OTHER modalities, keep only this one
    for mod in active:
        keep = _MOD_MAP.get(mod, [mod])
        all_mods = [m for mlist in _MOD_MAP.values() for m in mlist]
        mask_out = [m for m in all_mods if m not in keep]
        y_pred = _infer_with_mask(model, X, mask_out).tolist()
        reports.append(compute_report(
            name=mod,
            y_true=y_true,
            y_pred=y_pred,
            label_names=EMOTION_LABELS,
        ))

    # Full fusion (no masking)
    y_pred_full = _infer_with_mask(model, X, []).tolist()
    reports.append(compute_report(
        name="fusion",
        y_true=y_true,
        y_pred=y_pred_full,
        label_names=EMOTION_LABELS,
    ))

    return reports
