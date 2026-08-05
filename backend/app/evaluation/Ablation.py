"""
ablation.py — Real TCMT modality ablation using the held-out real test split.

Data source
-----------
Loads ``ml/datasets/processed/eval_test_split.npz``, which is generated
by ``scripts/cache_eval_test_split.py``.

Raises ``RuntimeError`` explicitly when:
  - The cached test split is not found  (run scripts/cache_eval_test_split.py)
  - The TCMT checkpoint is not found    (run python -m ml.training.train_tcmt)

There is NO silent fallback to synthetic data or simulation.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from app.Evaluation.Metrics import EvaluationReport, compute_report

# ── paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent.parent

WEIGHT_PATH = _REPO_ROOT / "ml" / "models" / "weights" / "tcmt_trained.pt"
TEST_SPLIT_PATH = _REPO_ROOT / "ml" / "datasets" / "processed" / "eval_test_split.npz"

MODALITIES = ["facial", "audio", "physiological", "hci"]

EMOTION_LABELS = {0: "neutral", 1: "happy", 2: "sad", 3: "angry"}

_MOD_MAP = {
    "facial":        ["face", "gaze"],
    "audio":         ["voice"],
    "physiological": ["pose"],
    "hci":           ["hci"],
}


# ── dataclasses ─────────────────────────────────────────────────────────────────

@dataclass
class AblationResult:
    active_modalities: List[str]
    dropped_modalities: List[str]
    report: EvaluationReport


@dataclass
class AblationStudy:
    modality_subset: List[str]
    n_samples: int                          # populated by run_ablation()
    seed: int                               # populated by run_ablation()
    results: List[AblationResult] = field(default_factory=list)


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
        from ml.fusion.Tcmt import TCMT
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

def _infer(model, X: np.ndarray, mask_out: List[str]) -> list:
    import torch
    from ml.fusion.FeatureUtils import modality_slice
    Xm = X.copy()
    for mod in mask_out:
        s, e = modality_slice(mod)
        Xm[:, s:e] = 0.0
    with torch.no_grad():
        out = model(torch.tensor(Xm, dtype=torch.float32))
    logits = np.array(out["emotion_logits"])
    return logits.argmax(axis=1).tolist()


# ── public API ──────────────────────────────────────────────────────────────────

def run_ablation(
    n_samples: int = 500,
    seed: int = 42,
    modality_subset: Optional[List[str]] = None,
) -> AblationStudy:
    """
    Enumerate all non-empty subsets of ``modality_subset``, run TCMT on each
    with the remaining modalities masked to zero, and record the resulting F1.

    Parameters
    ----------
    n_samples:
        Maximum number of test samples to use (capped by the cached split size).
    seed:
        Informational only — the test split is already fixed by training seed=42.
    modality_subset:
        Which modalities to enumerate.  Defaults to all four.

    Returns
    -------
    AblationStudy with ``n_samples`` and ``seed`` populated and all subset results.

    Raises
    ------
    RuntimeError
        If the test split cache or checkpoint is missing.
    """
    subset = modality_subset or MODALITIES
    data = _load_real_test_data(n_samples)
    model = _load_model()

    X = data["X"]
    y_true = (data["emotion"] % 4).tolist()
    actual_n = len(X)

    all_feat_mods = [m for mlist in _MOD_MAP.values() for m in mlist]
    study = AblationStudy(modality_subset=subset, n_samples=actual_n, seed=seed)

    for r in range(1, len(subset) + 1):
        for combo in itertools.combinations(subset, r):
            active  = list(combo)
            dropped = [m for m in subset if m not in active]
            keep    = [fm for bm in active for fm in _MOD_MAP.get(bm, [bm])]
            mask    = [m for m in all_feat_mods if m not in keep]
            y_pred  = _infer(model, X, mask)
            report  = compute_report(
                name="+".join(active),
                y_true=y_true,
                y_pred=y_pred,
                label_names=EMOTION_LABELS,
            )
            study.results.append(AblationResult(
                active_modalities=active,
                dropped_modalities=dropped,
                report=report,
            ))

    return study
