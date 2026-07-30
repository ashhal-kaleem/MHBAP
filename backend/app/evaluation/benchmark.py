"""
benchmark.py — Real TCMT benchmark (Phase F).

Replaces the random-simulation approach with actual model inference
on a held-out synthetic test split. Each "modality" ablation zeros out
that modality's feature slice before passing through TCMT.

Falls back gracefully to the original simulation if weights not found.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import numpy as np

from backend.app.evaluation.metrics import EvaluationReport, compute_report

WEIGHT_PATH = Path(__file__).parent.parent.parent.parent / \
    "ml" / "models" / "weights" / "tcmt_trained.pt"

MODALITIES = ["facial", "audio", "physiological", "hci"]

# Map benchmark modality names → TCMT feature-vector modality keys
_MOD_MAP = {
    "facial":        ["face", "gaze"],
    "audio":         ["voice"],
    "physiological": ["pose"],
    "hci":           ["hci"],
}

EMOTION_LABELS = {0: "neutral", 1: "happy", 2: "sad", 3: "angry"}


def _load_model():
    """Load trained TCMT; return None if weights missing."""
    try:
        import torch
        from ml.fusion.tcmt import TCMT
        if not WEIGHT_PATH.exists():
            return None
        ckpt  = torch.load(str(WEIGHT_PATH), map_location="cpu")
        model = TCMT()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
    except Exception:
        return None


def _get_test_data(n_samples: int = 1000, seed: int = 42):
    """Return test split from deterministic dataset."""
    from ml.training.dataset import make_dataset
    _, _, test = make_dataset(n_samples=max(n_samples * 7, 3000), seed=seed)
    # Trim to n_samples
    return {k: v[:n_samples] for k, v in test.items()}


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
    logits = np.array(out["emotion_logits"])   # (N, 8) or (N, 4)
    # Map 8-class logits to 4-class for EMOTION_LABELS
    n_cls = logits.shape[1]
    if n_cls > 4:
        logits = logits[:, :4]   # take first 4 classes
    return logits.argmax(axis=1)


def run_benchmark(
    n_samples: int = 1000,
    seed: int = 42,
    modalities: Optional[List[str]] = None,
) -> List[EvaluationReport]:
    """
    Evaluate each modality independently plus the full fusion model.
    Uses real TCMT inference. Falls back to simulation if no weights.
    """
    active = modalities or MODALITIES
    model  = _load_model()

    if model is None:
        # Graceful fallback — original simulation
        from backend.app.evaluation.benchmark import run_benchmark as _sim
        return _sim(n_samples=n_samples, seed=seed, modalities=modalities)

    data   = _get_test_data(n_samples=n_samples, seed=seed)
    X      = data["X"]
    y_true = (data["emotion"] % 4).tolist()   # clip to 4-class

    reports: List[EvaluationReport] = []

    # Per-modality: zero out ALL OTHER modalities
    for mod in active:
        keep = _MOD_MAP.get(mod, [mod])
        all_mods = [m for mlist in _MOD_MAP.values() for m in mlist]
        mask_out = [m for m in all_mods if m not in keep]
        y_pred   = _infer_with_mask(model, X, mask_out).tolist()
        reports.append(compute_report(
            name=mod, y_true=y_true, y_pred=y_pred,
            label_names=EMOTION_LABELS,
        ))

    # Full fusion (no masking)
    y_pred_full = _infer_with_mask(model, X, []).tolist()
    reports.append(compute_report(
        name="fusion", y_true=y_true, y_pred=y_pred_full,
        label_names=EMOTION_LABELS,
    ))

    return reports
