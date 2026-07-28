"""
nl_explainer.py — rule-based natural-language explanation generator.

Converts a PredictionResult + SHAP modality weights into a 1-2 sentence
human-readable explanation shown in the dashboard XAI panel.

No LLM dependency — runs offline, deterministic, always fast.
"""
from __future__ import annotations
from typing import Dict, Optional

from ml.fusion.predictor import PredictionResult

_STRESS_LEVELS = [(3.0,"low"),(6.5,"moderate"),(10.0,"high")]
_ENGAGEMENT_LEVELS = [(0.35,"low"),(0.65,"moderate"),(1.0,"high")]
_ATTENTION_LEVELS  = [(0.35,"low"),(0.65,"moderate"),(1.0,"high")]
_FATIGUE_LEVELS    = [(0.35,"low"),(0.65,"moderate"),(1.0,"high")]


def _level(value: float, thresholds) -> str:
    for threshold, label in thresholds:
        if value <= threshold:
            return label
    return thresholds[-1][1]


def _top_modality(shap: Dict[str, float]) -> str:
    return max(shap, key=shap.get) if shap else "multimodal"


_MODALITY_PHRASES = {
    "face":  "facial expression cues",
    "gaze":  "gaze and blink patterns",
    "pose":  "body posture signals",
    "voice": "vocal prosody features",
    "hci":   "keyboard and mouse dynamics",
}


def generate_explanation(
    prediction: PredictionResult,
    shap_weights: Optional[Dict[str, float]] = None,
    head: str = "stress",
) -> str:
    """Return a 1-2 sentence natural-language explanation."""
    stress_lv     = _level(prediction.stress,     _STRESS_LEVELS)
    engagement_lv = _level(prediction.engagement, _ENGAGEMENT_LEVELS)
    attention_lv  = _level(prediction.attention,  _ATTENTION_LEVELS)
    fatigue_lv    = _level(prediction.fatigue,    _FATIGUE_LEVELS)
    emotion       = prediction.emotion

    top_mod = _top_modality(shap_weights or {})
    mod_phrase = _MODALITY_PHRASES.get(top_mod, "multimodal signals")

    sentences = [
        f"The user appears {emotion} with {stress_lv} stress "
        f"({prediction.stress:.1f}/10), "
        f"{engagement_lv} engagement, {attention_lv} attention, "
        f"and {fatigue_lv} fatigue."
    ]
    if shap_weights:
        pct = int((shap_weights.get(top_mod, 0.0)) * 100)
        sentences.append(
            f"The prediction is primarily driven by {mod_phrase} "
            f"({pct}% of the model's attribution)."
        )
    return " ".join(sentences)
