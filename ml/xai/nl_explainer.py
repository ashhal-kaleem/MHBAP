"""
nl_explainer.py — rule-based natural-language explanation generator.

Converts a PredictionResult + SHAP modality weights into a 1-3 sentence
human-readable explanation shown in the dashboard XAI panel.

Design goals:
  - No LLM dependency — runs offline, deterministic, always fast
  - Multi-head aware (stress / engagement / attention / fatigue)
  - Confidence-qualified language ("clearly", "somewhat", "borderline")
  - Contextualises modality contributions naturally
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from ml.fusion.predictor import PredictionResult


# ── thresholds ────────────────────────────────────────────────────────────────

def _band(value: float, bands: List[Tuple[float, str]]) -> str:
    for threshold, label in bands:
        if value <= threshold:
            return label
    return bands[-1][1]


def _confidence(value: float, low: float, high: float) -> str:
    """Return an adverb based on distance from decision boundaries."""
    mid = (low + high) / 2
    margin = min(abs(value - low), abs(value - high))
    if margin < 0.05:
        return "borderline"
    if value < mid:
        return "somewhat"
    return "clearly"


_STRESS_BANDS:      List[Tuple[float, str]] = [(0.25, "low"), (0.55, "moderate"), (0.80, "high"), (1.0, "very high")]
_ENGAGE_BANDS:      List[Tuple[float, str]] = [(0.30, "low"), (0.60, "moderate"), (0.85, "high"), (1.0, "very high")]
_ATTENTION_BANDS:   List[Tuple[float, str]] = [(0.30, "low"), (0.60, "moderate"), (0.85, "high"), (1.0, "very high")]
_FATIGUE_BANDS:     List[Tuple[float, str]] = [(0.25, "low"), (0.55, "moderate"), (0.80, "high"), (1.0, "very high")]

_MODALITY_PHRASES: Dict[str, str] = {
    "face":  "facial expression dynamics",
    "gaze":  "gaze direction and blink patterns",
    "pose":  "body posture and movement",
    "voice": "vocal prosody and speech energy",
    "hci":   "keyboard and mouse interaction dynamics",
}

_EMOTION_ADJECTIVES: Dict[str, str] = {
    "happy":     "positive",
    "sad":       "subdued",
    "angry":     "agitated",
    "fearful":   "anxious",
    "disgusted": "aversive",
    "surprised": "alert",
    "neutral":   "neutral",
    "confused":  "uncertain",
}


def _top_two(shap: Dict[str, float]) -> Tuple[Optional[str], Optional[str]]:
    if not shap:
        return None, None
    ranked = sorted(shap.items(), key=lambda x: x[1], reverse=True)
    top = ranked[0][0] if ranked else None
    second = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0.10 else None
    return top, second


def generate_explanation(
    prediction: PredictionResult,
    shap_weights: Optional[Dict[str, float]] = None,
    head: str = "stress",
) -> str:
    """
    Return a 1-3 sentence natural-language explanation for a single prediction.

    Parameters
    ----------
    prediction : PredictionResult
    shap_weights : flat {modality: weight} dict for the requested head
    head : which output head to foreground ('stress', 'engagement', 'attention', 'fatigue')
    """
    s_lv  = _band(prediction.stress,     _STRESS_BANDS)
    e_lv  = _band(prediction.engagement, _ENGAGE_BANDS)
    a_lv  = _band(prediction.attention,  _ATTENTION_BANDS)
    f_lv  = _band(prediction.fatigue,    _FATIGUE_BANDS)
    emo   = prediction.emotion
    emo_adj = _EMOTION_ADJECTIVES.get(emo, "")

    # ── sentence 1: overall state ────────────────────────────────────────────
    emo_str = f" with a {emo_adj} affect" if emo_adj and emo_adj != "neutral" else ""
    s1 = (
        f"The system detected a {emo} emotional state{emo_str}. "
        f"Current readings: stress {s_lv} ({prediction.stress:.0%}), "
        f"engagement {e_lv} ({prediction.engagement:.0%}), "
        f"attention {a_lv} ({prediction.attention:.0%}), "
        f"fatigue {f_lv} ({prediction.fatigue:.0%})."
    )

    if not shap_weights:
        return s1

    top_mod, second_mod = _top_two(shap_weights)
    if top_mod is None:
        return s1

    top_phrase  = _MODALITY_PHRASES.get(top_mod,  top_mod)
    top_pct     = int(shap_weights.get(top_mod, 0) * 100)
    head_label  = head.replace("_", " ")

    # ── sentence 2: primary driver ───────────────────────────────────────────
    s2 = (
        f"The {head_label} estimate is primarily driven by {top_phrase} "
        f"({top_pct}% of the model attribution)."
    )

    # ── sentence 3: secondary driver or interpretive note ────────────────────
    s3 = ""
    if second_mod:
        sec_phrase = _MODALITY_PHRASES.get(second_mod, second_mod)
        sec_pct    = int(shap_weights.get(second_mod, 0) * 100)
        s3 = f"Secondary contribution from {sec_phrase} ({sec_pct}%)."
    elif prediction.stress > 0.75 and head == "stress":
        s3 = "Elevated stress at this level may benefit from a short break."
    elif prediction.fatigue > 0.70 and head == "fatigue":
        s3 = "High fatigue detected — performance may be impaired."
    elif prediction.engagement < 0.30 and head == "engagement":
        s3 = "Low engagement may indicate distraction or loss of interest."

    parts = [s1, s2]
    if s3:
        parts.append(s3)
    return " ".join(parts)
