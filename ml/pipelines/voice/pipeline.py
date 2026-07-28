"""
Voice pipeline — raw PCM audio chunk → prosodic + spectral features.

Input : 1-D float32 numpy array (mono, 16 kHz recommended)
Output: dict of 20 floats

Features
--------
mfcc_1 … mfcc_13   : first 13 MFCCs (mean over chunk)
pitch_mean          : mean F0 in Hz (0 if unvoiced)
pitch_std           : std of F0
energy              : RMS energy normalised [0, 1]
zcr                 : zero-crossing rate [0, 1]
spectral_centroid   : normalised [0, 1]
speaking_rate_proxy : voiced frames ratio [0, 1]
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ml.pipelines.base import BasePipeline

_SR = 16_000  # expected sample rate


class VoicePipeline(BasePipeline):
    MODALITY = "voice"

    def process(self, audio_chunk: Optional[np.ndarray]) -> Dict[str, float]:  # type: ignore[override]
        zeros: Dict[str, float] = {f"mfcc_{i}": 0.0 for i in range(1, 14)}
        zeros.update({
            "pitch_mean": 0.0, "pitch_std": 0.0,
            "energy": 0.0, "zcr": 0.0,
            "spectral_centroid": 0.0, "speaking_rate_proxy": 0.0,
        })

        if audio_chunk is None or len(audio_chunk) < 256:
            return zeros

        try:
            import librosa  # type: ignore
        except ImportError:
            return zeros

        y = audio_chunk.astype(np.float32)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=_SR, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1)

        # Pitch (pyin — probabilistic YIN)
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=_SR
        )
        voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean = float(voiced_f0.mean()) if len(voiced_f0) > 0 else 0.0
        pitch_std  = float(voiced_f0.std())  if len(voiced_f0) > 0 else 0.0

        # Energy
        rms = librosa.feature.rms(y=y)[0].mean()
        energy = float(np.clip(rms * 10, 0.0, 1.0))

        # ZCR
        zcr_val = float(librosa.feature.zero_crossing_rate(y)[0].mean())
        zcr = float(np.clip(zcr_val * 4, 0.0, 1.0))

        # Spectral centroid
        sc = librosa.feature.spectral_centroid(y=y, sr=_SR)[0].mean()
        spectral_centroid = float(np.clip(sc / 8000.0, 0.0, 1.0))

        # Speaking-rate proxy
        speaking_rate_proxy = (
            float(voiced_flag.mean()) if voiced_flag is not None and len(voiced_flag) > 0 else 0.0
        )

        features: Dict[str, float] = {f"mfcc_{i+1}": float(mfcc_means[i]) for i in range(13)}
        features.update({
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "energy": energy,
            "zcr": zcr,
            "spectral_centroid": spectral_centroid,
            "speaking_rate_proxy": speaking_rate_proxy,
        })
        return features

    def warm_up(self) -> None:
        self._loaded = True

    def extract(self, data):  # type: ignore[override]
        return self.process(data)
