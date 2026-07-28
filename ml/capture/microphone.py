"""
Microphone capture — streams audio chunks via sounddevice into a queue.
Chunk size and sample rate match librosa defaults used in Phase 5 voice pipeline.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

import numpy as np

SAMPLE_RATE = 16_000   # Hz — matches wav2vec2 / Whisper / opensmile defaults
CHUNK_MS = 250         # ms per callback chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


class MicrophoneCapture:
    """
    Non-blocking microphone reader.

    Usage
    -----
    mic = MicrophoneCapture(device=None, sample_rate=16000)
    mic.start()
    chunk = mic.get_chunk()   # np.ndarray (N,) float32 or None
    mic.stop()
    """

    def __init__(
        self,
        device: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        queue_size: int = 16,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._stream = None

    # ------------------------------------------------------------------
    def start(self) -> "MicrophoneCapture":
        self._stop_event.clear()
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            return self   # sounddevice not installed — mic silently unavailable

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_samples,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self) -> Optional[np.ndarray]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    # ------------------------------------------------------------------
    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if self._stop_event.is_set():
            return
        mono = indata[:, 0].copy()
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(mono)
