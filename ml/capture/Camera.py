"""
Camera capture thread — wraps OpenCV VideoCapture in a daemon thread
so the main loop can call `get_frame()` without blocking.

Phase 4 scope: raw BGR frames delivered via a thread-safe queue.
Phase 5 pipelines consume frames from that queue.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np


class CameraCapture:
    """
    Non-blocking webcam reader.

    Usage
    -----
    cam = CameraCapture(device_id=0, fps=15)
    cam.start()
    frame = cam.get_frame()   # np.ndarray (H, W, 3) BGR or None
    cam.stop()
    """

    def __init__(self, device_id: int = 0, fps: int = 15, queue_size: int = 4) -> None:
        self.device_id = device_id
        self.fps = fps
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None   # cv2.VideoCapture — imported lazily

    # ------------------------------------------------------------------
    def start(self) -> "CameraCapture":
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Return latest frame or None if unavailable."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError:
            return   # cv2 not installed — camera silently unavailable

        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            return

        interval = 1.0 / self.fps
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            ret, frame = self._cap.read()
            if ret:
                # Drop oldest if consumer is slow
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                self._queue.put_nowait(frame)
            elapsed = time.monotonic() - t0
            sleep = max(0.0, interval - elapsed)
            time.sleep(sleep)
