"""
Camera capture thread — wraps OpenCV VideoCapture in a daemon thread
so the main loop can call `get_frame()` without blocking.

Phase 4 scope: raw BGR frames delivered via a thread-safe queue.
Phase 5 pipelines consume frames from that queue.

Startup optimisation (Fix A):
  - The last working device index is persisted in a process-level variable
    (_PREFERRED_DEVICE_IDX) so that subsequent CameraCapture instances
    (e.g. after stop/start) skip the full 0-3 scan and go straight to the
    known-good index.  0-3 fallback still runs if that index fails.
  - Validation uses a single frame probe (not 5) to minimise open latency.
  - On first run the scan prefers the configured device_id; on re-scan after
    a disconnect the preferred index is tried first before falling back.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np

# Process-level cache of the last working camera index.
# Written by _capture_loop on first successful open; reset to None when a
# device dies so the next iteration tries all indices fresh.
_PREFERRED_DEVICE_IDX: Optional[int] = None


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
        from loguru import logger
        logger.info("CameraCapture constructed: device_id={}, fps={}", device_id, fps)
        self.device_id = device_id
        self.fps = fps
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None   # cv2.VideoCapture — imported lazily

    # ------------------------------------------------------------------
    def start(self) -> "CameraCapture":
        from loguru import logger
        logger.info("CameraCapture.start() — spawning capture thread")
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
    @staticmethod
    def _try_open_device(dev_idx: int):
        """
        Open a device and validate it with a single frame read.
        Returns (cap, frame) on success, (None, None) on failure.
        The returned cap is already open; caller owns it.
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            return None, None

        cap = cv2.VideoCapture(dev_idx)
        if not cap.isOpened():
            cap.release()
            return None, None

        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return cap, frame

        cap.release()
        return None, None

    def _capture_loop(self) -> None:
        global _PREFERRED_DEVICE_IDX
        from loguru import logger
        logger.info("CameraCapture: _capture_loop started")

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            logger.error("CameraCapture: cv2 not installed ({})", exc)
            return

        interval = 1.0 / self.fps
        _MAX_READ_FAILURES = 10

        while not self._stop_event.is_set():
            self._cap = None
            t_scan_start = time.monotonic()

            # ── Build probe order ─────────────────────────────────────
            # 1. Process-level preferred index (last known-good, fastest path)
            # 2. Configured device_id (constructor argument)
            # 3. Remaining indices 0-3 as fallback
            seen: list[int] = []
            preferred = _PREFERRED_DEVICE_IDX
            if preferred is not None:
                seen.append(preferred)
            if self.device_id not in seen:
                seen.append(self.device_id)
            for i in range(4):
                if i not in seen:
                    seen.append(i)
            # ─────────────────────────────────────────────────────────

            logger.info(
                "CameraCapture: scan order={} (preferred={})",
                seen, preferred,
            )

            probe_frame = None
            for dev_idx in seen:
                if self._stop_event.is_set():
                    break
                logger.info("CameraCapture: probing device {}...", dev_idx)
                cap, frame = self._try_open_device(dev_idx)
                if cap is None:
                    logger.info("CameraCapture: device {} unusable, skipping", dev_idx)
                    continue

                self._cap = cap
                self.device_id = dev_idx
                _PREFERRED_DEVICE_IDX = dev_idx
                probe_frame = frame
                t_scan = time.monotonic() - t_scan_start
                logger.info(
                    "CameraCapture: device {} selected in {:.3f}s (shape={})",
                    dev_idx, t_scan, frame.shape,
                )
                break

            if self._cap is None:
                logger.error(
                    "CameraCapture: no usable device found (tried {}), retrying in 2s",
                    seen,
                )
                time.sleep(2.0)
                continue

            # Enqueue the probe frame immediately so the first tick isn't starved
            if probe_frame is not None:
                try:
                    self._queue.put_nowait(probe_frame)
                except queue.Full:
                    pass

            # ── Steady-state read loop ────────────────────────────────
            frame_count = 0
            consecutive_failures = 0
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                ret, frame = self._cap.read()

                if ret and frame is not None and frame.size > 0:
                    consecutive_failures = 0
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._queue.put_nowait(frame)
                    frame_count += 1
                    if frame_count % 150 == 0:
                        logger.info(
                            "CameraCapture: device={} frames={} shape={}",
                            self.device_id, frame_count, frame.shape,
                        )
                else:
                    consecutive_failures += 1
                    if consecutive_failures < _MAX_READ_FAILURES:
                        elapsed = time.monotonic() - t0
                        time.sleep(max(0.0, interval - elapsed))
                        continue
                    logger.warning(
                        "CameraCapture: device {} — {} consecutive failures, re-scanning",
                        self.device_id, consecutive_failures,
                    )
                    if self._cap:
                        self._cap.release()
                    self._cap = None
                    # Invalidate preferred cache so next scan tries all indices
                    _PREFERRED_DEVICE_IDX = None
                    break

                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, interval - elapsed))
