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
        from loguru import logger
        logger.info("UNMISTAKABLE LOG: CameraCapture constructed with device_id={}, fps={}", device_id, fps)
        self.device_id = device_id
        self.fps = fps
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None   # cv2.VideoCapture — imported lazily

    # ------------------------------------------------------------------
    def start(self) -> "CameraCapture":
        from loguru import logger
        logger.info("UNMISTAKABLE LOG: CameraCapture.start() called! Starting daemon thread.")
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
        from loguru import logger
        logger.info("UNMISTAKABLE LOG: CameraCapture _capture_loop() entry!")
        logger.info("CameraCapture thread started.")
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            logger.error("CameraCapture failed: cv2 not installed ({})", exc)
            return

        interval = 1.0 / self.fps
        
        while not self._stop_event.is_set():
            logger.info("CameraCapture scanning for video devices...")
            self._cap = None
            # Prioritize configured device_id, then fall back to indices 0-3
            devices_to_try = [self.device_id] + [i for i in range(4) if i != self.device_id]
            # Remove duplicates while preserving order
            devices_to_try = list(dict.fromkeys(devices_to_try))

            for dev_idx in devices_to_try:
                logger.info("CameraCapture: testing device index {}...", dev_idx)
                cap = cv2.VideoCapture(dev_idx)
                
                if not cap.isOpened():
                    logger.info("CameraCapture: device {} isOpened() returned False, skipping.", dev_idx)
                    cap.release()
                    continue

                # Read a few frames to ensure the device is truly active and not a dummy
                valid_device = False
                frame_shape = None
                for frame_idx in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        valid_device = True
                        frame_shape = frame.shape
                    else:
                        is_valid = frame is not None and frame.size > 0
                        logger.warning(
                            "CameraCapture: device {} read() failed on test frame {} (ret={}, valid_frame={})",
                            dev_idx, frame_idx, ret, is_valid
                        )
                        valid_device = False
                        break
                    time.sleep(0.05)
                    
                if valid_device:
                    self._cap = cap
                    self.device_id = dev_idx
                    if dev_idx == 1:
                        logger.info("CameraCapture: CLEAR LOG - successfully selected device 1! Real frames are flowing. frame shape={}", frame_shape)
                    else:
                        logger.info(
                            "CameraCapture: successfully selected device {}, frame shape={}",
                            dev_idx, frame_shape,
                        )
                    self._queue.put_nowait(frame)
                    break
                else:
                    logger.warning("CameraCapture: device {} failed frame validation, skipping virtual/non-reading device.", dev_idx)
                    cap.release()

            if self._cap is None:
                logger.error("CameraCapture failed: no usable video device found (tried {})", devices_to_try)
                time.sleep(2.0)
                continue

            frame_count = 0
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Drop oldest if consumer is slow
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                    self._queue.put_nowait(frame)

                    frame_count += 1
                    if frame_count % 150 == 0:  # every ~10 s at 15 fps
                        logger.info(
                            "CameraCapture: device={} frames_captured={} shape={}",
                            self.device_id, frame_count, frame.shape,
                        )
                else:
                    logger.warning("CameraCapture: cap.read() returned False on device {}", self.device_id)
                    if self._cap:
                        self._cap.release()
                    self._cap = None
                    # Move to the next device to stop retrying the failed one
                    self.device_id = (self.device_id + 1) % 4
                    break
                    
                elapsed = time.monotonic() - t0
                sleep = max(0.0, interval - elapsed)
                time.sleep(sleep)
