"""
benchmark_camera_startup.py
Measures CameraCapture device-selection latency before and after the
preferred-index cache is populated.

Run from repo root:
    python scripts/benchmark_camera_startup.py

Reports:
  - Cold start: time from CameraCapture.start() until first non-None get_frame()
  - Warm start: same, using the cached _PREFERRED_DEVICE_IDX
  - Speedup factor
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import ml.capture.Camera as _cam_mod
from ml.capture.Camera import CameraCapture

_POLL_INTERVAL = 0.02   # 20 ms poll
_MAX_WAIT      = 15.0   # give up after 15 s


def measure_startup(label: str) -> float | None:
    """Start a CameraCapture and measure seconds until first real frame."""
    cam = CameraCapture(device_id=0, fps=15)
    t0 = time.monotonic()
    cam.start()
    deadline = t0 + _MAX_WAIT
    while time.monotonic() < deadline:
        f = cam.get_frame()
        if f is not None:
            elapsed = time.monotonic() - t0
            cam.stop()
            print(f"  [{label}] first frame in {elapsed:.3f}s  (preferred_idx={_cam_mod._PREFERRED_DEVICE_IDX})")
            return elapsed
        time.sleep(_POLL_INTERVAL)
    cam.stop()
    print(f"  [{label}] TIMEOUT — no frame in {_MAX_WAIT}s")
    return None


def main() -> None:
    print("\n=== Camera startup benchmark ===\n")

    # Ensure cache is clear
    _cam_mod._PREFERRED_DEVICE_IDX = None
    print("Cold start (no cached index):")
    cold = measure_startup("cold")

    print("\nWarm start (using cached index):")
    warm = measure_startup("warm")

    print()
    if cold is not None and warm is not None:
        speedup = cold / warm if warm > 0 else float("inf")
        print(f"  Cold: {cold:.3f}s")
        print(f"  Warm: {warm:.3f}s")
        print(f"  Speedup: {speedup:.1f}x")
        if warm < 1.0:
            print("  PASS — warm start < 1 s")
        else:
            print("  WARN — warm start still slow; check for slow device open")
    else:
        print("  Incomplete — check camera availability")


if __name__ == "__main__":
    main()
