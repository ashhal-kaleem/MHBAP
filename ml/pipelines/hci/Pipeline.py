"""
HCI pipeline — mouse + keyboard event buffer → interaction features.

Input: list of raw events from hci_listener (dicts with 'type', 'ts', ...)
Output: dict of 10 floats over the analysis window

Features
--------
mouse_speed          : avg cursor speed px/s normalised [0, 1]
mouse_acceleration   : avg acceleration [0, 1]
click_rate           : clicks/sec [0, 1]
scroll_intensity     : scroll wheel activity [0, 1]
keystroke_rate       : keystrokes/sec [0, 1]
dwell_time           : mean pause between keystrokes (s) normalised [0, 1]
error_rate_proxy     : backspace ratio [0, 1]
typing_rhythm_std    : std of inter-key intervals normalised [0, 1]
mouse_pause_ratio    : fraction of window with zero movement [0, 1]
interaction_entropy  : Shannon entropy of event-type distribution [0, 1]
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any

import numpy as np

from ml.pipelines.Base import BasePipeline


class HCIPipeline(BasePipeline):
    MODALITY = "hci"

    def process(self, events: List[dict] | Tuple[List[Any], List[Any]]) -> Dict[str, float]:
        if isinstance(events, tuple):
            mouse_events, key_events = events
            converted = []

            for e in mouse_events:
                converted.append({
                    "type": (
                        "mouse_move" if e.event == "move"
                        else "mouse_click" if "click" in e.event
                        else "mouse_scroll"
                    ),
                    "ts": e.t,
                    "x": e.x,
                    "y": e.y,
                    "dy": e.scroll_dy,
                    "button": e.button,
                })

            for e in key_events:
                converted.append({
                    "type": "key_press",
                    "ts": e.t,
                    "key": e.category,
                })

            events = converted
            events.sort(key=lambda e: e["ts"])
        zeros = {k: 0.0 for k in [
            "mouse_speed", "mouse_acceleration", "click_rate",
            "scroll_intensity", "keystroke_rate", "dwell_time",
            "error_rate_proxy", "typing_rhythm_std",
            "mouse_pause_ratio", "interaction_entropy",
        ]}
        if not events:
            return zeros

        t0 = float(events[0]["ts"])
        t1 = float(events[-1]["ts"])
        window = max(t1 - t0, 1e-3)

        mouse_moves = [e for e in events if e["type"] == "mouse_move"]
        clicks = [e for e in events if e["type"] == "mouse_click"]
        scrolls = [e for e in events if e["type"] == "mouse_scroll"]
        keystrokes = [e for e in events if e["type"] == "key_press"]
        backspaces = [e for e in keystrokes if e.get("key") in ("backspace", "Key.backspace")]

        # Mouse speed & acceleration
        speeds = []
        for i in range(1, len(mouse_moves)):
            dt = max(float(mouse_moves[i]["ts"]) - float(mouse_moves[i - 1]["ts"]), 1e-4,)
            dx = float(mouse_moves[i]["x"]) - float(mouse_moves[i - 1]["x"])
            dy = float(mouse_moves[i]["y"]) - float(mouse_moves[i - 1]["y"])
            speeds.append(math.hypot(dx, dy) / dt)

        mouse_speed = float(np.clip(np.mean(speeds) / 2000.0, 0.0, 1.0)) if speeds else 0.0

        accels = [abs(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]
        mouse_acceleration = float(np.clip(np.mean(accels) / 5000.0, 0.0, 1.0)) if accels else 0.0

        # Click & scroll rates
        click_rate = float(np.clip(len(clicks) / window, 0.0, 1.0))
        scroll_intensity = float(np.clip(
            sum(abs(float(e.get("dy", 0))) for e in scrolls) / (window * 20), 0.0, 1.0))

        # Keystroke dynamics
        keystroke_rate = float(np.clip(len(keystrokes) / window / 5.0, 0.0, 1.0))

        iki = []
        for i in range(1, len(keystrokes)):
            iki.append(float(keystrokes[i]["ts"]) - float(keystrokes[i - 1]["ts"]))

        dwell_time = float(np.clip(np.mean(iki) / 2.0, 0.0, 1.0)) if iki else 0.0
        typing_rhythm_std = float(np.clip(np.std(iki) / 1.0, 0.0, 1.0)) if iki else 0.0
        error_rate_proxy = (len(backspaces) / max(len(keystrokes), 1))

        # Mouse pause ratio
        if mouse_moves:
            pause_count = sum(1 for i in range(1, len(mouse_moves))
                              if (float(mouse_moves[i]["ts"]) - float(mouse_moves[i - 1]["ts"])) > 0.5)
            mouse_pause_ratio = float(pause_count / max(len(mouse_moves) - 1, 1))
        else:
            # No events this tick → unknown, not "fully paused"; use 0.0 so the
            # TCMT feature vector is neutral rather than saturated at 1.0.
            mouse_pause_ratio = 0.0

        # Interaction entropy
        type_counts = {}
        for e in events:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
        total = sum(type_counts.values())
        probs = [c / total for c in type_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        interaction_entropy = float(np.clip(entropy / 3.0, 0.0, 1.0))  # max ~3 bits for 8 types

        return {
            "mouse_speed": mouse_speed,
            "mouse_acceleration": mouse_acceleration,
            "click_rate": click_rate,
            "scroll_intensity": scroll_intensity,
            "keystroke_rate": keystroke_rate,
            "dwell_time": dwell_time,
            "error_rate_proxy": error_rate_proxy,
            "typing_rhythm_std": typing_rhythm_std,
            "mouse_pause_ratio": mouse_pause_ratio,
            "interaction_entropy": interaction_entropy,
        }

    def warm_up(self) -> None:
        self._loaded = True

    def extract(self, data):  # type: ignore[override]
        return self.process(data)
