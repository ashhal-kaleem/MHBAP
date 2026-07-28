"""
HCI event listener — captures mouse and keyboard events via pynput.
Stores raw events in a thread-safe ring buffer; the HCI pipeline
drains them every tick to compute behavioural features.

Privacy note: key *content* is never stored — only key-press timing
and categorical labels (letter / digit / special / modifier).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple


@dataclass
class MouseEvent:
    t: float          # unix timestamp
    x: int
    y: int
    event: str        # "move" | "click_down" | "click_up" | "scroll"
    button: str = ""
    scroll_dy: int = 0


@dataclass
class KeyEvent:
    t: float
    event: str        # "down" | "up"
    category: str     # "letter" | "digit" | "special" | "modifier"


class HCIListener:
    """
    Non-blocking HCI event collector.

    Usage
    -----
    listener = HCIListener(buffer_size=500)
    listener.start()
    mouse_events, key_events = listener.drain()
    listener.stop()
    """

    def __init__(self, buffer_size: int = 500) -> None:
        self._mouse_buf: Deque[MouseEvent] = deque(maxlen=buffer_size)
        self._key_buf: Deque[KeyEvent] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._mouse_listener = None
        self._key_listener = None
        self._last_mouse_pos: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    def start(self) -> "HCIListener":
        try:
            from pynput import mouse, keyboard  # type: ignore
        except ImportError:
            return self   # pynput not installed — HCI silently unavailable

        self._mouse_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._key_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._mouse_listener.start()
        self._key_listener.start()
        return self

    def stop(self) -> None:
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._key_listener:
            self._key_listener.stop()

    def drain(self) -> Tuple[list, list]:
        """Return and clear accumulated events since last call."""
        with self._lock:
            m = list(self._mouse_buf)
            k = list(self._key_buf)
            self._mouse_buf.clear()
            self._key_buf.clear()
        return m, k

    # ------------------------------------------------------------------
    def _on_move(self, x: int, y: int) -> None:
        with self._lock:
            self._mouse_buf.append(MouseEvent(t=time.time(), x=x, y=y, event="move"))
        self._last_mouse_pos = (x, y)

    def _on_click(self, x: int, y: int, button: object, pressed: bool) -> None:
        evt = "click_down" if pressed else "click_up"
        with self._lock:
            self._mouse_buf.append(
                MouseEvent(t=time.time(), x=x, y=y, event=evt, button=str(button))
            )

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        with self._lock:
            self._mouse_buf.append(
                MouseEvent(t=time.time(), x=x, y=y, event="scroll", scroll_dy=dy)
            )

    @staticmethod
    def _categorise_key(key: object) -> str:
        try:
            from pynput.keyboard import Key  # type: ignore
            if key in (Key.shift, Key.ctrl, Key.alt, Key.cmd,
                       Key.shift_r, Key.ctrl_r, Key.alt_r):
                return "modifier"
            c = key.char
            if c is None:
                return "special"
            return "letter" if c.isalpha() else ("digit" if c.isdigit() else "special")
        except AttributeError:
            return "special"

    def _on_key_press(self, key: object) -> None:
        with self._lock:
            self._key_buf.append(
                KeyEvent(t=time.time(), event="down", category=self._categorise_key(key))
            )

    def _on_key_release(self, key: object) -> None:
        with self._lock:
            self._key_buf.append(
                KeyEvent(t=time.time(), event="up", category=self._categorise_key(key))
            )
