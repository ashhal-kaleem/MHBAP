"""
stream_bus.py — in-process pub/sub bus for WebSocket streaming.

Each session gets an asyncio.Queue per connected client.
SessionRunner pushes WsMessage dicts here; the WebSocket handler drains them.

No Redis required (Phase 9 will swap this for Redis pub/sub).
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Dict, List

# session_id (str) → list of per-client queues
_subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)


def subscribe(session_id: str, maxsize: int = 64) -> asyncio.Queue:
    """Register a new WebSocket client for a session. Returns its queue."""
    q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    _subscribers[session_id].append(q)
    return q


def unsubscribe(session_id: str, q: asyncio.Queue) -> None:
    """Remove a client queue when the WebSocket disconnects."""
    try:
        _subscribers[session_id].remove(q)
    except ValueError:
        pass
    if not _subscribers[session_id]:
        del _subscribers[session_id]


def publish(session_id: str, message: Any) -> None:
    """Push a message to all clients subscribed to this session (non-blocking)."""
    for q in list(_subscribers.get(session_id, [])):
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            pass   # slow consumer — drop frame rather than back-pressure
