"""
Unit tests for ml/DataWriter.py — no backend DB, no event loop flush.

These tests exercise only the queue-management half of DataWriter
(write / put_nowait / QueueFull). The _flush_loop is never started,
so no app.db imports are needed.
"""
from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from uuid import uuid4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid():
    return uuid4()


def _now():
    return datetime.now(timezone.utc)


def _run(coro):
    """Run a coroutine synchronously (Python 3.9-compatible)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# DataWriter tests
# ---------------------------------------------------------------------------

class TestDataWriterImport(unittest.TestCase):
    """DataWriter must be importable without the backend DB stack."""

    def test_import_succeeds(self):
        from ml.DataWriter import DataWriter  # must not raise
        self.assertTrue(callable(DataWriter))

    def test_instantiation_succeeds(self):
        from ml.DataWriter import DataWriter
        writer = DataWriter(queue_size=4)
        self.assertEqual(writer._queue.maxsize, 4)
        self.assertIsNone(writer._task)


class TestDataWriterWrite(unittest.TestCase):
    """test_write_forbidden: a full queue silently drops without raising."""

    def test_write_enqueues_item(self):
        """A write to an empty queue adds exactly one item."""
        from ml.DataWriter import DataWriter
        writer = DataWriter(queue_size=8)
        _run(writer.write(_uuid(), "face", {"au_jaw_drop": 0.3}, _now()))
        self.assertEqual(writer._queue.qsize(), 1)

    def test_write_stores_correct_keys(self):
        """Enqueued dict must contain session_id, modality, features, timestamp."""
        from ml.DataWriter import DataWriter
        writer = DataWriter(queue_size=8)
        sid = _uuid()
        ts = _now()
        feats = {"au_jaw_drop": 0.3, "au_lip_corner": 0.1}
        _run(writer.write(sid, "face", feats, ts))
        item = writer._queue.get_nowait()
        self.assertEqual(item["session_id"], sid)
        self.assertEqual(item["modality"], "face")
        self.assertEqual(item["features"], feats)
        self.assertEqual(item["timestamp"], ts)

    def test_write_forbidden(self):
        """
        Writing to a full queue must be silently dropped - no exception raised.

        This is the forbidden write contract: when the in-memory queue is
        at capacity, DataWriter.write() swallows asyncio.QueueFull and logs a
        warning instead of propagating the error to the caller.
        """
        from ml.DataWriter import DataWriter
        writer = DataWriter(queue_size=1)

        # Fill the queue to capacity
        _run(writer.write(_uuid(), "face", {}, _now()))
        self.assertEqual(writer._queue.qsize(), 1)

        # This second write must not raise - the forbidden write is silently dropped
        try:
            _run(writer.write(_uuid(), "face", {"extra": 1.0}, _now()))
        except Exception as exc:
            self.fail(
                f"DataWriter.write() raised {type(exc).__name__} on full queue: {exc}"
            )

        # Queue size must remain at 1 (the extra item was dropped, not enqueued)
        self.assertEqual(writer._queue.qsize(), 1)

    def test_write_forbidden_logs_warning(self):
        """
        A full-queue drop must call logger.warning with a 'queue full' message.

        Uses mock.patch.object on the stdlib logger directly so the assertion
        is independent of logging handlers, level configuration, loguru shims,
        or pytest log-capture plugins — any of which can silently swallow
        records when assertLogs() is used with a logger object.
        """
        from unittest.mock import patch
        from ml.DataWriter import DataWriter
        import ml.DataWriter as dw_module

        writer = DataWriter(queue_size=1)
        _run(writer.write(_uuid(), "face", {}, _now()))  # fill queue to capacity

        with patch.object(dw_module.logger, "warning") as mock_warn:
            _run(writer.write(_uuid(), "voice", {}, _now()))  # must drop + warn

        self.assertTrue(
            mock_warn.called,
            "logger.warning was never called on a full-queue write",
        )
        # Verify the message text mentions queue fullness
        warn_args = " ".join(str(a) for a in mock_warn.call_args[0]).lower()
        self.assertIn(
            "queue full",
            warn_args,
            f"logger.warning called but message does not mention 'queue full': {mock_warn.call_args}",
        )

    def test_write_no_warning_when_queue_has_space(self):
        """
        logger.warning must NOT be called when the queue has room.

        Negative-path companion to test_write_forbidden_logs_warning:
        confirms the warning is triggered by fullness, not by every write.
        """
        from unittest.mock import patch
        from ml.DataWriter import DataWriter
        import ml.DataWriter as dw_module

        writer = DataWriter(queue_size=4)

        with patch.object(dw_module.logger, "warning") as mock_warn:
            _run(writer.write(_uuid(), "face", {}, _now()))
            _run(writer.write(_uuid(), "gaze", {}, _now()))

        self.assertFalse(
            mock_warn.called,
            f"logger.warning fired unexpectedly on a non-full queue: {mock_warn.call_args_list}",
        )

    def test_write_multiple_modalities_fit(self):
        """Multiple writes below capacity all land in the queue."""
        from ml.DataWriter import DataWriter
        writer = DataWriter(queue_size=5)
        modalities = ["face", "gaze", "pose", "voice", "hci"]
        for mod in modalities:
            _run(writer.write(_uuid(), mod, {}, _now()))
        self.assertEqual(writer._queue.qsize(), 5)

    def test_write_is_nonblocking(self):
        """write() must return immediately (put_nowait, not blocking put)."""
        from ml.DataWriter import DataWriter
        import time
        writer = DataWriter(queue_size=4)
        t0 = time.monotonic()
        for _ in range(4):
            _run(writer.write(_uuid(), "face", {}, _now()))
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.1)


class TestDataWriterLifecycle(unittest.TestCase):
    """start() and stop() must round-trip cleanly (no flush, no DB calls)."""

    def test_stop_before_start_is_safe(self):
        """stop() on a writer that was never started must not raise."""
        from ml.DataWriter import DataWriter
        writer = DataWriter()

        async def _go():
            await writer.stop()

        try:
            _run(_go())
        except Exception as exc:
            self.fail(f"stop() before start() raised: {exc}")

    def test_start_and_stop_roundtrip(self):
        """start() then stop() must cancel the background task cleanly."""
        from ml.DataWriter import DataWriter

        async def _go():
            writer = DataWriter(queue_size=4)
            await writer.start()
            self.assertIsNotNone(writer._task)
            await writer.stop()
            self.assertTrue(writer._task.done())

        _run(_go())


if __name__ == "__main__":
    unittest.main()
