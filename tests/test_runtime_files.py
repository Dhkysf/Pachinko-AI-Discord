"""Offline unit tests for core/runtime.py (locks, stop request, atomic status, rotating logs)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import unittest
from logging.handlers import RotatingFileHandler
from pathlib import Path
from unittest.mock import patch

from core.runtime import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    acquire_lock,
    is_python_process,
    release_lock,
    setup_logging,
    watch_stop_request,
    write_status,
)


class TestRuntimeFiles(unittest.TestCase):
    def test_lock_create_refuse_and_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "run" / "bot.lock"
            ok, existing = acquire_lock(lock_path, started=1234, is_python_pid=lambda _pid: True)
            self.assertTrue(ok)
            self.assertIsNone(existing)
            data = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(data["pid"], os.getpid())
            self.assertEqual(data["started"], 1234)

            # Simulate another live python process holding the lock
            lock_path.write_text(json.dumps({"pid": 99999, "started": 1000}), encoding="utf-8")
            ok2, existing2 = acquire_lock(lock_path, is_python_pid=lambda pid: pid == 99999)
            self.assertFalse(ok2)
            self.assertEqual(existing2, 99999)

            # release_lock must not delete another PID's lock
            release_lock(lock_path)
            self.assertTrue(lock_path.exists())

            # release_lock deletes our own lock
            lock_path.write_text(json.dumps({"pid": os.getpid(), "started": 1000}), encoding="utf-8")
            release_lock(lock_path)
            self.assertFalse(lock_path.exists())

    def test_lock_replaces_stale_and_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "bot.lock"
            lock_path.write_text(json.dumps({"pid": 99999, "started": 1000}), encoding="utf-8")
            ok, existing = acquire_lock(lock_path, started=2000, is_python_pid=lambda _pid: False)
            self.assertTrue(ok)
            self.assertIsNone(existing)
            self.assertEqual(json.loads(lock_path.read_text(encoding="utf-8"))["pid"], os.getpid())

            lock_path.write_text("{corrupt", encoding="utf-8")
            ok2, _ = acquire_lock(lock_path, started=2001, is_python_pid=lambda _pid: True)
            self.assertTrue(ok2)
            self.assertEqual(json.loads(lock_path.read_text(encoding="utf-8"))["started"], 2001)

    def test_is_python_process_self_and_dead(self) -> None:
        self.assertTrue(is_python_process(os.getpid()))
        self.assertFalse(is_python_process(0))
        self.assertFalse(is_python_process(99999999))

    def test_status_atomic_rapid_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "status.json"
            errors: list[Exception] = []
            stop_flag = threading.Event()
            required = {"state", "pid", "started", "discord_ready", "voice_sessions", "llm_ok"}

            def reader() -> None:
                while not stop_flag.is_set():
                    try:
                        raw = status_path.read_text(encoding="utf-8")
                        parsed = json.loads(raw)
                        if not required.issubset(parsed.keys()):
                            errors.append(AssertionError(f"Missing keys: {parsed}"))
                    except (FileNotFoundError, PermissionError):
                        continue
                    except Exception as e:
                        errors.append(e)
                        return

            t = threading.Thread(target=reader, daemon=True)
            t.start()
            try:
                for i in range(60):
                    write_status(
                        status_path,
                        state="running",
                        started=1000,
                        discord_ready=bool(i % 2),
                        voice_sessions=i % 3,
                        llm_ok=True,
                    )
            finally:
                stop_flag.set()
                t.join(timeout=2.0)

            self.assertEqual(errors, [])
            final = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertTrue(required.issubset(final.keys()))

    def test_write_status_permission_error_logs_and_cleans_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            status_path = tmp_dir / "status.json"
            with (
                patch("core.runtime.os.replace", side_effect=PermissionError("locked")) as mock_replace,
                patch("core.runtime.time.sleep", return_value=None),
                self.assertLogs("pachinko.runtime", level="WARNING") as cm,
            ):
                write_status(
                    status_path,
                    state="running",
                    started=1000,
                    discord_ready=True,
                    voice_sessions=1,
                    llm_ok=True,
                )

            self.assertEqual(mock_replace.call_count, 20)
            self.assertTrue(any("Failed to write status file" in msg for msg in cm.output))
            self.assertEqual(list(tmp_dir.iterdir()), [])

    def test_setup_logging_creates_rotating_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "logs"
            handler = setup_logging(log_dir)
            try:
                self.assertIsInstance(handler, RotatingFileHandler)
                self.assertEqual(handler.maxBytes, LOG_MAX_BYTES)
                self.assertEqual(handler.backupCount, LOG_BACKUP_COUNT)
                self.assertEqual(handler.encoding, "utf-8")
                logging.getLogger("pachinko.test").info("hello runtime log")
                handler.flush()
                log_file = log_dir / "pachinko.log"
                self.assertTrue(log_file.exists())
                self.assertIn("hello runtime log", log_file.read_text(encoding="utf-8"))
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()


class TestStopRequestWatcher(unittest.IsolatedAsyncioTestCase):
    async def test_stop_request_watcher_triggers_once_and_deletes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stop_path = Path(tmp) / "stop.request"
            calls = 0

            def on_stop() -> None:
                nonlocal calls
                calls += 1

            task = asyncio.create_task(watch_stop_request(stop_path, on_stop, interval=0.02))
            await asyncio.sleep(0.05)
            self.assertEqual(calls, 0)
            stop_path.write_text("", encoding="utf-8")
            await asyncio.wait_for(task, timeout=1.0)
            self.assertEqual(calls, 1)
            self.assertFalse(stop_path.exists())


if __name__ == "__main__":
    unittest.main()
