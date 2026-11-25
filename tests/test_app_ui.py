"""Offline unit tests for UI background runner and Startup shortcut target check."""
from __future__ import annotations

import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app.ui import _run_off_ui, _startup_shortcut_points_here


class _FakeRoot:
    def __init__(self) -> None:
        self.calls: list[tuple[int, object]] = []

    def after(self, ms: int, fn: object) -> None:
        self.calls.append((ms, fn))


class TestAppUiHelpers(unittest.TestCase):
    def test_slow_check_runs_off_ui_thread_and_result_comes_back_via_after(self) -> None:
        root = _FakeRoot()
        got: list[object] = []
        work_tid: list[int] = []

        def work() -> str:
            work_tid.append(threading.get_ident())
            time.sleep(0.3)
            return "checked"

        t0 = time.monotonic()
        thread = _run_off_ui(root, work, got.append)
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.2)

        thread.join(5.0)
        self.assertEqual(len(work_tid), 1)
        self.assertNotEqual(work_tid[0], threading.get_ident())
        self.assertEqual(got, [])
        self.assertEqual(len(root.calls), 1)
        self.assertEqual(root.calls[0][0], 0)

        cb = root.calls[0][1]
        assert callable(cb)
        cb()
        self.assertEqual(got, ["checked"])

    def test_work_exception_is_handed_to_done(self) -> None:
        root = _FakeRoot()
        got: list[object] = []

        def work() -> str:
            raise RuntimeError("boom")

        thread = _run_off_ui(root, work, got.append)
        thread.join(5.0)
        cb = root.calls[0][1]
        assert callable(cb)
        cb()
        self.assertEqual(len(got), 1)
        self.assertIsInstance(got[0], RuntimeError)

    def test_startup_shortcut_points_here(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "repo"
            root.mkdir()
            shortcut = Path(tmpdir) / "Pachinko.lnk"

            with patch("app.ui.subprocess.run") as mock_run:
                self.assertFalse(_startup_shortcut_points_here(shortcut, root))
                mock_run.assert_not_called()

            shortcut.write_bytes(b"lnk")
            ok_proc = subprocess.CompletedProcess([], 0, stdout=f'"{root / "Pachinko.pyw"}"\r\n', stderr="")
            with patch("app.ui.subprocess.run", return_value=ok_proc):
                self.assertTrue(_startup_shortcut_points_here(shortcut, root))

            other_proc = subprocess.CompletedProcess([], 0, stdout='"C:\\Other\\Pachinko.pyw"\r\n', stderr="")
            with patch("app.ui.subprocess.run", return_value=other_proc):
                self.assertFalse(_startup_shortcut_points_here(shortcut, root))

            with patch("app.ui.subprocess.run", side_effect=OSError("no ps")):
                self.assertFalse(_startup_shortcut_points_here(shortcut, root))

            err_proc = subprocess.CompletedProcess([], 1, stdout="", stderr="err")
            with patch("app.ui.subprocess.run", return_value=err_proc):
                self.assertFalse(_startup_shortcut_points_here(shortcut, root))


if __name__ == "__main__":
    unittest.main()
