"""Offline unit tests for app/process.py using a fake bot subprocess."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.process import (
    STATE_ALREADY_RUNNING,
    STATE_CRASHED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    BotProcessManager,
    format_uptime,
    load_settings,
    save_settings,
    tail_log_lines,
)
from core.runtime import EXIT_ALREADY_RUNNING, EXIT_CRASH

_FAKE_BOT_SCRIPT = """\
import os, sys, time
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from core.runtime import acquire_lock, release_lock, write_status, remove_status

data_dir = Path(sys.argv[2])
mode = sys.argv[3]
run_dir = data_dir / "run"
lock_path = run_dir / "bot.lock"
status_path = run_dir / "status.json"
stop_path = run_dir / "stop.request"
count_file = run_dir / "starts.count"

run_dir.mkdir(parents=True, exist_ok=True)
prev = int(count_file.read_text(encoding="utf-8")) if count_file.exists() else 0
count_file.write_text(str(prev + 1), encoding="utf-8")

if mode == "exit3":
    sys.exit(3)

started = int(time.time())
ok, _ = acquire_lock(lock_path, started=started)
if not ok:
    sys.exit(3)

try:
    write_status(
        status_path,
        state="running",
        started=started,
        discord_ready=True,
        voice_sessions=2,
        llm_ok=True,
    )
    if mode == "crash":
        time.sleep(0.05)
        sys.exit(1)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if mode != "ignore_stop" and stop_path.exists():
            stop_path.unlink(missing_ok=True)
            break
        time.sleep(0.05)
finally:
    release_lock(lock_path)
    remove_status(status_path)
"""


def _wait_until(pred, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


class TestAppProcessManager(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.repo_root = Path(__file__).resolve().parent.parent
        self.script_path = self.tmp_dir / "fake_bot.py"
        self.script_path.write_text(_FAKE_BOT_SCRIPT, encoding="utf-8")
        self._procs: list[subprocess.Popen[bytes]] = []

    def tearDown(self) -> None:
        for p in self._procs:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=3.0)
        self._tmp.cleanup()

    def _cmd(self, mode: str) -> list[str]:
        return [sys.executable, str(self.script_path), str(self.repo_root), str(self.tmp_dir), mode]

    def test_start_status_and_clean_stop(self) -> None:
        mgr = BotProcessManager(self.repo_root, data_dir=self.tmp_dir, bot_cmd=self._cmd("normal"))
        self.assertTrue(mgr.start())
        try:
            self.assertTrue(_wait_until(lambda: mgr.poll().state == STATE_RUNNING))
            st = mgr.poll()
            self.assertFalse(st.external)
            self.assertTrue(st.discord_ready)
            self.assertTrue(st.llm_ok)
            self.assertEqual(st.voice_sessions, 2)
            self.assertTrue(mgr.stop(timeout=5.0))
            self.assertEqual(mgr.poll().state, STATE_STOPPED)
        finally:
            mgr.force_stop()

    def test_force_stop_after_timeout(self) -> None:
        mgr = BotProcessManager(self.repo_root, data_dir=self.tmp_dir, bot_cmd=self._cmd("ignore_stop"))
        self.assertTrue(mgr.start())
        try:
            self.assertTrue(_wait_until(lambda: mgr.poll().state == STATE_RUNNING))
            self.assertFalse(mgr.stop(timeout=0.4, poll_interval=0.05))
            self.assertTrue(mgr.force_stop())
            st = mgr.poll()
            self.assertEqual(st.state, STATE_STOPPED)
            self.assertFalse(mgr.lock_path.exists())
        finally:
            mgr.force_stop()

    def test_crash_detection(self) -> None:
        mgr = BotProcessManager(self.repo_root, data_dir=self.tmp_dir, bot_cmd=self._cmd("crash"))
        self.assertTrue(mgr.start())
        self.assertTrue(_wait_until(lambda: mgr.poll().state == STATE_CRASHED))
        self.assertEqual(mgr.poll().exit_code, EXIT_CRASH)

    def test_exit_code_3_already_running(self) -> None:
        mgr = BotProcessManager(
            self.repo_root,
            data_dir=self.tmp_dir,
            bot_cmd=self._cmd("exit3"),
            auto_restart=True,
        )
        self.assertTrue(mgr.start())
        self.assertTrue(_wait_until(lambda: mgr.poll().state == STATE_ALREADY_RUNNING))
        self.assertEqual(mgr.poll().exit_code, EXIT_ALREADY_RUNNING)

    def test_external_bot_detection_and_stop(self) -> None:
        ext = subprocess.Popen(
            self._cmd("normal"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(ext)
        lock_path = self.tmp_dir / "run" / "bot.lock"
        status_path = self.tmp_dir / "run" / "status.json"
        self.assertTrue(_wait_until(lambda: lock_path.exists() and status_path.exists()))

        mgr = BotProcessManager(self.repo_root, data_dir=self.tmp_dir, bot_cmd=self._cmd("normal"))
        st = mgr.poll()
        lock_pid = json.loads(lock_path.read_text(encoding="utf-8"))["pid"]
        self.assertTrue(st.external)
        self.assertEqual(st.state, STATE_RUNNING)
        self.assertEqual(st.pid, lock_pid)
        self.assertFalse(mgr.start())

        self.assertTrue(mgr.stop(timeout=5.0))
        self.assertEqual(mgr.poll().state, STATE_STOPPED)

    def test_auto_restart_limit(self) -> None:
        mgr = BotProcessManager(
            self.repo_root,
            data_dir=self.tmp_dir,
            bot_cmd=self._cmd("crash"),
            auto_restart=True,
            max_restarts=3,
            restart_window_sec=600.0,
        )
        self.assertTrue(mgr.start())
        try:
            self.assertTrue(_wait_until(lambda: mgr.poll().state == STATE_CRASHED, timeout=8.0))
            starts = int((self.tmp_dir / "run" / "starts.count").read_text(encoding="utf-8"))
            self.assertEqual(starts, 4)  # 1 initial + 3 auto-restarts
        finally:
            mgr.force_stop()

    def test_helpers_tail_uptime_and_settings(self) -> None:
        self.assertEqual(format_uptime(45), "45s")
        self.assertEqual(format_uptime(780), "13m")
        self.assertEqual(format_uptime(7980), "2h 13m")

        log_path = self.tmp_dir / "pachinko.log"
        log_path.write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")
        tailed = tail_log_lines(log_path, max_lines=300)
        lines = tailed.splitlines()
        self.assertEqual(len(lines), 300)
        self.assertEqual(lines[0], "line 200")
        self.assertEqual(lines[-1], "line 499")

        cfg_path = self.tmp_dir / "app_settings.json"
        save_settings(cfg_path, {"start_bot_on_open": True, "minimize_to_tray_on_close": False})
        loaded = load_settings(cfg_path)
        self.assertTrue(loaded["start_bot_on_open"])
        self.assertFalse(loaded["minimize_to_tray_on_close"])
        self.assertFalse(loaded["auto_restart_on_crash"])

    def test_status_read_hiccups_keep_last_status(self) -> None:
        mgr = BotProcessManager(self.repo_root, data_dir=self.tmp_dir, bot_cmd=self._cmd("normal"))
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 42424
        mgr._proc = fake_proc

        mgr.run_dir.mkdir(parents=True, exist_ok=True)
        mgr.status_path.write_text(
            json.dumps(
                {
                    "state": "running",
                    "pid": 42424,
                    "started": int(time.time()) - 10,
                    "discord_ready": True,
                    "voice_sessions": 2,
                    "llm_ok": True,
                }
            ),
            encoding="utf-8",
        )

        st0 = mgr.poll()
        self.assertEqual(st0.state, STATE_RUNNING)
        self.assertTrue(st0.discord_ready)
        self.assertEqual(st0.voice_sessions, 2)
        self.assertTrue(st0.llm_ok)

        real_read_text = Path.read_text
        calls = 0

        def transient_perm_error(path_self: Path, *args, **kwargs):
            nonlocal calls
            if path_self == mgr.status_path and calls == 0:
                calls += 1
                raise PermissionError("locked")
            return real_read_text(path_self, *args, **kwargs)

        with (
            patch("app.process.time.sleep", return_value=None),
            patch.object(Path, "read_text", autospec=True, side_effect=transient_perm_error),
        ):
            st_retry = mgr.poll()
        self.assertEqual(calls, 1)
        self.assertEqual(st_retry.state, STATE_RUNNING)
        self.assertTrue(st_retry.discord_ready)

        with (
            patch("app.process.time.sleep", return_value=None),
            patch.object(Path, "read_text", side_effect=PermissionError("locked")),
        ):
            st_perm = mgr.poll()
        self.assertEqual(st_perm.state, STATE_RUNNING)
        self.assertTrue(st_perm.discord_ready)
        self.assertEqual(st_perm.voice_sessions, 2)
        self.assertTrue(st_perm.llm_ok)

        mgr.status_path.unlink()
        with patch("app.process.time.sleep", return_value=None):
            st_missing = mgr.poll()
        self.assertEqual(st_missing.state, STATE_RUNNING)
        self.assertTrue(st_missing.discord_ready)

        mgr.status_path.write_text("{broken", encoding="utf-8")
        with patch("app.process.time.sleep", return_value=None):
            st_bad_json = mgr.poll()
        self.assertEqual(st_bad_json.state, STATE_RUNNING)
        self.assertTrue(st_bad_json.discord_ready)

    def test_setup_launch_args_have_no_full_path(self) -> None:
        from app.ui import PachinkoControlApp

        rd_dir = self.tmp_dir / "R&D"
        rd_dir.mkdir(parents=True, exist_ok=True)
        (rd_dir / "setup.bat").write_text("@echo off\n", encoding="utf-8")

        fake_app = SimpleNamespace(manager=SimpleNamespace(root_dir=rd_dir))
        with patch("app.ui.subprocess.Popen") as mock_popen:
            PachinkoControlApp.repair_install(fake_app)
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        self.assertEqual(args, ["cmd.exe", "/c", "setup.bat"])
        self.assertEqual(mock_popen.call_args.kwargs.get("cwd"), str(rd_dir))
        for arg in args:
            self.assertNotIn(os.sep, arg)
            if arg != "/c":
                self.assertNotIn("/", arg)
            self.assertNotIn("R&D", arg)

        old_cwd = os.getcwd()
        try:
            with patch.dict(os.environ, {"PACHINKO_APP_BOOTSTRAPPED": "1"}):
                loader = importlib.machinery.SourceFileLoader(
                    "pachinko_launcher_test", str(self.repo_root / "Pachinko.pyw")
                )
                spec = importlib.util.spec_from_loader(loader.name, loader)
                assert spec is not None
                mod = importlib.util.module_from_spec(spec)
                loader.exec_module(mod)
        finally:
            os.chdir(old_cwd)

        with (
            patch("tkinter.Tk"),
            patch("tkinter.messagebox.askyesno", return_value=True),
            patch.object(mod.subprocess, "Popen") as mock_pyw_popen,
            patch.object(mod, "SETUP_BAT", rd_dir / "setup.bat"),
        ):
            mod._offer_setup("Run setup?")
        mock_pyw_popen.assert_called_once()
        pyw_args = mock_pyw_popen.call_args[0][0]
        self.assertEqual(pyw_args, ["cmd.exe", "/c", "setup.bat"])
        self.assertEqual(mock_pyw_popen.call_args.kwargs.get("cwd"), str(mod.ROOT))
        for arg in pyw_args:
            self.assertNotIn(os.sep, arg)
            if arg != "/c":
                self.assertNotIn("/", arg)
            self.assertNotIn(str(mod.ROOT), arg)

    def test_spawn_removes_stale_status_and_poll_shows_starting(self) -> None:
        mgr = BotProcessManager(self.repo_root, data_dir=self.tmp_dir, bot_cmd=self._cmd("normal"))
        mgr.run_dir.mkdir(parents=True, exist_ok=True)
        mgr.status_path.write_text(
            json.dumps(
                {
                    "state": "running",
                    "pid": 99999,
                    "started": int(time.time()) - 3600,
                    "discord_ready": True,
                    "llm_ok": True,
                }
            ),
            encoding="utf-8",
        )
        fake_proc = MagicMock()
        fake_proc.poll.return_value = None
        fake_proc.pid = 11111
        with patch("app.process.subprocess.Popen", return_value=fake_proc):
            self.assertTrue(mgr.start())
        self.assertFalse(mgr.status_path.exists())
        st = mgr.poll()
        self.assertEqual(st.state, STATE_STARTING)
        self.assertFalse(st.discord_ready)
        self.assertEqual(st.uptime, 0)

    def test_force_stop_removes_dead_lock_when_launcher_pid_differs(self) -> None:
        alive_pids = {18344, 26168}

        def fake_is_python(pid: int, _started=None) -> bool:
            return pid in alive_pids

        mgr = BotProcessManager(
            self.repo_root,
            data_dir=self.tmp_dir,
            bot_cmd=self._cmd("normal"),
            is_python_pid=fake_is_python,
        )
        mgr.run_dir.mkdir(parents=True, exist_ok=True)
        mgr.lock_path.write_text(json.dumps({"pid": 26168, "started": 1000}), encoding="utf-8")

        killed = False

        def on_kill(*_args, **_kwargs) -> None:
            nonlocal killed
            killed = True
            alive_pids.discard(18344)
            alive_pids.discard(26168)

        fake_proc = MagicMock()
        fake_proc.pid = 18344
        fake_proc.poll.side_effect = lambda: 1 if killed else None
        mgr._proc = fake_proc

        with (
            patch("app.process.subprocess.run", side_effect=on_kill) as mock_run,
            patch("app.process.os.kill", side_effect=on_kill),
        ):
            self.assertTrue(mgr.force_stop())
        if sys.platform == "win32":
            mock_run.assert_called_once()
            self.assertEqual(mock_run.call_args[0][0], ["taskkill", "/PID", "18344", "/T", "/F"])
        self.assertFalse(mgr.lock_path.exists())

        # Opposite: if child lock_pid stays alive after kill, lock_path must remain.
        alive_pids = {18344, 26168}
        killed = False
        mgr.lock_path.write_text(json.dumps({"pid": 26168, "started": 1000}), encoding="utf-8")
        fake_proc2 = MagicMock()
        fake_proc2.pid = 18344
        fake_proc2.poll.side_effect = lambda: 1 if killed else None
        mgr._proc = fake_proc2

        def on_kill_launcher_only(*_args, **_kwargs) -> None:
            nonlocal killed
            killed = True
            alive_pids.discard(18344)

        with (
            patch("app.process.subprocess.run", side_effect=on_kill_launcher_only),
            patch("app.process.os.kill", side_effect=on_kill_launcher_only),
        ):
            self.assertTrue(mgr.force_stop())
        self.assertTrue(mgr.lock_path.exists())

    def test_force_stop_refuses_mismatched_start_time_and_allows_unreadable(self) -> None:
        mgr = BotProcessManager(self.repo_root, data_dir=self.tmp_dir, bot_cmd=self._cmd("normal"))
        mgr.run_dir.mkdir(parents=True, exist_ok=True)
        mgr.lock_path.write_text(json.dumps({"pid": 55555, "started": 1000}), encoding="utf-8")

        with (
            patch("core.runtime._process_info", return_value=(True, 4600.0)),
            patch("app.process.subprocess.run") as mock_run,
            patch("app.process.os.kill") as mock_kill,
        ):
            self.assertEqual(mgr.poll().state, STATE_STOPPED)
            mgr.force_stop()
            mock_run.assert_not_called()
            mock_kill.assert_not_called()
            self.assertFalse(mgr.force_stop(pid=55555))
            mock_run.assert_not_called()
            mock_kill.assert_not_called()

        with (
            patch("core.runtime._process_info", return_value=(True, None)),
            patch("app.process.subprocess.run") as mock_run,
            patch("app.process.os.kill") as mock_kill,
        ):
            st = mgr.poll()
            self.assertTrue(st.external)
            self.assertEqual(st.pid, 55555)
            mgr.force_stop()
            if sys.platform == "win32":
                mock_run.assert_called_once()
                self.assertEqual(mock_run.call_args[0][0], ["taskkill", "/PID", "55555", "/T", "/F"])
            else:
                mock_kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
