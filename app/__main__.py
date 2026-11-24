"""Single-instance entrypoint for Pachinko Control."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import secrets
import socket
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Callable

from app.process import BotProcessManager
from app.tray import TrayController
from app.ui import PachinkoControlApp
from core.runtime import atomic_write_json

_ERROR_ALREADY_EXISTS = 183


class SingleInstance:
    """Enforces one Pachinko Control window per repo using a Windows mutex + localhost socket."""

    def __init__(self, root_dir: Path, lock_path: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.lock_path = lock_path
        self._mutex_handle = None
        self._sock: socket.socket | None = None
        self._token = secrets.token_hex(16)
        self._stop_event = threading.Event()

    def acquire(self) -> bool:
        if sys.platform == "win32":
            digest = hashlib.sha256(str(self.root_dir).lower().encode("utf-8")).hexdigest()[:16]
            name = f"Local\\PachinkoControl_{digest}"
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.CreateMutexW(None, False, name)
            if handle:
                self._mutex_handle = handle
                if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
                    return False

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(5)
        sock.settimeout(0.5)
        self._sock = sock
        port = sock.getsockname()[1]
        atomic_write_json(
            self.lock_path,
            {"pid": os.getpid(), "port": port, "token": self._token},
        )
        return True

    def notify_existing(self, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data = json.loads(self.lock_path.read_text(encoding="utf-8"))
                port = int(data["port"])
                token = str(data["token"])
                with socket.create_connection(("127.0.0.1", port), timeout=0.5) as conn:
                    conn.sendall(f"SHOW {token}\n".encode("utf-8"))
                return True
            except (OSError, ValueError, KeyError, TypeError):
                time.sleep(0.05)
        return False

    def serve(self, on_show: Callable[[], None]) -> None:
        if self._sock is None:
            return

        def loop() -> None:
            assert self._sock is not None
            while not self._stop_event.is_set():
                try:
                    conn, _ = self._sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with conn:
                    try:
                        msg = conn.recv(256).decode("utf-8", errors="ignore").strip()
                    except OSError:
                        continue
                    if msg == f"SHOW {self._token}":
                        on_show()

        threading.Thread(target=loop, daemon=True).start()

    def close(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        try:
            if self.lock_path.exists():
                data = json.loads(self.lock_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("pid") == os.getpid():
                    self.lock_path.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError):
            pass
        if sys.platform == "win32" and self._mutex_handle:
            try:
                ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(self._mutex_handle)
            except Exception:
                pass
            self._mutex_handle = None


def main() -> int:
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    instance = SingleInstance(root_dir, data_dir / "run" / "app.lock")
    if not instance.acquire():
        instance.notify_existing()
        instance.close()
        return 0

    root = tk.Tk()
    manager = BotProcessManager(root_dir, data_dir=data_dir)
    app = PachinkoControlApp(root, manager)
    tray = TrayController(
        dispatch=lambda fn: root.after(0, fn),
        on_show=app.show_window,
        on_start=app.start_bot,
        on_stop=app.stop_bot,
        on_restart=app.restart_bot,
        on_open_panel=app.open_web_panel,
        on_quit=app.quit_app,
    )
    app.attach_tray(tray)
    tray.start()
    instance.serve(lambda: root.after(0, app.show_window))

    if app.settings.get("start_bot_on_open"):
        app.start_bot()

    try:
        root.mainloop()
    finally:
        tray.stop()
        instance.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
