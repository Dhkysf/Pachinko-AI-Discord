"""Runtime files for background control: rotating logs, single-instance lock, status.json, and stop.request."""
from __future__ import annotations

import asyncio
import ctypes
import inspect
import json
import logging
import os
import sys
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

EXIT_OK = 0
EXIT_CRASH = 1
EXIT_ALREADY_RUNNING = 3

LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

log = logging.getLogger("pachinko.runtime")


def setup_logging(log_dir: Path) -> RotatingFileHandler:
    """Configure root logging to console and rotating UTF-8 file without removing existing filters."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / "pachinko.log").resolve()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and Path(h.baseFilename).resolve() == log_path:
            return h

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return file_handler


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically via a temp file in the same directory + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    data = json.dumps(payload, ensure_ascii=False)
    try:
        tmp.write_text(data, encoding="utf-8")
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(0.05, 0.01 * (attempt + 1)))
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def is_python_process(pid: int) -> bool:
    """Return True if pid is a live Python process."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True

    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_uint32(0)
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            if code.value != 259:  # STILL_ACTIVE
                return False
            buf = ctypes.create_unicode_buffer(1024)
            size = ctypes.c_uint32(len(buf))
            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return Path(buf.value).name.lower().startswith("python")
            return False
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    comm = Path(f"/proc/{pid}/comm")
    if comm.exists():
        try:
            return "python" in comm.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return True
    return True


def acquire_lock(
    lock_path: Path,
    *,
    started: int | None = None,
    is_python_pid: Callable[[int], bool] = is_python_process,
) -> tuple[bool, int | None]:
    """Acquire single-instance lock; return (True, None) or (False, existing_pid)."""
    if lock_path.exists():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
            existing_pid = data.get("pid") if isinstance(data, dict) else None
            if isinstance(existing_pid, int) and is_python_pid(existing_pid):
                return False, existing_pid
        except (OSError, ValueError, TypeError):
            pass

    atomic_write_json(
        lock_path,
        {
            "pid": os.getpid(),
            "started": int(time.time()) if started is None else int(started),
        },
    )
    return True, None


def release_lock(lock_path: Path) -> None:
    """Remove lock_path only if it belongs to the current process."""
    try:
        if not lock_path.exists():
            return
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("pid") == os.getpid():
            lock_path.unlink(missing_ok=True)
    except OSError:
        pass
    except ValueError:
        pass


def write_status(
    status_path: Path,
    *,
    state: str,
    started: int,
    discord_ready: bool,
    voice_sessions: int,
    llm_ok: bool,
    ollama_model: str = "",
) -> None:
    """Atomically write runtime status.json for the desktop control app."""
    try:
        atomic_write_json(
            status_path,
            {
                "state": state,
                "pid": os.getpid(),
                "started": int(started),
                "discord_ready": bool(discord_ready),
                "voice_sessions": int(voice_sessions),
                "llm_ok": bool(llm_ok),
                "ollama_model": ollama_model,
            },
        )
    except Exception as exc:
        log.warning("Failed to write status file %s: %s", status_path, exc)


def remove_status(status_path: Path) -> None:
    """Best-effort removal of status.json on clean exit."""
    try:
        status_path.unlink(missing_ok=True)
    except OSError:
        pass


async def watch_stop_request(
    stop_path: Path,
    on_stop: Callable[[], Any],
    *,
    interval: float = 1.0,
) -> None:
    """Poll for stop.request; when present, delete it and trigger on_stop once."""
    while True:
        if stop_path.exists():
            try:
                stop_path.unlink(missing_ok=True)
            except OSError:
                pass
            res = on_stop()
            if inspect.isawaitable(res):
                await res
            return
        await asyncio.sleep(interval)
