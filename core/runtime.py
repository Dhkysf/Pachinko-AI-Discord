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
LOCK_START_TOLERANCE_SEC = 10.0
_WIN_EPOCH_TICKS = 116_444_736_000_000_000
_KERNEL32: Any = None

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


def _process_info(pid: int) -> tuple[bool, float | None]:
    """Return (is_live_python, creation_unix_ts_or_none) for pid."""
    if sys.platform == "win32":
        global _KERNEL32
        from ctypes import wintypes

        if _KERNEL32 is None:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            k32.GetExitCodeProcess.restype = wintypes.BOOL
            k32.QueryFullProcessImageNameW.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.LPWSTR,
                ctypes.POINTER(wintypes.DWORD),
            ]
            k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
            ft_ptr = ctypes.POINTER(wintypes.FILETIME)
            k32.GetProcessTimes.argtypes = [wintypes.HANDLE, ft_ptr, ft_ptr, ft_ptr, ft_ptr]
            k32.GetProcessTimes.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            k32.CloseHandle.restype = wintypes.BOOL
            _KERNEL32 = k32
        k32 = _KERNEL32
        handle = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False, None
        try:
            code = wintypes.DWORD(0)
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                return False, None
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            if not k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return False, None
            if not Path(buf.value).name.lower().startswith("python"):
                return False, None
            creation = wintypes.FILETIME()
            exit_t = wintypes.FILETIME()
            kernel_t = wintypes.FILETIME()
            user_t = wintypes.FILETIME()
            if k32.GetProcessTimes(
                handle, ctypes.byref(creation), ctypes.byref(exit_t), ctypes.byref(kernel_t), ctypes.byref(user_t)
            ):
                ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                if ticks > 0:
                    return True, (ticks - _WIN_EPOCH_TICKS) / 10_000_000.0
            return True, None
        finally:
            k32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, None
    except PermissionError:
        return True, None
    comm = Path(f"/proc/{pid}/comm")
    if comm.exists():
        try:
            return "python" in comm.read_text(encoding="utf-8", errors="ignore").lower(), None
        except OSError:
            return True, None
    return True, None


def is_python_process(pid: int, started: int | float | None = None) -> bool:
    """Return True if pid is a live Python process (and matches started within 10s when known)."""
    if isinstance(pid, bool) or pid <= 0:
        return False
    if pid == os.getpid() and started is None:
        return True
    is_py, proc_started = _process_info(pid)
    if not is_py:
        return False
    if started is not None and not isinstance(started, bool) and proc_started is not None:
        if abs(proc_started - float(started)) > LOCK_START_TOLERANCE_SEC:
            return False
    return True


def check_python_pid(fn: Callable[..., bool], pid: int, started: Any = None) -> bool:
    """Invoke a 1-arg or 2-arg (pid, started) process-check callback safely."""
    started_val = started if isinstance(started, (int, float)) and not isinstance(started, bool) else None
    if fn is is_python_process:
        return is_python_process(pid, started_val)
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return bool(fn(pid))
    accepts_two = sum(
        1 for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ) >= 2 or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    return bool(fn(pid, started_val)) if accepts_two else bool(fn(pid))


def acquire_lock(
    lock_path: Path,
    *,
    started: int | None = None,
    is_python_pid: Callable[..., bool] = is_python_process,
) -> tuple[bool, int | None]:
    """Acquire single-instance lock; return (True, None) or (False, existing_pid)."""
    if lock_path.exists():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
            existing_pid = data.get("pid") if isinstance(data, dict) else None
            existing_started = data.get("started") if isinstance(data, dict) else None
            if (
                isinstance(existing_pid, int)
                and not isinstance(existing_pid, bool)
                and check_python_pid(is_python_pid, existing_pid, existing_started)
            ):
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
    llm_provider: str = "",
    llm_model: str = "",
    ollama_model: str = "",
    llm_last_provider: str = "",
    llm_last_model: str = "",
    llm_fallback_active: bool = False,
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
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "ollama_model": ollama_model,
                "llm_last_provider": llm_last_provider,
                "llm_last_model": llm_last_model,
                "llm_fallback_active": bool(llm_fallback_active),
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
