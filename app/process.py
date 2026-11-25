"""Bot process manager, status polling, log tailing, and app settings (no GUI)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import dotenv_values

from core.runtime import (
    EXIT_ALREADY_RUNNING,
    EXIT_OK,
    atomic_write_json,
    check_python_pid,
    is_python_process,
)

STATE_STOPPED = "Stopped"
STATE_STARTING = "Starting"
STATE_RUNNING = "Running"
STATE_STOPPING = "Stopping"
STATE_CRASHED = "Crashed"
STATE_ALREADY_RUNNING = "Already running"

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

DEFAULT_SETTINGS: dict[str, Any] = {
    "start_bot_on_open": False,
    "start_with_windows": False,
    "auto_restart_on_crash": False,
    "minimize_to_tray_on_close": True,
    "llm_source": "",
    "model_ollama": "",
    "model_ninerouter": "",
}

LLM_SOURCES = ("ninerouter", "ollama")
SOURCE_LABELS: dict[str, str] = {
    "ninerouter": "9router",
    "ollama": "Ollama",
    "openai_compat": "OpenAI-compat",
}
MODEL_SETTING_KEY: dict[str, str] = {
    "ollama": "model_ollama",
    "ninerouter": "model_ninerouter",
}
SOURCE_MODEL_ENV_VAR: dict[str, str] = {
    "ollama": "OLLAMA_MODEL",
    "ninerouter": "NINEROUTER_MODEL",
}
_LLM_OVERRIDE_VARS = ("LLM_PROVIDER", "OLLAMA_MODEL", "NINEROUTER_MODEL")


def source_label(source: str) -> str:
    """Return display label for an LLM source."""
    key = (source or "").strip().lower()
    return SOURCE_LABELS.get(key, key or "9router")


def llm_display_label(provider: str, fallback_active: bool = False) -> str:
    """Return display label for the active LLM provider, noting Ollama fallback."""
    if fallback_active:
        return "Ollama fallback"
    return source_label(provider)


def build_llm_env_overrides(llm_source: str, llm_model: str) -> dict[str, str]:
    """Return process env overrides for the selected LLM source/model, or {} for '(use .env)'."""
    src = (llm_source or "").strip().lower()
    if src not in SOURCE_MODEL_ENV_VAR:
        return {}
    out = {"LLM_PROVIDER": src}
    model = (llm_model or "").strip()
    if model:
        out[SOURCE_MODEL_ENV_VAR[src]] = model
    return out


def effective_llm_selection(settings: dict[str, Any], env: dict[str, Any]) -> tuple[str, str]:
    """Resolve effective (source, model) from UI settings falling back to .env summary."""
    src = str(settings.get("llm_source", "") or "").strip().lower()
    if src not in LLM_SOURCES:
        env_src = str(env.get("llm_provider", "ninerouter") or "ninerouter").strip().lower()
        return env_src, str(env.get(f"{env_src}_model", "") or "").strip()
    model_key = MODEL_SETTING_KEY[src]
    chosen = str(settings.get(model_key, "") or "").strip()
    return src, chosen or str(env.get(f"{src}_model", "") or "").strip()

_STATUS_MAP = {
    "starting": STATE_STARTING,
    "running": STATE_RUNNING,
    "stopping": STATE_STOPPING,
}


@dataclass(frozen=True)
class BotStatus:
    state: str
    external: bool = False
    pid: int | None = None
    started: int | None = None
    uptime: int = 0
    discord_ready: bool = False
    llm_ok: bool = False
    voice_sessions: int = 0
    exit_code: int | None = None
    llm_provider: str = ""
    llm_model: str = ""
    ollama_model: str = ""
    llm_last_provider: str = ""
    llm_last_model: str = ""
    llm_fallback_active: bool = False


def format_llm_status(status: BotStatus, default_source: str = "ninerouter") -> str:
    """Format the status bar LLM label from BotStatus."""
    src = status.llm_last_provider or status.llm_provider or default_source
    label = llm_display_label(src, status.llm_fallback_active)
    model = status.llm_last_model or status.llm_model or status.ollama_model
    state = "ok" if status.llm_ok else "error"
    return f"LLM ({label}): {state}" + (f" ({model})" if model else "")


def format_uptime(seconds: int) -> str:
    """Format uptime seconds into compact human text (e.g. '45s', '13m', '2h 13m')."""
    secs = max(0, int(seconds))
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m"
    hours, rem_mins = divmod(mins, 60)
    return f"{hours}h {rem_mins}m"


def tail_log_lines(log_path: Path, max_lines: int = 300, max_bytes: int = 131072) -> str:
    """Read up to the last max_lines of a UTF-8 log file without keeping it open."""
    if not log_path.exists():
        return ""
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            offset = max(0, size - max_bytes)
            f.seek(offset, os.SEEK_SET)
            raw = f.read()
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if offset > 0 and lines:
        # Drop partial first line when we seeked into the middle of the file.
        lines = lines[1:]
    return "\n".join(lines[-max_lines:])


def load_settings(path: Path) -> dict[str, Any]:
    """Load app_settings.json merged over DEFAULT_SETTINGS, migrating legacy ollama_model."""
    settings: dict[str, Any] = dict(DEFAULT_SETTINGS)
    if not path.exists():
        return settings
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for key, default in DEFAULT_SETTINGS.items():
                if key in raw:
                    if isinstance(default, bool):
                        settings[key] = bool(raw[key])
                    else:
                        settings[key] = str(raw[key] or "").strip()
            if "model_ollama" not in raw and raw.get("ollama_model"):
                settings["model_ollama"] = str(raw["ollama_model"] or "").strip()
            src = str(settings.get("llm_source", "")).strip().lower()
            settings["llm_source"] = src if src in LLM_SOURCES else ""
    except (OSError, ValueError, TypeError):
        pass
    return settings


def save_settings(path: Path, settings: dict[str, Any]) -> None:
    """Persist app_settings.json atomically."""
    clean: dict[str, Any] = {}
    for key, default in DEFAULT_SETTINGS.items():
        val = settings.get(key, default)
        clean[key] = bool(val) if isinstance(default, bool) else str(val or "").strip()
    atomic_write_json(path, clean)


def read_env_summary(root_dir: Path) -> dict[str, Any]:
    """Read non-secret .env summary (presence of keys only, never their values)."""
    env_path = root_dir / ".env"
    summary: dict[str, Any] = {
        "env_exists": env_path.exists(),
        "has_discord_token": False,
        "has_ninerouter_api_key": False,
        "llm_provider": "ninerouter",
        "llm_fallback": "ollama",
        "ollama_base_url": "http://127.0.0.1:11434",
        "ollama_model": "qwen3:8b",
        "ollama_keep_alive": "15m",
        "ninerouter_base_url": "http://127.0.0.1:20128/v1",
        "ninerouter_model": "",
        "openai_compat_model": "qwen3-8b",
        "panel_host": "127.0.0.1",
        "panel_port": 7878,
    }
    if not summary["env_exists"]:
        return summary
    try:
        vals = dotenv_values(env_path)
    except OSError:
        return summary

    summary["has_discord_token"] = bool((vals.get("DISCORD_TOKEN") or "").strip())
    summary["has_ninerouter_api_key"] = bool((vals.get("NINEROUTER_API_KEY") or "").strip())
    raw_prov = str(vals.get("LLM_PROVIDER") or "").strip().lower()
    if raw_prov in ("ninerouter", "ollama", "openai_compat"):
        summary["llm_provider"] = raw_prov
    raw_fb = str(vals.get("LLM_FALLBACK") or "").strip().lower()
    if raw_fb in ("ollama", "none"):
        summary["llm_fallback"] = raw_fb
    if (vals.get("OLLAMA_BASE_URL") or "").strip():
        summary["ollama_base_url"] = str(vals["OLLAMA_BASE_URL"]).strip()
    if (vals.get("OLLAMA_MODEL") or "").strip():
        summary["ollama_model"] = str(vals["OLLAMA_MODEL"]).strip()
    if (vals.get("OLLAMA_KEEP_ALIVE") or "").strip():
        summary["ollama_keep_alive"] = str(vals["OLLAMA_KEEP_ALIVE"]).strip()
    if (vals.get("NINEROUTER_BASE_URL") or "").strip():
        summary["ninerouter_base_url"] = str(vals["NINEROUTER_BASE_URL"]).strip()
    if (vals.get("NINEROUTER_MODEL") or "").strip():
        summary["ninerouter_model"] = str(vals["NINEROUTER_MODEL"]).strip()
    if (vals.get("OPENAI_MODEL") or "").strip():
        summary["openai_compat_model"] = str(vals["OPENAI_MODEL"]).strip()
    if (vals.get("PANEL_HOST") or "").strip():
        summary["panel_host"] = str(vals["PANEL_HOST"]).strip()
    raw_port = vals.get("PANEL_PORT")
    if raw_port is not None and str(raw_port).strip():
        try:
            summary["panel_port"] = int(str(raw_port).strip())
        except (ValueError, TypeError):
            pass
    return summary


def read_llm_api_key(root_dir: Path, source: str) -> str:
    """Read only the requested provider API key from .env for model-list requests (never store/log)."""
    if (source or "").strip().lower() != "ninerouter":
        return ""
    key_name = "NINEROUTER_API_KEY"
    env_path = root_dir / ".env"
    if not env_path.exists():
        return ""
    try:
        vals = dotenv_values(env_path)
    except OSError:
        return ""
    return str(vals.get(key_name) or "").strip()


def read_panel_port(root_dir: Path) -> int:
    """Read only PANEL_PORT from .env (default 7878) without mutating os.environ."""
    return int(read_env_summary(root_dir)["panel_port"])


def read_ollama_url(root_dir: Path) -> str:
    """Read only OLLAMA_BASE_URL from .env (default http://127.0.0.1:11434)."""
    return str(read_env_summary(root_dir)["ollama_base_url"])


class BotProcessManager:
    """Manages the bot process lifecycle, status.json polling, and crash restarts."""

    def __init__(
        self,
        root_dir: Path,
        *,
        data_dir: Path | None = None,
        bot_cmd: list[str] | None = None,
        auto_restart: bool = False,
        llm_source: str = "",
        llm_model: str = "",
        max_restarts: int = 3,
        restart_window_sec: float = 600.0,
        is_python_pid: Callable[..., bool] = is_python_process,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.data_dir = Path(data_dir) if data_dir is not None else (self.root_dir / "data")
        self.run_dir = self.data_dir / "run"
        self.lock_path = self.run_dir / "bot.lock"
        self.status_path = self.run_dir / "status.json"
        self.stop_path = self.run_dir / "stop.request"
        self.log_path = self.data_dir / "logs" / "pachinko.log"
        self.bot_cmd = list(bot_cmd) if bot_cmd is not None else [
            str(self.root_dir / ".venv" / "Scripts" / "python.exe"),
            str(self.root_dir / "main.py"),
        ]
        self.auto_restart = auto_restart
        self.llm_source = llm_source.strip().lower()
        self.llm_model = llm_model.strip()
        self.max_restarts = max_restarts
        self.restart_window_sec = restart_window_sec
        self._is_python_pid = is_python_pid
        self._clock = clock

        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._stopping = False
        self._stop_requested = False
        self._terminal_state = STATE_STOPPED
        self._last_exit_code: int | None = None
        self._last_status: BotStatus | None = None
        self._restart_timestamps: list[float] = []

    def _read_json(self, path: Path, *, retry: bool = False) -> dict[str, Any] | None:
        attempts = 2 if retry else 1
        for attempt in range(attempts):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
            except (PermissionError, FileNotFoundError, OSError, ValueError, TypeError):
                pass
            if attempt == 0 and attempts > 1:
                time.sleep(0.05)
        return None

    def _read_live_lock_pid(self) -> tuple[int | None, int | None]:
        data = self._read_json(self.lock_path)
        if not data:
            return None, None
        pid = data.get("pid")
        started = data.get("started")
        started_val = started if isinstance(started, (int, float)) and not isinstance(started, bool) else None
        if isinstance(pid, int) and not isinstance(pid, bool) and check_python_pid(self._is_python_pid, pid, started_val):
            return pid, int(started_val) if started_val is not None else None
        return None, None

    def _spawn(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop_path.unlink(missing_ok=True)
        self.status_path.unlink(missing_ok=True)
        self._stopping = False
        self._stop_requested = False
        self._last_status = None
        env = os.environ.copy()
        for var in _LLM_OVERRIDE_VARS:
            env.pop(var, None)
        env.update(build_llm_env_overrides(self.llm_source, self.llm_model))
        self._proc = subprocess.Popen(
            self.bot_cmd,
            cwd=str(self.root_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )

    def start(self) -> bool:
        """Start the bot if not already running (managed or external)."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False
            ext_pid, _ = self._read_live_lock_pid()
            if ext_pid is not None:
                return False
            self._terminal_state = STATE_STOPPED
            self._last_exit_code = None
            self._restart_timestamps.clear()
            self._spawn()
            return True

    def stop(self, timeout: float = 20.0, poll_interval: float = 0.2) -> bool:
        """Request a clean stop via stop.request and wait up to timeout seconds."""
        with self._lock:
            proc = self._proc if (self._proc is not None and self._proc.poll() is None) else None
            ext_pid, ext_started = self._read_live_lock_pid() if proc is None else (None, None)
            if proc is None and ext_pid is None:
                self._stopping = False
                return True
            self._stopping = True
            self._stop_requested = True
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.stop_path.write_text("", encoding="utf-8")

        deadline = self._clock() + max(0.0, timeout)
        while True:
            if proc is not None:
                if proc.poll() is not None:
                    return True
            else:
                if not check_python_pid(self._is_python_pid, ext_pid, ext_started):  # type: ignore[arg-type]
                    return True
            if self._clock() >= deadline:
                return False
            time.sleep(poll_interval)

    def cancel_stop(self) -> None:
        """Clear stopping flags if the user declined Force stop."""
        with self._lock:
            self._stopping = False
            self._stop_requested = False

    def force_stop(self, pid: int | None = None) -> bool:
        """Force-kill the bot process tree and clean up stale runtime files."""
        with self._lock:
            proc = self._proc
            target_pid = pid
            target_started: int | None = None
            if target_pid is None:
                if proc is not None and proc.poll() is None:
                    target_pid = proc.pid
                else:
                    target_pid, target_started = self._read_live_lock_pid()
            elif proc is None or proc.poll() is not None or target_pid != proc.pid:
                lock_data = self._read_json(self.lock_path)
                if lock_data and lock_data.get("pid") == target_pid:
                    raw_started = lock_data.get("started")
                    if isinstance(raw_started, (int, float)) and not isinstance(raw_started, bool):
                        target_started = int(raw_started)
                if not check_python_pid(self._is_python_pid, target_pid, target_started):
                    self._stopping = False
                    self._stop_requested = True
                    return False

            self._stopping = False
            self._stop_requested = True
            if target_pid is None:
                self._terminal_state = STATE_STOPPED
                return True

        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(target_pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
        else:
            try:
                os.kill(target_pid, signal.SIGKILL)
            except OSError:
                pass

        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass

        with self._lock:
            still_alive = (
                (proc is not None and proc.poll() is None)
                or check_python_pid(self._is_python_pid, target_pid, target_started)
            )
            if not still_alive:
                self._proc = None
                self._terminal_state = STATE_STOPPED
                self._last_exit_code = None
                self._last_status = None
                self.stop_path.unlink(missing_ok=True)
                self.status_path.unlink(missing_ok=True)
                lock_pid, _ = self._read_live_lock_pid()
                if lock_pid is None:
                    self.lock_path.unlink(missing_ok=True)
                return True
            return False

    def restart(
        self,
        timeout: float = 20.0,
        on_timeout_force: Callable[[], bool] | None = None,
    ) -> bool:
        """Stop the bot (optionally force-stopping on timeout) and start it again."""
        if not self.stop(timeout=timeout):
            if on_timeout_force is not None and on_timeout_force():
                if not self.force_stop():
                    return False
            else:
                self.cancel_stop()
                return False
        return self.start()

    def _status_from_file(self, expected_pid: int | None, default_state: str, *, external: bool) -> BotStatus:
        data = self._read_json(self.status_path, retry=True)
        if data and (expected_pid is None or data.get("pid") == expected_pid):
            raw_state = str(data.get("state", "")).lower()
            state = STATE_STOPPING if self._stopping else _STATUS_MAP.get(raw_state, default_state)
            started = data.get("started")
            started_int = started if isinstance(started, int) else None
            uptime = max(0, int(time.time()) - started_int) if started_int is not None else 0
            pid = data.get("pid") if isinstance(data.get("pid"), int) else expected_pid
            ollama_model = str(data.get("ollama_model", "") or "")
            llm_model = str(data.get("llm_model", "") or "") or ollama_model
            status = BotStatus(
                state=state,
                external=external,
                pid=pid,
                started=started_int,
                uptime=uptime,
                discord_ready=bool(data.get("discord_ready", False)),
                llm_ok=bool(data.get("llm_ok", False)),
                voice_sessions=int(data.get("voice_sessions", 0) or 0),
                llm_provider=str(data.get("llm_provider", "") or ""),
                llm_model=llm_model,
                ollama_model=ollama_model,
                llm_last_provider=str(data.get("llm_last_provider", "") or ""),
                llm_last_model=str(data.get("llm_last_model", "") or ""),
                llm_fallback_active=bool(data.get("llm_fallback_active", False)),
            )
            self._last_status = status
            return status

        prev = self._last_status
        if prev is not None and (expected_pid is None or prev.pid == expected_pid):
            state = STATE_STOPPING if self._stopping else prev.state
            uptime = max(0, int(time.time()) - prev.started) if prev.started is not None else prev.uptime
            return BotStatus(
                state=state,
                external=external,
                pid=prev.pid if prev.pid is not None else expected_pid,
                started=prev.started,
                uptime=uptime,
                discord_ready=prev.discord_ready,
                llm_ok=prev.llm_ok,
                voice_sessions=prev.voice_sessions,
                llm_provider=prev.llm_provider,
                llm_model=prev.llm_model,
                ollama_model=prev.ollama_model,
                llm_last_provider=prev.llm_last_provider,
                llm_last_model=prev.llm_last_model,
                llm_fallback_active=prev.llm_fallback_active,
            )

        state = STATE_STOPPING if self._stopping else default_state
        return BotStatus(state=state, external=external, pid=expected_pid)

    def poll(self) -> BotStatus:
        """Inspect process state and status.json, handling crash auto-restarts."""
        with self._lock:
            if self._proc is not None:
                rc = self._proc.poll()
                if rc is None:
                    return self._status_from_file(None, STATE_STARTING, external=False)

                # Managed process has exited.
                self._proc = None
                self._last_exit_code = rc
                self._last_status = None
                was_stop_requested = self._stop_requested
                self._stopping = False
                self._stop_requested = False

                if was_stop_requested or rc == EXIT_OK:
                    self._terminal_state = STATE_STOPPED
                elif rc == EXIT_ALREADY_RUNNING:
                    self._terminal_state = STATE_ALREADY_RUNNING
                else:
                    now = self._clock()
                    cutoff = now - self.restart_window_sec
                    self._restart_timestamps = [t for t in self._restart_timestamps if t >= cutoff]
                    if self.auto_restart and len(self._restart_timestamps) < self.max_restarts:
                        self._restart_timestamps.append(now)
                        self._spawn()
                        return BotStatus(state=STATE_STARTING, pid=self._proc.pid)
                    self._terminal_state = STATE_CRASHED

            ext_pid, ext_started = self._read_live_lock_pid()
            if ext_pid is not None:
                if self._terminal_state == STATE_ALREADY_RUNNING:
                    status = self._status_from_file(ext_pid, STATE_ALREADY_RUNNING, external=True)
                    return BotStatus(
                        state=STATE_ALREADY_RUNNING,
                        external=True,
                        pid=ext_pid,
                        started=status.started or ext_started,
                        uptime=status.uptime,
                        discord_ready=status.discord_ready,
                        llm_ok=status.llm_ok,
                        voice_sessions=status.voice_sessions,
                        exit_code=self._last_exit_code,
                        llm_provider=status.llm_provider,
                        llm_model=status.llm_model,
                        ollama_model=status.ollama_model,
                        llm_last_provider=status.llm_last_provider,
                        llm_last_model=status.llm_last_model,
                        llm_fallback_active=status.llm_fallback_active,
                    )
                status = self._status_from_file(ext_pid, STATE_RUNNING, external=True)
                if status.started is None and ext_started is not None:
                    return BotStatus(
                        state=status.state,
                        external=True,
                        pid=ext_pid,
                        started=ext_started,
                        uptime=max(0, int(time.time()) - ext_started),
                        discord_ready=status.discord_ready,
                        llm_ok=status.llm_ok,
                        voice_sessions=status.voice_sessions,
                        llm_provider=status.llm_provider,
                        llm_model=status.llm_model,
                        ollama_model=status.ollama_model,
                        llm_last_provider=status.llm_last_provider,
                        llm_last_model=status.llm_last_model,
                        llm_fallback_active=status.llm_fallback_active,
                    )
                return status

            if self._stopping:
                self._stopping = False
                self._stop_requested = False
                self._terminal_state = STATE_STOPPED

            self._last_status = None
            return BotStatus(state=self._terminal_state, exit_code=self._last_exit_code)
