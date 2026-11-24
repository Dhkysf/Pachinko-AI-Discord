"""Pre-flight checks, Ollama helpers, SQLite backup, and diagnostics for Pachinko Control."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import subprocess
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

from app.process import BotStatus, read_env_summary, tail_log_lines
from core.safety import redact_secrets

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_BACKUP_NAME_RE = re.compile(r"^pachinko-\d{8}-\d{6}\.db$")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""
    code: str = ""


def fetch_ollama_models(
    base_url: str,
    *,
    timeout: float = 2.5,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[bool, list[str], str]:
    """Query Ollama /api/tags and return (reachable, model_names, error)."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with opener(url, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if status != 200:
                return False, [], f"HTTP {status}"
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace"))
        models = [
            str(m["name"]).strip()
            for m in (data.get("models") or [])
            if isinstance(m, dict) and m.get("name")
        ]
        return True, models, ""
    except (urllib.error.URLError, OSError, TimeoutError, ValueError, TypeError) as exc:
        return False, [], str(exc)


def model_is_installed(model: str, installed: list[str]) -> bool:
    """Return True if model matches an installed Ollama model (allowing implicit :latest)."""
    target = model.strip()
    if not target:
        return False
    if target in installed:
        return True
    if ":" not in target and f"{target}:latest" in installed:
        return True
    return False


def start_ollama(
    *,
    which: Callable[[str], str | None] = shutil.which,
    localappdata: str | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
) -> tuple[bool, str]:
    """Launch Ollama desktop app if installed, else `ollama serve` hidden."""
    base = localappdata if localappdata is not None else os.getenv("LOCALAPPDATA", "")
    if base:
        desktop = Path(base) / "Programs" / "Ollama" / "ollama app.exe"
        if desktop.exists():
            popen(
                [str(desktop)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, f"Started {desktop.name}"

    cli = which("ollama")
    if cli:
        popen(
            [cli, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
        return True, "Started `ollama serve`"

    return False, "Ollama executable not found. Install Ollama first."


def warm_up_ollama(
    base_url: str,
    model: str,
    keep_alive: str = "15m",
    *,
    timeout: float = 60.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> bool:
    """Send a tiny keep-alive request to Ollama so the model is loaded in VRAM."""
    if not model.strip():
        return False
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = json.dumps(
        {
            "model": model.strip(),
            "prompt": "",
            "stream": False,
            "keep_alive": keep_alive or "15m",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener(req, timeout=timeout) as resp:
            return getattr(resp, "status", 200) == 200
    except Exception:
        return False


def is_port_free(host: str, port: int) -> bool:
    """Return True if TCP (host, port) can be bound right now."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host or "127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def run_preflight(
    root_dir: Path,
    *,
    selected_model: str = "",
    bot_running: bool = False,
    which: Callable[[str], str | None] = shutil.which,
    fetch_models: Callable[[str], tuple[bool, list[str], str]] = fetch_ollama_models,
    port_free: Callable[[str, int], bool] = is_port_free,
) -> tuple[list[CheckResult], list[str]]:
    """Run pre-flight checks and return (results, installed_ollama_models)."""
    env = read_env_summary(root_dir)
    results: list[CheckResult] = []
    installed: list[str] = []

    if not env["env_exists"]:
        results.append(
            CheckResult(
                name=".env & DISCORD_TOKEN",
                ok=False,
                detail=".env file is missing",
                hint="Click 'Edit settings' or run setup.bat to create .env.",
            )
        )
    elif not env["has_discord_token"]:
        results.append(
            CheckResult(
                name=".env & DISCORD_TOKEN",
                ok=False,
                detail="DISCORD_TOKEN is empty in .env",
                hint="Click 'Edit settings' and paste your Discord bot token.",
            )
        )
    else:
        results.append(
            CheckResult(
                name=".env & DISCORD_TOKEN",
                ok=True,
                detail=".env found and DISCORD_TOKEN is set",
            )
        )

    ffmpeg_path = which("ffmpeg")
    if ffmpeg_path:
        results.append(CheckResult(name="ffmpeg", ok=True, detail="Found on PATH"))
    else:
        results.append(
            CheckResult(
                name="ffmpeg",
                ok=False,
                detail="ffmpeg not found on PATH",
                hint="Install ffmpeg (winget install Gyan.FFmpeg) and restart the app.",
            )
        )

    if env["llm_provider"] != "ollama":
        results.append(
            CheckResult(
                name="Ollama",
                ok=True,
                detail=f"Skipped (LLM_PROVIDER={env['llm_provider']})",
            )
        )
    else:
        effective_model = selected_model.strip() or env["ollama_model"]
        reachable, installed, err = fetch_models(env["ollama_base_url"])
        if not reachable:
            results.append(
                CheckResult(
                    name="Ollama",
                    ok=False,
                    detail=f"Not reachable at {env['ollama_base_url']} ({err or 'connection failed'})",
                    hint="Click 'Start Ollama' or launch the Ollama app.",
                    code="ollama_down",
                )
            )
        elif not model_is_installed(effective_model, installed):
            results.append(
                CheckResult(
                    name="Ollama",
                    ok=False,
                    detail=f"Model '{effective_model}' is not installed",
                    hint=f"Run `ollama pull {effective_model}` or pick an installed model.",
                )
            )
        else:
            results.append(
                CheckResult(
                    name="Ollama",
                    ok=True,
                    detail=f"Reachable; model '{effective_model}' installed",
                )
            )

    port = int(env["panel_port"])
    host = str(env["panel_host"])
    if bot_running:
        results.append(
            CheckResult(
                name=f"Panel port {port}",
                ok=True,
                detail="In use by the running bot",
            )
        )
    elif port_free(host, port):
        results.append(
            CheckResult(
                name=f"Panel port {port}",
                ok=True,
                detail="Port is free",
            )
        )
    else:
        results.append(
            CheckResult(
                name=f"Panel port {port}",
                ok=False,
                detail=f"Port {port} is already in use",
                hint="Stop the other process or change PANEL_PORT in .env.",
            )
        )

    return results, installed


def backup_database(
    db_path: Path,
    backups_dir: Path,
    *,
    keep: int = 14,
    now: datetime | None = None,
) -> Path:
    """Create a consistent SQLite online backup and keep the newest `keep` backups."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    target = backups_dir / f"pachinko-{ts}.db"
    tmp = backups_dir / f".pachinko-{ts}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"

    src: sqlite3.Connection | None = None
    dst: sqlite3.Connection | None = None
    try:
        src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)
        dst = sqlite3.connect(str(tmp))
        src.backup(dst)
    finally:
        if dst is not None:
            dst.close()
        if src is not None:
            src.close()

    try:
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    if keep > 0:
        backups = sorted(
            (p for p in backups_dir.iterdir() if p.is_file() and _BACKUP_NAME_RE.match(p.name)),
            key=lambda p: p.name,
            reverse=True,
        )
        for old in backups[keep:]:
            old.unlink(missing_ok=True)

    return target


def collect_versions() -> dict[str, str]:
    """Return Python, py-cord, and yt-dlp versions."""
    versions = {"python": platform.python_version()}
    for pkg in ("py-cord", "yt-dlp"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def build_diagnostics(
    *,
    status: BotStatus,
    selected_model: str,
    env_model: str,
    preflight: list[CheckResult],
    log_path: Path,
    log_lines: int = 60,
) -> str:
    """Build a redacted diagnostics report for clipboard sharing."""
    versions = collect_versions()
    effective = selected_model.strip() or env_model
    lines = [
        "=== Pachinko Control Diagnostics ===",
        f"Bot state: {status.state} (external={status.external}, pid={status.pid}, uptime={status.uptime}s)",
        f"Discord ready: {status.discord_ready} | LLM ok: {status.llm_ok} | Voice sessions: {status.voice_sessions}",
        f"Versions: Python {versions['python']} | py-cord {versions['py-cord']} | yt-dlp {versions['yt-dlp']}",
        f"Model: selected={selected_model or '(use .env)'} | effective={effective} | running={status.ollama_model or '-'}",
        "",
        "--- Pre-flight ---",
    ]
    for item in preflight:
        tag = "OK" if item.ok else "PROBLEM"
        lines.append(f"[{tag}] {item.name}: {item.detail}")
        if not item.ok and item.hint:
            lines.append(f"  Hint: {item.hint}")

    lines.append("")
    lines.append(f"--- Last {log_lines} log lines ---")
    tail = tail_log_lines(log_path, max_lines=log_lines)
    lines.append(tail if tail else "(no log file)")

    return redact_secrets("\n".join(lines), "[REDACTED]")
