"""Double-clickable Windows launcher for Pachinko Control (stdlib-only bootstrap)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"
SETUP_BAT = ROOT / "setup.bat"
_BOOTSTRAPPED_ENV = "PACHINKO_APP_BOOTSTRAPPED"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def _offer_setup(message: str) -> None:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        if messagebox.askyesno("Pachinko Control", message):
            if SETUP_BAT.exists():
                subprocess.Popen(
                    ["cmd.exe", "/c", "setup.bat"],
                    cwd=str(ROOT),
                    creationflags=_CREATE_NEW_CONSOLE,
                )
    finally:
        root.destroy()


def _launch_venv() -> None:
    """Re-run this launcher under .venv pythonw; offer setup if it dies right away."""
    env = os.environ.copy()
    env[_BOOTSTRAPPED_ENV] = "1"
    try:
        proc = subprocess.Popen(
            [str(VENV_PYTHONW), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
        code = proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        return  # still running: normal start
    except OSError as exc:
        _offer_setup(f"Pachinko could not start ({exc}). Repair with setup.bat?")
        return
    if code != 0:
        _offer_setup(f"Pachinko could not start (exit code {code}). The .venv may be broken. Repair with setup.bat?")


def _bootstrap() -> None:
    if not VENV_PYTHONW.exists():
        _offer_setup("Pachinko is not installed yet. Run setup now?")
        sys.exit(0)

    current = os.path.normcase(os.path.abspath(sys.executable))
    target = os.path.normcase(os.path.abspath(str(VENV_PYTHONW)))
    if current != target and os.environ.get(_BOOTSTRAPPED_ENV) != "1":
        _launch_venv()
        sys.exit(0)


_bootstrap()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

try:
    from app.__main__ import main
except ModuleNotFoundError as exc:
    _offer_setup(f"Missing dependency ({exc.name}). Run setup now?")
    sys.exit(1)

if __name__ == "__main__":
    raise SystemExit(main())
