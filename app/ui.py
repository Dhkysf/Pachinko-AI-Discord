"""Tkinter desktop window for Pachinko Control."""
from __future__ import annotations

import os
import subprocess
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Callable

from PIL import ImageTk

from app.process import (
    STATE_ALREADY_RUNNING,
    STATE_CRASHED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_STOPPING,
    BotProcessManager,
    BotStatus,
    format_uptime,
    load_settings,
    read_panel_port,
    save_settings,
    tail_log_lines,
)
from app.tray import TrayController, make_icon_image, status_color

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def _startup_shortcut_path() -> Path:
    appdata = os.getenv("APPDATA", "")
    if not appdata:
        return Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "Pachinko.lnk"
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "Pachinko.lnk"


class PachinkoControlApp:
    """Thin Tkinter UI wired to BotProcessManager and TrayController."""

    def __init__(self, root: tk.Tk, manager: BotProcessManager) -> None:
        self.root = root
        self.manager = manager
        self.settings_path = manager.data_dir / "app_settings.json"
        self.settings = load_settings(self.settings_path)
        self.tray: TrayController | None = None

        self._busy = False
        self._closing = False
        self._tray_notified_once = False
        self._last_log_sig: tuple[int, int] | None = None
        self._last_status = BotStatus(state=STATE_STOPPED)

        self.root.title("Pachinko Control")
        self.root.geometry("760x560")
        self.root.minsize(640, 440)
        self._icon_photo = ImageTk.PhotoImage(make_icon_image(status_color(STATE_STOPPED), size=32))
        self.root.iconphoto(True, self._icon_photo)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._sync_startup_checkbox()
        self._poll_loop()

    def attach_tray(self, tray: TrayController) -> None:
        self.tray = tray

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # --- Status header ---
        status_frame = ttk.LabelFrame(outer, text="Status", padding=8)
        status_frame.pack(fill="x", pady=(0, 8))

        top_row = ttk.Frame(status_frame)
        top_row.pack(fill="x")

        self._dot_canvas = tk.Canvas(top_row, width=16, height=16, highlightthickness=0)
        self._dot_canvas.pack(side="left", padx=(0, 6))
        self._dot_id = self._dot_canvas.create_oval(2, 2, 14, 14, fill=status_color(STATE_STOPPED), outline="")

        self._state_var = tk.StringVar(value=STATE_STOPPED)
        ttk.Label(top_row, textvariable=self._state_var, font=("Segoe UI", 10, "bold")).pack(side="left")

        self._uptime_var = tk.StringVar(value="Uptime: -")
        ttk.Label(top_row, textvariable=self._uptime_var).pack(side="right")

        detail_row = ttk.Frame(status_frame)
        detail_row.pack(fill="x", pady=(6, 0))
        self._discord_var = tk.StringVar(value="Discord: not ready")
        self._llm_var = tk.StringVar(value="LLM (Ollama): error")
        self._voice_var = tk.StringVar(value="Voice sessions: 0")
        ttk.Label(detail_row, textvariable=self._discord_var).pack(side="left", padx=(0, 18))
        ttk.Label(detail_row, textvariable=self._llm_var).pack(side="left", padx=(0, 18))
        ttk.Label(detail_row, textvariable=self._voice_var).pack(side="left")

        # --- Action buttons ---
        btn_frame = ttk.Frame(outer)
        btn_frame.pack(fill="x", pady=(0, 8))

        row1 = ttk.Frame(btn_frame)
        row1.pack(fill="x", pady=(0, 4))
        self._btn_start = ttk.Button(row1, text="Start bot", command=self.start_bot)
        self._btn_stop = ttk.Button(row1, text="Stop bot", command=self.stop_bot)
        self._btn_restart = ttk.Button(row1, text="Restart bot", command=self.restart_bot)
        self._btn_panel = ttk.Button(row1, text="Open web panel", command=self.open_web_panel)
        for btn in (self._btn_start, self._btn_stop, self._btn_restart, self._btn_panel):
            btn.pack(side="left", padx=(0, 6))

        row2 = ttk.Frame(btn_frame)
        row2.pack(fill="x")
        self._btn_checks = ttk.Button(row2, text="Run checks", command=self.run_checks)
        self._btn_ytdlp = ttk.Button(row2, text="Update yt-dlp", command=self.update_ytdlp)
        self._btn_repair = ttk.Button(row2, text="Repair install", command=self.repair_install)
        self._btn_env = ttk.Button(row2, text="Edit settings", command=self.edit_settings)
        self._btn_logs = ttk.Button(row2, text="Open logs folder", command=self.open_logs_folder)
        self._btn_root = ttk.Button(row2, text="Open project folder", command=self.open_project_folder)
        for btn in (
            self._btn_checks,
            self._btn_ytdlp,
            self._btn_repair,
            self._btn_env,
            self._btn_logs,
            self._btn_root,
        ):
            btn.pack(side="left", padx=(0, 6))

        # --- Settings checkboxes ---
        cfg_frame = ttk.LabelFrame(outer, text="Settings", padding=8)
        cfg_frame.pack(fill="x", pady=(0, 8))

        self._var_start_open = tk.BooleanVar(value=self.settings["start_bot_on_open"])
        self._var_start_win = tk.BooleanVar(value=self.settings["start_with_windows"])
        self._var_auto_restart = tk.BooleanVar(value=self.settings["auto_restart_on_crash"])
        self._var_min_tray = tk.BooleanVar(value=self.settings["minimize_to_tray_on_close"])

        self.manager.auto_restart = self.settings["auto_restart_on_crash"]

        ttk.Checkbutton(
            cfg_frame,
            text="Start bot when the app opens",
            variable=self._var_start_open,
            command=self._on_settings_changed,
        ).grid(row=0, column=0, sticky="w", padx=(0, 16))
        ttk.Checkbutton(
            cfg_frame,
            text="Start the app with Windows",
            variable=self._var_start_win,
            command=self._on_startup_toggle,
        ).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(
            cfg_frame,
            text="Auto-restart on crash",
            variable=self._var_auto_restart,
            command=self._on_settings_changed,
        ).grid(row=1, column=0, sticky="w", padx=(0, 16), pady=(4, 0))
        ttk.Checkbutton(
            cfg_frame,
            text="Minimize to tray on close",
            variable=self._var_min_tray,
            command=self._on_settings_changed,
        ).grid(row=1, column=1, sticky="w", pady=(4, 0))

        # --- Log view ---
        log_frame = ttk.LabelFrame(outer, text="Logs (data/logs/pachinko.log)", padding=8)
        log_frame.pack(fill="both", expand=True)

        log_bar = ttk.Frame(log_frame)
        log_bar.pack(fill="x", pady=(0, 4))
        self._var_pause_scroll = tk.BooleanVar(value=False)
        ttk.Checkbutton(log_bar, text="Pause scroll", variable=self._var_pause_scroll).pack(side="right")

        self._log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap="none",
            font=("Consolas", 9),
            state="disabled",
            height=14,
        )
        self._log_text.pack(fill="both", expand=True)

    def _sync_startup_checkbox(self) -> None:
        exists = _startup_shortcut_path().exists()
        if exists != self._var_start_win.get():
            self._var_start_win.set(exists)
            self.settings["start_with_windows"] = exists
            save_settings(self.settings_path, self.settings)

    def _on_settings_changed(self) -> None:
        self.settings["start_bot_on_open"] = self._var_start_open.get()
        self.settings["auto_restart_on_crash"] = self._var_auto_restart.get()
        self.settings["minimize_to_tray_on_close"] = self._var_min_tray.get()
        self.manager.auto_restart = self.settings["auto_restart_on_crash"]
        save_settings(self.settings_path, self.settings)

    def _on_startup_toggle(self) -> None:
        want = self._var_start_win.get()
        shortcut = _startup_shortcut_path()
        if want:
            script = self.manager.root_dir / "create_shortcut.ps1"
            if not script.exists():
                messagebox.showerror("Startup shortcut", "create_shortcut.ps1 was not found.")
                self._var_start_win.set(False)
                return
            res = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-Destination",
                    str(shortcut.parent),
                ],
                cwd=str(self.manager.root_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
            if res.returncode != 0 or not shortcut.exists():
                messagebox.showerror("Startup shortcut", res.stderr.strip() or "Failed to create startup shortcut.")
                self._var_start_win.set(False)
                want = False
        else:
            try:
                shortcut.unlink(missing_ok=True)
            except OSError as exc:
                messagebox.showerror("Startup shortcut", f"Failed to remove startup shortcut: {exc}")
                self._var_start_win.set(True)
                want = True

        self.settings["start_with_windows"] = want
        save_settings(self.settings_path, self.settings)

    def _run_bg(self, fn: Callable[[], None]) -> None:
        if self._busy:
            return
        self._busy = True
        self._update_buttons(self._last_status)

        def worker() -> None:
            try:
                fn()
            finally:
                self.root.after(0, self._finish_bg)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_bg(self) -> None:
        self._busy = False
        self._refresh_status()

    def _ask_force_stop(self) -> bool:
        ev = threading.Event()
        result = [False]

        def prompt() -> None:
            result[0] = messagebox.askyesno(
                "Bot did not stop",
                "The bot did not exit within 20 seconds.\nForce stop?",
                parent=self.root,
            )
            ev.set()

        self.root.after(0, prompt)
        ev.wait()
        return result[0]

    def start_bot(self) -> None:
        def task() -> None:
            self.manager.start()

        self._run_bg(task)

    def stop_bot(self) -> None:
        def task() -> None:
            if not self.manager.stop(timeout=20.0):
                if self._ask_force_stop():
                    self.manager.force_stop()
                else:
                    self.manager.cancel_stop()

        self._run_bg(task)

    def restart_bot(self) -> None:
        def task() -> None:
            self.manager.restart(timeout=20.0, on_timeout_force=self._ask_force_stop)

        self._run_bg(task)

    def open_web_panel(self) -> None:
        port = read_panel_port(self.manager.root_dir)
        webbrowser.open(f"http://127.0.0.1:{port}")

    def run_checks(self) -> None:
        def task() -> None:
            py = str(self.manager.root_dir / ".venv" / "Scripts" / "python.exe")
            targets = [
                "brain", "core", "discord_bot", "heart", "memory", "play",
                "sing", "speech", "voice", "web", "app", "tests",
                "config.py", "main.py", "setup_models.py", "Pachinko.pyw",
            ]
            c_res = subprocess.run(
                [py, "-m", "compileall", "-q", *targets],
                cwd=str(self.manager.root_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
            u_res = subprocess.run(
                [py, "-m", "unittest", "discover", "tests", "-v"],
                cwd=str(self.manager.root_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
            combined = "\n".join(part for part in (c_res.stdout.strip(), u_res.stdout.strip()) if part)
            tail = "\n".join(combined.splitlines()[-40:])
            passed = c_res.returncode == 0 and u_res.returncode == 0
            title = "Checks: PASS" if passed else "Checks: FAIL"
            self.root.after(0, lambda: self._show_output_popup(title, tail or title, passed))

        self._run_bg(task)

    def update_ytdlp(self) -> None:
        def task() -> None:
            py = str(self.manager.root_dir / ".venv" / "Scripts" / "python.exe")
            res = subprocess.run(
                [py, "-m", "pip", "install", "-U", "yt-dlp", "-q"],
                cwd=str(self.manager.root_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
            ok = res.returncode == 0
            msg = "yt-dlp updated successfully." if ok else (res.stdout.strip() or "Failed to update yt-dlp.")
            self.root.after(
                0,
                lambda: messagebox.showinfo("Update yt-dlp", msg, parent=self.root)
                if ok
                else messagebox.showerror("Update yt-dlp", msg, parent=self.root),
            )

        self._run_bg(task)

    def _show_output_popup(self, title: str, text: str, passed: bool) -> None:
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("680x420")
        header = ttk.Label(
            win,
            text="PASS" if passed else "FAIL",
            foreground="#16a34a" if passed else "#dc2626",
            font=("Segoe UI", 11, "bold"),
            padding=8,
        )
        header.pack(anchor="w")
        box = scrolledtext.ScrolledText(win, wrap="word", font=("Consolas", 9))
        box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        box.insert("1.0", text)
        box.configure(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 8))

    def repair_install(self) -> None:
        bat = self.manager.root_dir / "setup.bat"
        if bat.exists():
            subprocess.Popen(["cmd.exe", "/c", str(bat)], cwd=str(self.manager.root_dir), creationflags=_CREATE_NEW_CONSOLE)

    def edit_settings(self) -> None:
        env_path = self.manager.root_dir / ".env"
        if not env_path.exists():
            example = self.manager.root_dir / ".env.example"
            if example.exists():
                env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        subprocess.Popen(["notepad.exe", str(env_path)], cwd=str(self.manager.root_dir))

    def open_logs_folder(self) -> None:
        logs_dir = self.manager.data_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(logs_dir))

    def open_project_folder(self) -> None:
        os.startfile(str(self.manager.root_dir))

    def show_window(self) -> None:
        self.root.deiconify()
        self.root.state("normal")
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(50, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()
        self._refresh_logs(force=True)

    def _on_close(self) -> None:
        if self._var_min_tray.get() and self.tray is not None:
            self.root.withdraw()
            if not self._tray_notified_once:
                self._tray_notified_once = True
                self.tray.notify("Pachinko is still running in the tray")
            return
        self.quit_app()

    def quit_app(self) -> None:
        if self._closing:
            return
        status = self.manager.poll()
        running = status.state in (STATE_RUNNING, STATE_STARTING, STATE_STOPPING, STATE_ALREADY_RUNNING) or status.external
        if running:
            choice = messagebox.askyesnocancel(
                "Quit Pachinko Control",
                "Stop the bot too?",
                parent=self.root,
            )
            if choice is None:
                return
            if choice is True:
                self._closing = True
                self._run_bg(self._stop_then_exit)
                return

        self._exit_now()

    def _stop_then_exit(self) -> None:
        if not self.manager.stop(timeout=20.0):
            if self._ask_force_stop():
                self.manager.force_stop()
        self.root.after(0, self._exit_now)

    def _exit_now(self) -> None:
        if self.tray is not None:
            self.tray.stop()
        self.root.quit()

    def _update_buttons(self, status: BotStatus) -> None:
        can_start = (not self._busy) and status.state in (STATE_STOPPED, STATE_CRASHED) and not status.external
        can_stop = (not self._busy) and (
            status.state in (STATE_RUNNING, STATE_STARTING, STATE_ALREADY_RUNNING) or status.external
        )
        self._btn_start.configure(state="normal" if can_start else "disabled")
        self._btn_stop.configure(state="normal" if can_stop else "disabled")
        self._btn_restart.configure(state="normal" if can_stop else "disabled")
        self._btn_checks.configure(state="disabled" if self._busy else "normal")
        self._btn_ytdlp.configure(state="disabled" if self._busy else "normal")

    def _refresh_status(self) -> None:
        status = self.manager.poll()
        self._last_status = status

        self._dot_canvas.itemconfig(self._dot_id, fill=status_color(status.state))
        label = f"{status.state} (external)" if status.external else status.state
        self._state_var.set(label)
        self._uptime_var.set(f"Uptime: {format_uptime(status.uptime)}" if status.uptime > 0 else "Uptime: -")
        self._discord_var.set(f"Discord: {'ready' if status.discord_ready else 'not ready'}")
        self._llm_var.set(f"LLM (Ollama): {'ok' if status.llm_ok else 'error'}")
        self._voice_var.set(f"Voice sessions: {status.voice_sessions}")

        self._update_buttons(status)
        if self.tray is not None:
            self.tray.update(status)

    def _refresh_logs(self, *, force: bool = False) -> None:
        if not force and not self.root.winfo_viewable():
            return
        log_path = self.manager.log_path
        try:
            st = log_path.stat()
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            sig = None
        if not force and sig == self._last_log_sig:
            return
        self._last_log_sig = sig

        content = tail_log_lines(log_path, max_lines=300)
        yview = self._log_text.yview()
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.insert("1.0", content)
        self._log_text.configure(state="disabled")
        if self._var_pause_scroll.get() and yview:
            self._log_text.yview_moveto(yview[0])
        else:
            self._log_text.see("end")

    def _poll_loop(self) -> None:
        self._refresh_status()
        self._refresh_logs()
        self.root.after(2000, self._poll_loop)
