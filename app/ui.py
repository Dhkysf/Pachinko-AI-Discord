"""Tkinter desktop window for Pachinko Control."""
from __future__ import annotations

import os
import subprocess
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Callable

from PIL import ImageTk

from app.extras import (
    CheckResult,
    backup_database,
    build_diagnostics,
    fetch_models_for_source,
    fetch_ollama_models,
    ninerouter_dashboard_url,
    run_preflight,
    start_ollama,
    warm_up_ollama,
)
from app.process import (
    LLM_SOURCES,
    MODEL_SETTING_KEY,
    STATE_ALREADY_RUNNING,
    STATE_CRASHED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_STOPPING,
    BotProcessManager,
    BotStatus,
    effective_llm_selection,
    format_llm_status,
    format_uptime,
    llm_display_label,
    load_settings,
    read_env_summary,
    read_panel_port,
    save_settings,
    tail_log_lines,
)
from app.tray import TrayController, make_icon_image, status_color

_ENV_LABEL = "(use .env)"
_SOURCE_LABEL_TO_KEY = {_ENV_LABEL: "", "9router": "ninerouter", "Ollama": "ollama"}
_SOURCE_KEY_TO_LABEL = {v: k for k, v in _SOURCE_LABEL_TO_KEY.items()}

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
        self._warmed_run_key: tuple[int | None, int | None] | None = None
        self._model_cache: dict[str, list[str]] = {}
        self._model_errors: dict[str, str] = {}
        self._fetch_seq = 0

        self._sync_manager_llm()

        self.root.title("Pachinko Control")
        self.root.geometry("760x620")
        self.root.minsize(640, 480)
        self._icon_photo = ImageTk.PhotoImage(make_icon_image(status_color(STATE_STOPPED), size=32))
        self.root.iconphoto(True, self._icon_photo)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._sync_startup_checkbox()
        self._sync_model_controls(fetch_if_needed=True)
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
        self._llm_var = tk.StringVar(value="LLM (9router): error")
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
        style = ttk.Style(self.root)
        style.configure("RestartHighlight.TButton", foreground="#b45309", font=("Segoe UI", 9, "bold"))
        self._btn_restart = ttk.Button(row1, text="Restart bot", command=self.restart_bot)
        self._btn_panel = ttk.Button(row1, text="Open web panel", command=self.open_web_panel)
        self._btn_preflight = ttk.Button(row1, text="Check setup", command=self.check_setup)
        for btn in (self._btn_start, self._btn_stop, self._btn_restart, self._btn_panel, self._btn_preflight):
            btn.pack(side="left", padx=(0, 6))

        row2 = ttk.Frame(btn_frame)
        row2.pack(fill="x", pady=(0, 4))
        self._btn_backup = ttk.Button(row2, text="Backup now", command=self.backup_now)
        self._btn_backups = ttk.Button(row2, text="Open backups folder", command=self.open_backups_folder)
        self._btn_diag = ttk.Button(row2, text="Copy diagnostics", command=self.copy_diagnostics)
        self._btn_checks = ttk.Button(row2, text="Run checks", command=self.run_checks)
        self._btn_ytdlp = ttk.Button(row2, text="Update yt-dlp", command=self.update_ytdlp)
        for btn in (self._btn_backup, self._btn_backups, self._btn_diag, self._btn_checks, self._btn_ytdlp):
            btn.pack(side="left", padx=(0, 6))

        row3 = ttk.Frame(btn_frame)
        row3.pack(fill="x")
        self._btn_repair = ttk.Button(row3, text="Repair install", command=self.repair_install)
        self._btn_env = ttk.Button(row3, text="Edit settings", command=self.edit_settings)
        self._btn_logs = ttk.Button(row3, text="Open logs folder", command=self.open_logs_folder)
        self._btn_root = ttk.Button(row3, text="Open project folder", command=self.open_project_folder)
        for btn in (self._btn_repair, self._btn_env, self._btn_logs, self._btn_root):
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

        model_row = ttk.Frame(cfg_frame)
        model_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(model_row, text="LLM source:").pack(side="left", padx=(0, 6))
        saved_src = self.settings["llm_source"]
        self._var_source = tk.StringVar(value=_SOURCE_KEY_TO_LABEL.get(saved_src, _ENV_LABEL))
        self._source_combo = ttk.Combobox(
            model_row,
            textvariable=self._var_source,
            state="readonly",
            width=12,
            values=[_ENV_LABEL, "9router", "Ollama"],
        )
        self._source_combo.pack(side="left", padx=(0, 10))
        self._source_combo.bind("<<ComboboxSelected>>", self._on_source_selected)

        ttk.Label(model_row, text="Model:").pack(side="left", padx=(0, 6))
        self._var_model = tk.StringVar(value=_ENV_LABEL)
        self._model_combo = ttk.Combobox(
            model_row,
            textvariable=self._var_model,
            state="readonly",
            width=26,
            values=[_ENV_LABEL],
        )
        self._model_combo.pack(side="left", padx=(0, 6))
        self._model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        self._btn_refresh = ttk.Button(model_row, text="Refresh", command=self.refresh_models)
        self._btn_refresh.pack(side="left", padx=(0, 8))
        self._restart_needed_var = tk.StringVar(value="")
        ttk.Label(
            model_row,
            textvariable=self._restart_needed_var,
            foreground="#d97706",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        err_row = ttk.Frame(cfg_frame)
        err_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._model_err_var = tk.StringVar(value="")
        ttk.Label(err_row, textvariable=self._model_err_var, foreground="#dc2626").pack(side="left")
        self._btn_9router_dash = ttk.Button(
            err_row,
            text="Open 9router dashboard",
            command=self.open_ninerouter_dashboard,
        )

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

    def _selected_source(self) -> str:
        return self.settings.get("llm_source", "")

    def _selected_model(self) -> str:
        src = self._selected_source()
        if src not in LLM_SOURCES:
            return ""
        return self.settings.get(MODEL_SETTING_KEY[src], "")

    def _sync_manager_llm(self) -> None:
        self.manager.llm_source = self._selected_source()
        self.manager.llm_model = self._selected_model()

    def open_ninerouter_dashboard(self) -> None:
        env = read_env_summary(self.manager.root_dir)
        webbrowser.open(ninerouter_dashboard_url(env["ninerouter_base_url"]))

    def start_bot(self) -> None:
        def task() -> None:
            src = self._selected_source()
            results, models = run_preflight(
                self.manager.root_dir,
                selected_source=src,
                selected_model=self._selected_model(),
                bot_running=False,
            )
            if src in LLM_SOURCES and models:
                self.root.after(0, lambda s=src, m=models: self._apply_models(s, True, m, ""))
            if all(r.ok for r in results):
                self.manager.start()
            else:
                self.root.after(0, lambda: self._show_preflight_dialog(results, allow_start=True))

        self._run_bg(task)

    def check_setup(self) -> None:
        def task() -> None:
            src = self._selected_source()
            running = self._is_bot_active(self._last_status)
            results, models = run_preflight(
                self.manager.root_dir,
                selected_source=src,
                selected_model=self._selected_model(),
                bot_running=running,
            )
            if src in LLM_SOURCES and models:
                self.root.after(0, lambda s=src, m=models: self._apply_models(s, True, m, ""))
            self.root.after(0, lambda: self._show_preflight_dialog(results, allow_start=False))

        self._run_bg(task)

    def refresh_models(self) -> None:
        src = self._selected_source()
        if src not in LLM_SOURCES:
            return
        self._fetch_seq += 1
        seq = self._fetch_seq

        def worker() -> None:
            reachable, models, err = fetch_models_for_source(src, self.manager.root_dir, timeout=5.0)
            self.root.after(0, lambda: self._on_models_fetched(seq, src, reachable, models, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_models_fetched(
        self,
        seq: int,
        source: str,
        reachable: bool,
        models: list[str],
        err: str,
    ) -> None:
        if seq != self._fetch_seq or source != self._selected_source():
            return
        self._apply_models(source, reachable, models, err)

    def _apply_models(self, source: str, reachable: bool, models: list[str], err: str) -> None:
        if reachable:
            self._model_cache[source] = list(models)
        if err:
            self._model_errors[source] = err
        else:
            self._model_errors.pop(source, None)
        if source == self._selected_source():
            self._sync_model_controls(fetch_if_needed=False)

    def _sync_model_controls(self, *, fetch_if_needed: bool) -> None:
        src = self._selected_source()
        if src not in LLM_SOURCES:
            self._var_model.set(_ENV_LABEL)
            self._model_combo.configure(values=[_ENV_LABEL], state="disabled")
            self._btn_refresh.configure(state="disabled")
            self._model_err_var.set("")
            self._btn_9router_dash.pack_forget()
            return

        self._model_combo.configure(state="readonly")
        self._btn_refresh.configure(state="normal")
        if src == "ninerouter":
            self._btn_9router_dash.pack(side="right")
        else:
            self._btn_9router_dash.pack_forget()

        cached = self._model_cache.get(src, [])
        current = self._selected_model()
        values = [_ENV_LABEL, *cached]
        if current and current not in values:
            values.append(current)
        self._model_combo.configure(values=values)
        self._var_model.set(current or _ENV_LABEL)
        self._model_err_var.set(self._model_errors.get(src, ""))

        if fetch_if_needed and src not in self._model_cache:
            self.refresh_models()

    def _on_source_selected(self, _event: object = None) -> None:
        label = self._var_source.get().strip()
        src = _SOURCE_LABEL_TO_KEY.get(label, "")
        self.settings["llm_source"] = src
        self._sync_manager_llm()
        save_settings(self.settings_path, self.settings)
        self._sync_model_controls(fetch_if_needed=True)
        self._update_restart_needed(self._last_status)

    def _on_model_selected(self, _event: object = None) -> None:
        src = self._selected_source()
        if src not in LLM_SOURCES:
            return
        raw = self._var_model.get().strip()
        chosen = "" if raw == _ENV_LABEL else raw
        self.settings[MODEL_SETTING_KEY[src]] = chosen
        self._sync_manager_llm()
        save_settings(self.settings_path, self.settings)
        self._update_restart_needed(self._last_status)

    def _update_restart_needed(self, status: BotStatus) -> None:
        running_model = status.llm_model or status.ollama_model
        if not self._is_bot_active(status) or not (status.llm_provider or running_model):
            self._restart_needed_var.set("")
            self._btn_restart.configure(style="TButton")
            return
        env = read_env_summary(self.manager.root_dir)
        eff_src, eff_model = effective_llm_selection(self.settings, env)
        running_src = (status.llm_provider or env["llm_provider"]).strip().lower()
        mismatch = (eff_src != running_src) or (bool(running_model) and eff_model != running_model)
        self._restart_needed_var.set("Restart needed" if mismatch else "")
        self._btn_restart.configure(style="RestartHighlight.TButton" if mismatch else "TButton")

    @staticmethod
    def _is_bot_active(status: BotStatus) -> bool:
        return status.state in (STATE_RUNNING, STATE_STARTING, STATE_STOPPING, STATE_ALREADY_RUNNING) or status.external

    def _show_preflight_dialog(self, results: list[CheckResult], *, allow_start: bool) -> None:
        win = tk.Toplevel(self.root)
        win.title("Setup Check")
        win.geometry("560x320")
        win.transient(self.root)

        body = ttk.Frame(win, padding=10)
        body.pack(fill="both", expand=True)

        def render(items: list[CheckResult]) -> None:
            for child in body.winfo_children():
                child.destroy()

            for item in items:
                row = ttk.Frame(body)
                row.pack(fill="x", pady=(0, 6))
                is_info = item.code == "ollama_fallback_down"
                badge_text = "INFO" if is_info else ("OK" if item.ok else "PROBLEM")
                badge_color = "#6b7280" if is_info else ("#16a34a" if item.ok else "#dc2626")
                badge = ttk.Label(
                    row,
                    text=badge_text,
                    foreground=badge_color,
                    font=("Segoe UI", 9, "bold"),
                    width=9,
                )
                badge.pack(side="left", anchor="n")
                text_col = ttk.Frame(row)
                text_col.pack(side="left", fill="x", expand=True)
                ttk.Label(text_col, text=f"{item.name}: {item.detail}", wraplength=430, justify="left").pack(anchor="w")
                if (not item.ok or is_info) and item.hint:
                    ttk.Label(
                        text_col,
                        text=f"Fix: {item.hint}",
                        foreground="#4b5563",
                        wraplength=430,
                        justify="left",
                    ).pack(anchor="w")

            btn_row = ttk.Frame(body)
            btn_row.pack(fill="x", pady=(10, 0))

            def recheck() -> None:
                src = self._selected_source()
                new_res, models = run_preflight(
                    self.manager.root_dir,
                    selected_source=src,
                    selected_model=self._selected_model(),
                    bot_running=self._is_bot_active(self._last_status),
                )
                if src in LLM_SOURCES and models:
                    self._apply_models(src, True, models, "")
                render(new_res)

            if any((not r.ok) and r.code == "ollama_down" for r in items):
                def launch_ollama() -> None:
                    ok, msg = start_ollama()
                    if not ok:
                        messagebox.showerror("Start Ollama", msg, parent=win)
                        return
                    for _ in range(10):
                        time.sleep(0.5)
                        reachable, models, _ = fetch_ollama_models(
                            read_env_summary(self.manager.root_dir)["ollama_base_url"]
                        )
                        if reachable:
                            self._apply_models("ollama", True, models, "")
                            break
                    recheck()

                ttk.Button(btn_row, text="Start Ollama", command=launch_ollama).pack(side="left", padx=(0, 6))

            eff_src, _ = effective_llm_selection(self.settings, read_env_summary(self.manager.root_dir))
            if eff_src == "ninerouter":
                ttk.Button(
                    btn_row,
                    text="Open 9router dashboard",
                    command=self.open_ninerouter_dashboard,
                ).pack(side="left", padx=(0, 6))

            ttk.Button(btn_row, text="Re-check", command=recheck).pack(side="left", padx=(0, 6))
            if allow_start:
                def start_anyway() -> None:
                    win.destroy()
                    self._run_bg(lambda: self.manager.start())

                ttk.Button(btn_row, text="Start anyway", command=start_anyway).pack(side="right")
                ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="right", padx=(0, 6))
            else:
                ttk.Button(btn_row, text="Close", command=win.destroy).pack(side="right")

        render(results)

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
            subprocess.Popen(["cmd.exe", "/c", "setup.bat"], cwd=str(self.manager.root_dir), creationflags=_CREATE_NEW_CONSOLE)

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

    def open_backups_folder(self) -> None:
        backups_dir = self.manager.data_dir / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(backups_dir))

    def backup_now(self) -> None:
        def task() -> None:
            db_path = self.manager.data_dir / "pachinko.db"
            backups_dir = self.manager.data_dir / "backups"
            try:
                created = backup_database(db_path, backups_dir)
                msg = f"Backup saved: {created.name}"
                self.root.after(0, lambda: messagebox.showinfo("Backup now", msg, parent=self.root))
            except Exception as exc:
                self.root.after(0, lambda err=str(exc): messagebox.showerror("Backup now", err, parent=self.root))

        self._run_bg(task)

    def copy_diagnostics(self) -> None:
        def task() -> None:
            status = self.manager.poll()
            env = read_env_summary(self.manager.root_dir)
            src = self._selected_source()
            mdl = self._selected_model()
            env_src = env["llm_provider"]
            env_mdl = str(env.get(f"{env_src}_model", "") or "")
            preflight, _ = run_preflight(
                self.manager.root_dir,
                selected_source=src,
                selected_model=mdl,
                bot_running=self._is_bot_active(status),
            )
            text = build_diagnostics(
                status=status,
                selected_source=src,
                selected_model=mdl,
                env_source=env_src,
                env_model=env_mdl,
                preflight=preflight,
                log_path=self.manager.log_path,
            )

            def copy_to_clipboard() -> None:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update()
                messagebox.showinfo("Copy diagnostics", "Diagnostics copied to clipboard.", parent=self.root)

            self.root.after(0, copy_to_clipboard)

        self._run_bg(task)

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
        for btn in (self._btn_preflight, self._btn_backup, self._btn_diag, self._btn_checks, self._btn_ytdlp):
            btn.configure(state="disabled" if self._busy else "normal")

    def _refresh_status(self) -> None:
        prev = self._last_status
        status = self.manager.poll()
        self._last_status = status

        self._dot_canvas.itemconfig(self._dot_id, fill=status_color(status.state))
        label = f"{status.state} (external)" if status.external else status.state
        self._state_var.set(label)
        self._uptime_var.set(f"Uptime: {format_uptime(status.uptime)}" if status.uptime > 0 else "Uptime: -")
        self._discord_var.set(f"Discord: {'ready' if status.discord_ready else 'not ready'}")
        env = read_env_summary(self.manager.root_dir)
        primary_src = status.llm_provider or self._selected_source() or env["llm_provider"]
        src_name = llm_display_label(status.llm_last_provider or primary_src, status.llm_fallback_active)
        active_model = status.llm_model or status.ollama_model
        self._llm_var.set(format_llm_status(status, default_source=primary_src))
        self._voice_var.set(f"Voice sessions: {status.voice_sessions}")

        self._update_buttons(status)
        self._update_restart_needed(status)
        if self.tray is not None:
            self.tray.update(status)
            if status.state == STATE_CRASHED and prev.state != STATE_CRASHED:
                self.tray.notify("Pachinko bot crashed. Check logs or copy diagnostics.")
            elif (
                prev.llm_ok
                and not status.llm_ok
                and status.state in (STATE_RUNNING, STATE_STARTING)
            ):
                self.tray.notify(f"{src_name} health changed to error.")

        if status.discord_ready:
            run_key = (status.pid, status.started)
            if run_key != self._warmed_run_key:
                self._warmed_run_key = run_key
                if primary_src == "ollama":
                    model = active_model or self.settings["model_ollama"] or env["ollama_model"]
                    threading.Thread(
                        target=warm_up_ollama,
                        args=(env["ollama_base_url"], model, env["ollama_keep_alive"]),
                        daemon=True,
                    ).start()
        elif status.state in (STATE_STOPPED, STATE_CRASHED) and not status.external:
            self._warmed_run_key = None

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
