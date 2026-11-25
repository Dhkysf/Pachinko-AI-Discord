"""System tray icon (Pillow + pystray) and .ico generator for Pachinko Control."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pystray
from PIL import Image, ImageDraw, ImageFont

from app.process import (
    STATE_ALREADY_RUNNING,
    STATE_CRASHED,
    STATE_RUNNING,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_STOPPING,
    BotStatus,
    format_uptime,
    llm_display_label,
)

STATUS_COLORS: dict[str, str] = {
    STATE_RUNNING: "#22c55e",
    STATE_STARTING: "#eab308",
    STATE_STOPPING: "#eab308",
    STATE_STOPPED: "#9ca3af",
    STATE_CRASHED: "#ef4444",
    STATE_ALREADY_RUNNING: "#ef4444",
}

_FONT_CANDIDATES = ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")


def status_color(state: str) -> str:
    return STATUS_COLORS.get(state, "#9ca3af")


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_icon_image(color: str, size: int = 64) -> Image.Image:
    """Draw a crisp 'P' on a colored circle using 4x supersampling."""
    scale = 4
    canvas_size = size * scale
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = 2 * scale
    draw.ellipse((pad, pad, canvas_size - pad, canvas_size - pad), fill=color)

    font = _load_bold_font(int(canvas_size * 0.62))
    bbox = draw.textbbox((0, 0), "P", font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (canvas_size - tw) / 2 - bbox[0]
    ty = (canvas_size - th) / 2 - bbox[1]
    draw.text((tx, ty), "P", fill="#ffffff", font=font)

    return img.resize((size, size), Image.Resampling.LANCZOS)


def write_ico(path: Path, color: str = "#22c55e") -> Path:
    """Generate a multi-size Windows .ico file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = make_icon_image(color, size=256)
    img.save(path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def format_tooltip(status: BotStatus) -> str:
    """Build the tray tooltip string, e.g. 'Pachinko - Running - 9router: gpt-4o (2h 13m)'."""
    label = f"{status.state} (external)" if status.external else status.state
    parts = [f"Pachinko - {label}"]
    active = status.state in (STATE_RUNNING, STATE_STARTING, STATE_STOPPING, STATE_ALREADY_RUNNING)
    shown_provider = status.llm_last_provider or status.llm_provider
    shown_model = status.llm_last_model or status.llm_model
    if active and (shown_provider or shown_model):
        src = llm_display_label(shown_provider, status.llm_fallback_active) if shown_provider else ""
        if src and shown_model:
            parts.append(f"{src}: {shown_model}")
        else:
            parts.append(src or shown_model)
    tip = " - ".join(parts)
    if status.state in (STATE_RUNNING, STATE_STARTING, STATE_STOPPING) and status.uptime > 0:
        tip = f"{tip} ({format_uptime(status.uptime)})"
    return tip[:119] + "…" if len(tip) > 120 else tip


class TrayController:
    """Manages the pystray icon and dispatches menu clicks onto the UI thread."""

    def __init__(
        self,
        *,
        dispatch: Callable[[Callable[[], None]], None],
        on_show: Callable[[], None],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_restart: Callable[[], None],
        on_open_panel: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        self._dispatch = dispatch
        self._state = STATE_STOPPED
        self._external = False
        self._last_color = status_color(STATE_STOPPED)
        self._last_tooltip = "Pachinko - Stopped"

        menu = pystray.Menu(
            pystray.MenuItem("Show window", lambda _icon, _item: self._dispatch(on_show), default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Start",
                lambda _icon, _item: self._dispatch(on_start),
                enabled=lambda _item: self._can_start(),
            ),
            pystray.MenuItem(
                "Stop",
                lambda _icon, _item: self._dispatch(on_stop),
                enabled=lambda _item: self._can_stop(),
            ),
            pystray.MenuItem(
                "Restart",
                lambda _icon, _item: self._dispatch(on_restart),
                enabled=lambda _item: self._can_stop(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open web panel", lambda _icon, _item: self._dispatch(on_open_panel)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", lambda _icon, _item: self._dispatch(on_quit)),
        )
        self._icon = pystray.Icon(
            "pachinko",
            make_icon_image(self._last_color),
            self._last_tooltip,
            menu=menu,
        )

    def _can_start(self) -> bool:
        return self._state in (STATE_STOPPED, STATE_CRASHED) and not self._external

    def _can_stop(self) -> bool:
        return self._state in (STATE_RUNNING, STATE_STARTING, STATE_ALREADY_RUNNING) or self._external

    def start(self) -> None:
        self._icon.run_detached()

    def update(self, status: BotStatus) -> None:
        state_changed = status.state != self._state or status.external != self._external
        self._state = status.state
        self._external = status.external

        color = status_color(status.state)
        if color != self._last_color:
            self._last_color = color
            self._icon.icon = make_icon_image(color)

        tooltip = format_tooltip(status)
        if tooltip != self._last_tooltip:
            self._last_tooltip = tooltip
            self._icon.title = tooltip

        if state_changed:
            try:
                self._icon.update_menu()
            except Exception:
                pass

    def notify(self, message: str, title: str = "Pachinko Control") -> None:
        try:
            self._icon.notify(message, title)
        except Exception:
            pass

    def stop(self) -> None:
        try:
            self._icon.stop()
        except Exception:
            pass


if __name__ == "__main__":
    if "--write-ico" in sys.argv:
        idx = sys.argv.index("--write-ico")
        target = Path(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else Path(__file__).with_name("pachinko.ico")
        write_ico(target)
