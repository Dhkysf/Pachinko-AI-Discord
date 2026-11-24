"""Autonomy & Action Risk Gate.

Every impactful action (tool call / autonomous initiative) passes through the Gate:
- `low`    : Runs immediately (web search, memory note, expression change).
- `medium` : Cooldown-limited & privacy-checked (set reminder, queue song).
- `high`   : Restricted to Owner only.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from config import config
from core.bus import EventBus
from core.store import Store

log = logging.getLogger("brain.gate")


RISK_LEVELS = {
    "web_search": "low",
    "remember_note": "low",
    "set_reminder": "medium",
    "queue_song": "medium",
}


class ActionGate:
    def __init__(self, store: Store, bus: EventBus) -> None:
        self.store = store
        self.bus = bus
        self.paused: bool = not config.autonomy_enabled
        self._cooldowns: dict[str, float] = {}

    async def set_paused(self, paused: bool) -> None:
        self.paused = paused
        await self.store.set_kv("autonomy_paused", paused)
        await self.bus.emit("gate.pause_changed", paused=paused)

    async def load(self) -> None:
        saved = await self.store.get_kv("autonomy_paused", None)
        if saved is not None:
            self.paused = bool(saved)

    async def check(
        self,
        capability: str,
        args: dict[str, Any],
        *,
        requester_id: str,
        requester_name: str,
        is_autonomous: bool = False,
    ) -> tuple[bool, str]:
        if is_autonomous and self.paused:
            await self.bus.emit("gate.decision", capability=capability, verdict="deny", reason="autonomy_paused")
            return False, "Autonomy is currently paused by admin."

        risk = RISK_LEVELS.get(capability)
        if risk is None:
            log.warning("Rejected unknown capability: %r (user=%s)", capability, requester_name)
            await self.bus.emit("gate.decision", capability=capability, verdict="deny", reason="unknown_capability")
            return False, f"Aksi `{capability}` tidak dikenali."
        is_owner = requester_id.isdigit() and int(requester_id) in config.owner_ids

        if risk == "medium":
            cd_key = f"{capability}:{requester_id}"
            now = time.time()
            last = self._cooldowns.get(cd_key, 0.0)
            if not is_owner and now - last < 8.0:
                return False, f"Wait a moment before using `{capability}` again."
            self._cooldowns[cd_key] = now

        if risk == "high" and not is_owner:
            await self.bus.emit("gate.decision", capability=capability, verdict="deny", reason="owner_only")
            return False, f"Action `{capability}` requires owner privileges."

        await self.bus.emit("gate.decision", capability=capability, verdict="allow", risk=risk, user=requester_name)
        return True, "ok"
