"""Autonomy & Action Risk Gate.

Setiap aksi berdampak (tool call / inisiatif mandiri) melewati Gate:
- `low`    : Jalan langsung (cari info, catat memori, ganti ekspresi).
- `medium` : Dibatasi cooldown & cek izin privasi (pasang timer, antre lagu).
- `high`   : Masuk antrean persetujuan Owner (bisa di-approve lewat Discord / Web Panel).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from config import config
from core.bus import EventBus
from core.store import Store


@dataclass
class PendingApproval:
    id: str
    capability: str
    args: dict[str, Any]
    requester_id: str
    requester_name: str
    reason: str
    created_at: float
    status: str = "pending"   # 'pending' | 'approved' | 'denied'


RISK_LEVELS = {
    "web_search": "low",
    "web_read": "low",
    "remember_note": "low",
    "set_reminder": "medium",
    "queue_song": "medium",
    "proactive_dm": "high",
}


class ActionGate:
    def __init__(self, store: Store, bus: EventBus) -> None:
        self.store = store
        self.bus = bus
        self.paused: bool = not config.autonomy_enabled
        self._cooldowns: dict[str, float] = {}
        self.approvals: dict[str, PendingApproval] = {}

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
            return False, "Otonomi sedang dijeda oleh admin."

        risk = RISK_LEVELS.get(capability, "medium")
        is_owner = requester_id.isdigit() and int(requester_id) in config.owner_ids

        if risk == "medium":
            cd_key = f"{capability}:{requester_id}"
            now = time.time()
            last = self._cooldowns.get(cd_key, 0.0)
            if not is_owner and now - last < 8.0:
                return False, f"Tunggu sebentar sebelum memakai `{capability}` lagi."
            self._cooldowns[cd_key] = now

        if risk == "high" and not is_owner:
            req = PendingApproval(
                id=uuid.uuid4().hex[:8],
                capability=capability,
                args=args,
                requester_id=requester_id,
                requester_name=requester_name,
                reason="Aksi berisiko tinggi membutuhkan izin owner",
                created_at=time.time(),
            )
            self.approvals[req.id] = req
            await self.bus.emit("gate.approval_requested", **asdict(req))
            return False, f"Aksi `{capability}` menunggu persetujuan owner (ID: {req.id})."

        await self.bus.emit("gate.decision", capability=capability, verdict="allow", risk=risk, user=requester_name)
        return True, "ok"

    async def resolve_approval(self, approval_id: str, approve: bool) -> PendingApproval | None:
        item = self.approvals.get(approval_id)
        if not item:
            return None
        item.status = "approved" if approve else "denied"
        await self.bus.emit("gate.approval_resolved", id=approval_id, status=item.status)
        return item
