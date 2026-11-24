"""Deterministic Triage & Channel Modes.

Menentukan apakah sebuah pesan di Discord membangunkan karakter (tanpa buang token LLM):
- `off`          : Mati total di channel ini.
- `mention-only` : Hanya merespons kalau di-mention, di-reply, DM, atau namanya disebut.
- `quiet`        : Bisa ikut nimbrung otomatis maks 2x/jam.
- `normal`       : Bisa ikut nimbrung otomatis maks 6x/jam.
- `lively`       : Aktif nimbrung otomatis maks 14x/jam.
"""
from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass

from brain.profile import CharacterProfile

MODE_HOURLY_CAP = {
    "off": 0,
    "mention-only": 0,
    "quiet": 2,
    "normal": 6,
    "lively": 14,
}


@dataclass(slots=True)
class TriageDecision:
    should_wake: bool
    reason: str          # 'direct_mention' | 'called_name' | 'ambient_glance' | 'ignored_*'
    direct: bool = False


class TriageEngine:
    def __init__(self) -> None:
        self._ambient_history: dict[str, deque[float]] = {}

    def normalize_wake_words(self, text: str, profile: CharacterProfile) -> str:
        """Perbaiki salah dengar STT pada nama panggilan (misal 'Era' -> 'Aira')."""
        out = text
        for wrong, right in profile.wake_aliases.items():
            out = re.sub(rf"\b{re.escape(wrong)}\b", right, out, flags=re.IGNORECASE)
        return out

    def is_addressed(self, text: str, profile: CharacterProfile, *, mentioned: bool = False, is_dm: bool = False) -> bool:
        if mentioned or is_dm:
            return True
        low = text.lower()
        names = [profile.display_name.lower(), *profile.nicknames]
        for n in names:
            if re.search(rf"\b{re.escape(n)}\b", low):
                return True
        return False

    def decide_text(
        self,
        *,
        room_id: str,
        text: str,
        mode: str,
        profile: CharacterProfile,
        mentioned: bool,
        is_dm: bool,
        autonomy_paused: bool,
        user_opted_out: bool,
    ) -> TriageDecision:
        if mode == "off" and not is_dm:
            return TriageDecision(False, "ignored_channel_off")

        addressed = self.is_addressed(text, profile, mentioned=mentioned, is_dm=is_dm)
        if addressed:
            return TriageDecision(True, "direct_mention" if (mentioned or is_dm) else "called_name", direct=True)

        if autonomy_paused:
            return TriageDecision(False, "ignored_autonomy_paused")
        if user_opted_out:
            return TriageDecision(False, "ignored_privacy_opt_out")

        cap = MODE_HOURLY_CAP.get(mode, 0)
        if cap <= 0 or len(text.strip()) < 8:
            return TriageDecision(False, "ignored_mode_ceiling")

        now = time.time()
        hist = self._ambient_history.setdefault(room_id, deque())
        while hist and now - hist[0] > 3600.0:
            hist.popleft()

        if len(hist) >= cap:
            return TriageDecision(False, "ignored_hourly_cap")

        min_gap = 3600.0 / (cap * 2.0)
        if hist and (now - hist[-1]) < min_gap:
            return TriageDecision(False, "ignored_cooldown")

        hist.append(now)
        return TriageDecision(True, "ambient_glance", direct=False)
