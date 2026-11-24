"""Heart Engine: Dynamic emotion simulation (half-life decay) & 9-Axis Relationship / Bond Ladder.

Core mechanism:
- At the end of its reply, the LLM may propose `<feel>self: joy medium | user: bond +, warmth +</feel>`.
- This module holds the reins: applying daily geometric damping (`0.75^n`) and daily caps
  so affection/closeness grows naturally over time instead of maxing out in a few chats.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import time
from typing import Any

from core.bus import EventBus
from core.store import Store

EMOTION_KEYS = (
    "joy", "anger", "sorrow", "fear", "surprise",
    "shy", "pride", "smug", "lonely", "curious", "grateful", "annoyed",
)

# Internal emotion -> standard voice expression map (9 expressions)
EMOTION_TO_EXPRESSION = {
    "joy": "happy",
    "grateful": "happy",
    "pride": "smug",
    "smug": "smug",
    "shy": "shy",
    "anger": "angry",
    "annoyed": "annoyed",
    "sorrow": "sad",
    "lonely": "sad",
    "fear": "surprised",
    "surprise": "surprised",
    "curious": "thinking",
}

BOND_TIERS: list[tuple[float, str]] = [
    (-100.0, "Bermusuhan"),
    (-20.0, "Jaga Jarak"),
    (0.0, "Orang Asing"),
    (25.0, "Kenalan Baru"),
    (75.0, "Teman Ngobrol"),
    (160.0, "Teman Akrab"),
    (300.0, "Teman Dekat"),
    (500.0, "Sobat Kental"),
    (750.0, "Orang Kepercayaan"),
    (920.0, "Tak Terpisahkan"),
]

_INTENSITY = {"weak": 0.22, "lemah": 0.22, "medium": 0.45, "sedang": 0.45, "strong": 0.75, "kuat": 0.75}
_FEEL_RE = re.compile(r"<feel>(.*?)</feel>", re.IGNORECASE | re.DOTALL)
_EMO_RE = re.compile(r"<emo:([a-z_]+)>", re.IGNORECASE)
_TRUNCATED_TAIL_RE = re.compile(
    r"(?:<feel\b.*|</feel\b[^>]*>?|<emo:[^>]*|</?emo\b[^>]*>?|</?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def bond_tier_name(bond: float) -> str:
    current = BOND_TIERS[2][1]
    for threshold, label in BOND_TIERS:
        if bond >= threshold:
            current = label
    return current


def extract_tags(raw_text: str) -> tuple[str, str | None, str | None]:
    """Extract clean text from `<emo:...>` and `<feel>...</feel>`."""
    emo_match = _EMO_RE.search(raw_text)
    emo = emo_match.group(1).lower() if emo_match else None

    feel_match = _FEEL_RE.search(raw_text)
    feel = feel_match.group(1).strip() if feel_match else None

    clean = _EMO_RE.sub("", raw_text)
    clean = _FEEL_RE.sub("", clean)
    clean = _TRUNCATED_TAIL_RE.sub("", clean)
    clean = clean.replace("<silence/>", "").strip()
    return clean, emo, feel


class HeartEngine:
    def __init__(self, store: Store, bus: EventBus, half_life_mins: float = 25.0) -> None:
        self.store = store
        self.bus = bus
        self.half_life_mins = max(5.0, half_life_mins)
        self.emotions: dict[str, float] = {k: 0.0 for k in EMOTION_KEYS}
        self.valence: float = 0.25
        self.arousal: float = 0.35
        self.energy: float = 0.80
        self.last_update: float = time.time()

    async def load(self) -> None:
        cur = await self.store.conn.execute("SELECT * FROM heart_self WHERE id = 1")
        row = await cur.fetchone()
        if row:
            saved = json.loads(row["emotions_json"])
            for k in EMOTION_KEYS:
                self.emotions[k] = float(saved.get(k, 0.0))
            self.valence = float(row["valence"])
            self.arousal = float(row["arousal"])
            self.energy = float(row["energy"])
            self.last_update = float(row["updated_at"])
            self._apply_decay()
        else:
            await self._save_self()

    def _apply_decay(self) -> None:
        now = time.time()
        elapsed_mins = max(0.0, (now - self.last_update) / 60.0)
        if elapsed_mins <= 0.05:
            return
        factor = math.pow(0.5, elapsed_mins / self.half_life_mins)
        for k in self.emotions:
            self.emotions[k] = round(self.emotions[k] * factor, 4)
            if self.emotions[k] < 0.02:
                self.emotions[k] = 0.0
        # Energy recovers slowly while resting
        self.energy = min(1.0, self.energy + 0.004 * elapsed_mins)
        self._recompute_mood()
        self.last_update = now

    def _recompute_mood(self) -> None:
        pos = self.emotions["joy"] + self.emotions["grateful"] + 0.7 * self.emotions["pride"] + 0.5 * self.emotions["curious"]
        neg = self.emotions["anger"] + self.emotions["sorrow"] + self.emotions["annoyed"] + 0.6 * self.emotions["fear"]
        self.valence = max(-1.0, min(1.0, round(0.20 + 0.65 * (pos - neg), 3)))
        act = self.emotions["surprise"] + self.emotions["anger"] + self.emotions["joy"] + self.emotions["shy"]
        self.arousal = max(0.0, min(1.0, round(0.25 + 0.55 * act, 3)))

    def dominant_emotion(self, temperament: str = "tsundere") -> str:
        self._apply_decay()
        best_k, best_v = max(self.emotions.items(), key=lambda kv: kv[1])
        if best_v < 0.15:
            return "neutral"
        # Tsundere masking: slight shyness/joy tends to be masked with shy expression
        if temperament == "tsundere" and best_k == "joy" and self.emotions["shy"] > 0.2:
            return "shy"
        return EMOTION_TO_EXPRESSION.get(best_k, "neutral")

    def state_summary(self, temperament: str = "tsundere") -> dict[str, Any]:
        self._apply_decay()
        active = {k: round(v, 2) for k, v in self.emotions.items() if v >= 0.08}
        return {
            "dominant": self.dominant_emotion(temperament),
            "valence": self.valence,
            "arousal": self.arousal,
            "energy": round(self.energy, 2),
            "active_emotions": active,
        }

    async def _save_self(self) -> None:
        await self.store.write(
            "INSERT INTO heart_self (id, emotions_json, valence, arousal, energy, updated_at) "
            "VALUES (1, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET emotions_json=excluded.emotions_json, "
            "valence=excluded.valence, arousal=excluded.arousal, energy=excluded.energy, updated_at=excluded.updated_at",
            (json.dumps(self.emotions), self.valence, self.arousal, self.energy, time.time()),
        )

    async def get_person(self, user_id: str, display_name: str = "User") -> dict[str, Any]:
        uid = str(user_id)
        cur = await self.store.conn.execute("SELECT * FROM heart_person WHERE user_id = ?", (uid,))
        row = await cur.fetchone()
        if not row:
            now = time.time()
            await self.store.write(
                "INSERT OR IGNORE INTO heart_person (user_id, display_name, last_seen) VALUES (?, ?, ?)",
                (uid, display_name, now),
            )
            cur = await self.store.conn.execute("SELECT * FROM heart_person WHERE user_id = ?", (uid,))
            row = await cur.fetchone()
        data = dict(row)
        data["tier"] = bond_tier_name(data["bond"])
        return data

    async def list_people(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = await self.store.conn.execute(
            "SELECT * FROM heart_person ORDER BY bond DESC, interactions DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["tier"] = bond_tier_name(d["bond"])
            out.append(d)
        return out

    async def touch_interaction(self, user_id: str, display_name: str, modality: str = "text") -> dict[str, Any]:
        """Record interaction & apply daily-damped baseline familiarity gain."""
        person = await self.get_person(user_id, display_name)
        today = dt.date.today().isoformat()
        counts = json.loads(person["daily_counts_json"] or "{}") if person["daily_date"] == today else {}

        talk_n = counts.get("talk", 0)
        damping = math.pow(0.80, talk_n)
        bond_gain = (1.6 if modality == "voice" else 0.9) * damping
        fam_gain = 1.2 * damping
        counts["talk"] = talk_n + 1

        new_bond = min(1000.0, round(person["bond"] + bond_gain, 2))
        new_fam = min(100.0, round(person["familiarity"] + fam_gain, 2))
        self.energy = max(0.15, self.energy - 0.01)

        await self.store.write(
            "UPDATE heart_person SET display_name=?, bond=?, familiarity=?, daily_date=?, "
            "daily_counts_json=?, interactions=interactions+1, last_seen=? WHERE user_id=?",
            (display_name, new_bond, new_fam, today, json.dumps(counts), time.time(), str(user_id)),
        )
        return await self.get_person(user_id, display_name)

    async def apply_feel_block(
        self,
        feel_text: str | None,
        explicit_emo: str | None,
        user_id: str | None,
        display_name: str = "User",
        temperament: str = "tsundere",
    ) -> str:
        """Process `<emo:...>` & `<feel>...</feel>` tags with geometric rate-limiting."""
        self._apply_decay()

        if explicit_emo:
            for k, expr_name in EMOTION_TO_EXPRESSION.items():
                if explicit_emo in (k, expr_name):
                    self.emotions[k] = min(1.0, self.emotions[k] + 0.30)
                    break

        if feel_text:
            clauses = [c.strip() for c in re.split(r"[|\n;]", feel_text) if c.strip()]
            for clause in clauses:
                low = clause.lower()
                if low.startswith(("self:", "aku:", "me:")):
                    rest = low.split(":", 1)[1].strip().split()
                    if rest and rest[0] in self.emotions:
                        strength = _INTENSITY.get(rest[1], 0.40) if len(rest) > 1 else 0.40
                        self.emotions[rest[0]] = min(1.0, round(self.emotions[rest[0]] + strength, 3))
                elif user_id and ":" in low:
                    rest = low.split(":", 1)[1].strip()
                    await self._apply_person_deltas(str(user_id), display_name, rest)

        self._recompute_mood()
        await self._save_self()
        emotion = explicit_emo if explicit_emo in EMOTION_TO_EXPRESSION.values() else self.dominant_emotion(temperament)
        await self.bus.emit("heart.updated", **self.state_summary(temperament))
        return emotion

    async def _apply_person_deltas(self, user_id: str, display_name: str, delta_spec: str) -> None:
        person = await self.get_person(user_id, display_name)
        today = dt.date.today().isoformat()
        counts = json.loads(person["daily_counts_json"] or "{}") if person["daily_date"] == today else {}

        updates: dict[str, float] = {}
        for item in delta_spec.split(","):
            tokens = item.strip().split()
            if len(tokens) < 2:
                continue
            field, sign_tok = tokens[0], tokens[1]
            direction = 1.0 if "+" in sign_tok else (-1.0 if "-" in sign_tok else 0.0)
            if direction == 0.0:
                continue

            cnt_key = f"{field}_{'+' if direction > 0 else '-'}"
            n = counts.get(cnt_key, 0)
            damp = math.pow(0.75, n)
            counts[cnt_key] = n + 1

            if field == "bond":
                step = (2.8 if direction > 0 else -5.5) * damp
                updates["bond"] = max(-100.0, min(1000.0, round(person["bond"] + step, 2)))
            elif field in ("trust", "familiarity"):
                step = (2.2 if direction > 0 else -4.0) * damp
                updates[field] = max(0.0, min(100.0, round(person[field] + step, 2)))
            elif field in ("warmth", "respect"):
                step = (0.08 if direction > 0 else -0.14) * damp
                updates[field] = max(-1.0, min(1.0, round(person[field] + step, 3)))
            elif field in ("ease", "grudge", "longing", "rapport"):
                step = (0.08 if direction > 0 else -0.08) * damp
                updates[field] = max(0.0, min(1.0, round(person[field] + step, 3)))

        if updates:
            set_sql = ", ".join(f"{k}=?" for k in updates)
            vals = [*updates.values(), today, json.dumps(counts), user_id]
            await self.store.write(
                f"UPDATE heart_person SET {set_sql}, daily_date=?, daily_counts_json=? WHERE user_id=?",
                vals,
            )
            updated = await self.get_person(user_id, display_name)
            await self.bus.emit("heart.bond_changed", user_id=user_id, display_name=display_name, bond=updated["bond"], tier=updated["tier"])

    async def add_gift_bonus(self, user_id: str, display_name: str, amount: int) -> dict[str, Any]:
        person = await self.get_person(user_id, display_name)
        bonus = min(8.0, round(math.log1p(max(1, amount)) * 1.4, 2))
        new_bond = min(1000.0, round(person["bond"] + bonus, 2))
        new_warmth = min(1.0, round(person["warmth"] + 0.08, 3))
        self.emotions["grateful"] = min(1.0, self.emotions["grateful"] + 0.35)
        self.emotions["joy"] = min(1.0, self.emotions["joy"] + 0.25)
        self._recompute_mood()
        await self._save_self()
        await self.store.write(
            "UPDATE heart_person SET bond=?, warmth=? WHERE user_id=?",
            (new_bond, new_warmth, str(user_id)),
        )
        return await self.get_person(user_id, display_name)
