"""Layered Memory Manager (L1 Transcript, L2 Hybrid FTS5 + Local Vector, L3 Living Notebook, & Schedules)."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from typing import Any

from core.bus import EventBus
from core.store import Store

VEC_DIM = 128


def local_embed(text: str) -> list[float]:
    """Ultra-fast local character/word n-gram vector (<1ms, 0 MB download, resilient to typos & Indonesian slang)."""
    vec = [0.0] * VEC_DIM
    norm_text = re.sub(r"\s+", " ", text.lower()).strip()
    if not norm_text:
        return vec
    tokens = re.findall(r"[a-z0-9]+", norm_text)
    features = list(tokens)
    for tok in tokens:
        if len(tok) >= 4:
            features.extend(tok[i : i + 3] for i in range(len(tok) - 2))
    for feat in features:
        h = int.from_bytes(hashlib.blake2b(feat.encode("utf-8"), digest_size=4).digest(), "little")
        idx = h % VEC_DIM
        sign = 1.0 if (h & (1 << 31)) else -1.0
        vec[idx] += sign
    mag = math.sqrt(sum(x * x for x in vec))
    return [round(x / mag, 4) for x in vec] if mag > 0 else vec


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _quote_fts_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


class MemoryManager:
    def __init__(self, store: Store, bus: EventBus) -> None:
        self.store = store
        self.bus = bus

    # ---------------- L1: Transcript ----------------
    async def add_transcript(
        self,
        *,
        room_id: str,
        speaker_id: str,
        speaker_name: str,
        role: str,
        modality: str,
        content: str,
        emotion: str = "neutral",
        interrupted: int | bool = 0,
        heard_content: str | None = None,
    ) -> int:
        cur = await self.store.write(
            "INSERT INTO transcripts "
            "(room_id, speaker_id, speaker_name, role, modality, content, heard_content, interrupted, emotion, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(room_id),
                str(speaker_id),
                speaker_name,
                role,
                modality,
                content,
                heard_content,
                1 if interrupted else 0,
                emotion,
                time.time(),
            ),
        )
        row_id = int(cur.lastrowid)
        await self.bus.emit(
            "memory.transcript",
            id=row_id,
            room_id=str(room_id),
            speaker_name=speaker_name,
            role=role,
            modality=modality,
            content=content,
            emotion=emotion,
        )
        return row_id

    async def mark_interrupted(self, entry_id: int, heard_content: str) -> None:
        """Record only the words the user actually heard before interruption (barge-in)."""
        await self.store.write(
            "UPDATE transcripts SET interrupted = 1, heard_content = ? WHERE id = ?",
            (heard_content, entry_id),
        )

    async def recent_transcript(self, room_id: str, limit: int = 24) -> list[dict[str, Any]]:
        cur = await self.store.conn.execute(
            "SELECT * FROM transcripts WHERE room_id = ? ORDER BY id DESC LIMIT ?",
            (str(room_id), limit),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def clear_room_transcript(self, room_id: str) -> None:
        await self.store.write("DELETE FROM transcripts WHERE room_id = ?", (str(room_id),))

    # ---------------- L2: Hybrid Episodic/Semantic Memory ----------------
    async def remember(
        self,
        content: str,
        *,
        scope: str = "global",
        subject_id: str = "",
        importance: float = 0.6,
    ) -> int:
        clean = content.strip()
        if not clean:
            return 0
        vec = local_embed(clean)
        cur = await self.store.write(
            "INSERT INTO memories (scope, subject_id, content, importance, vec_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (scope, str(subject_id), clean, max(0.1, min(1.0, importance)), json.dumps(vec), time.time()),
        )
        return int(cur.lastrowid)

    async def recall(
        self,
        query: str,
        *,
        scopes: tuple[str, ...] = ("global",),
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        q_vec = local_embed(query)
        fts_hits: set[int] = set()
        words = [w for w in re.findall(r"[a-zA-Z0-9]{3,}", query)][:6]
        if words:
            fts_q = " OR ".join(_quote_fts_term(w) for w in words)
            try:
                cur = await self.store.conn.execute(
                    "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ? LIMIT 25",
                    (fts_q,),
                )
                fts_hits = {int(r[0]) for r in await cur.fetchall()}
            except Exception:
                pass

        placeholders = ",".join("?" for _ in scopes)
        cur = await self.store.conn.execute(
            f"SELECT * FROM memories WHERE scope IN ({placeholders}) ORDER BY created_at DESC LIMIT 120",
            scopes,
        )
        candidates = await cur.fetchall()
        now = time.time()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in candidates:
            d = dict(row)
            vec = json.loads(d.pop("vec_json", "[]") or "[]")
            cos = max(0.0, _cosine(q_vec, vec))
            kw_bonus = 0.28 if d["id"] in fts_hits else 0.0
            age_days = max(0.0, (now - d["created_at"]) / 86400.0)
            recency = math.exp(-age_days / 21.0)
            score = 0.48 * cos + kw_bonus + 0.14 * recency + 0.10 * float(d["importance"])
            if score >= 0.16:
                d["score"] = round(score, 3)
                scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    async def delete_memories(self, ids: list[int]) -> int:
        unique_ids = list(dict.fromkeys(ids))
        if not unique_ids:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        cur = await self.store.write(
            f"DELETE FROM memories WHERE id IN ({placeholders})",
            unique_ids,
        )
        deleted = cur.rowcount
        if deleted:
            await self.bus.emit("memory.memories_deleted", count=deleted)
        return deleted

    # ---------------- L3: Living Notebook ----------------
    async def upsert_note(self, doc_key: str, fact: str, evidence: str = "") -> int:
        clean = fact.strip()
        if not clean:
            return 0
        async with self.store.write_tx() as conn:
            # Avoid exact duplicates
            cur = await conn.execute(
                "SELECT line_id FROM notebook WHERE doc_key = ? AND active = 1 AND lower(fact) = lower(?)",
                (doc_key, clean),
            )
            existing = await cur.fetchone()
            if existing:
                return int(existing["line_id"])

            cur = await conn.execute(
                "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
                (doc_key, clean, evidence[:200], time.time()),
            )
            row_id = int(cur.lastrowid)
        await self.bus.emit("memory.note_added", doc_key=doc_key, fact=clean)
        return row_id

    async def get_notes(self, doc_key: str, limit: int = 12) -> list[dict[str, Any]]:
        cur = await self.store.conn.execute(
            "SELECT * FROM notebook WHERE doc_key = ? AND active = 1 ORDER BY updated_at DESC LIMIT ?",
            (doc_key, limit),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def list_all_notes(self, limit: int = 80) -> list[dict[str, Any]]:
        cur = await self.store.conn.execute(
            "SELECT * FROM notebook WHERE active = 1 ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in await cur.fetchall()]

    async def delete_notes(self, line_ids: list[int]) -> int:
        unique_ids = list(dict.fromkeys(line_ids))
        if not unique_ids:
            return 0
        placeholders = ",".join("?" for _ in unique_ids)
        async with self.store.write_tx() as conn:
            cur = await conn.execute(
                f"SELECT line_id, doc_key, fact FROM notebook WHERE active = 1 AND line_id IN ({placeholders})",
                unique_ids,
            )
            rows = await cur.fetchall()
            if not rows:
                deleted = 0
            else:
                active_ids = [int(r["line_id"]) for r in rows]
                act_ph = ",".join("?" for _ in active_ids)
                upd = await conn.execute(
                    f"UPDATE notebook SET active = 0, updated_at = ? WHERE active = 1 AND line_id IN ({act_ph})",
                    [time.time(), *active_ids],
                )
                deleted = upd.rowcount
                for r in rows:
                    doc_key = r["doc_key"]
                    fact = r["fact"]
                    suffix = f" — {fact}"
                    await conn.execute(
                        "DELETE FROM memories "
                        "WHERE (scope = ? AND (content = ? OR substr(content, -length(?)) = ?)) "
                        "OR (scope = 'global' AND content = ?)",
                        (doc_key, fact, suffix, suffix, fact),
                    )
        if deleted:
            await self.bus.emit("memory.notes_deleted", count=deleted)
        return deleted

    async def auto_capture_user_facts(self, user_id: str, user_name: str, text: str) -> None:
        """Lightweight capture of explicit self-facts without extra LLM cost."""
        stripped = text.strip()
        if not stripped or stripped.endswith("?") or re.search(r"\bkamu\s+suka\b", stripped, re.IGNORECASE):
            return
        patterns = [
            (r"\b(?:nama(?:ku| gue)|panggil (?:aku|gue))\s+([A-Za-z0-9_ -]{2,25})", "Minta dipanggil: {0}"),
            (r"\b(?:aku|gue)\s+(?:suka(?:\s+banget)?|hobi|doyan)\s+([^.!?\n]{3,60})", "Suka/hobi: {0}"),
            (r"\b(?:aku|gue)\s+(?:benci|nggak suka|alergi)\s+([^.!?\n]{3,60})", "Kurang suka/hindari: {0}"),
            (r"\b(?:aku|gue)\s+(?:kuliah|kerja|sekolah)\s+(?:di|ambil|sebagai)\s+([^.!?\n]{3,60})", "Aktivitas/latar: {0}"),
        ]
        for pat, tmpl in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                fact = tmpl.format(m.group(1).strip())
                await self.upsert_note(f"person:{user_id}", fact, evidence=f"{user_name}: {text[:120]}")
                await self.remember(
                    f"{user_name} — {fact}",
                    scope=f"person:{user_id}",
                    subject_id=str(user_id),
                    importance=0.75,
                )

    # ---------------- Schedules & Reminders ----------------
    async def add_schedule(
        self,
        *,
        kind: str,
        room_id: str,
        user_id: str,
        user_name: str,
        note: str,
        due_at: float,
    ) -> int:
        cur = await self.store.write(
            "INSERT INTO schedules (kind, room_id, user_id, user_name, note, due_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, str(room_id), str(user_id), user_name, note.strip(), due_at, time.time()),
        )
        return int(cur.lastrowid)

    async def pop_due_schedules(self) -> list[dict[str, Any]]:
        now = time.time()
        async with self.store.write_tx() as conn:
            cur = await conn.execute(
                "SELECT * FROM schedules WHERE done = 0 AND due_at <= ? ORDER BY due_at ASC LIMIT 10",
                (now,),
            )
            rows = [dict(r) for r in await cur.fetchall()]
            if rows:
                ids = [r["id"] for r in rows]
                q = ",".join("?" for _ in ids)
                await conn.execute(f"UPDATE schedules SET done = 1 WHERE id IN ({q})", ids)
        return rows
