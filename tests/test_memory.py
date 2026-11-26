"""Offline unit tests for memory/manager.py."""
from __future__ import annotations

import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from brain.tools import ToolBox
from config import config
from core.bus import EventBus
from core.store import (
    FACT_DEDUPE_MIGRATION_KEY,
    PERSON_TOPIC_MIGRATION_KEY,
    Store,
    fact_key,
    migrate_fact_duplicates,
    migrate_person_topic_notes,
)
from memory.manager import VEC_DIM, MemoryManager, _cosine, _quote_fts_term, local_embed
from web.server import create_web_app


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, topic: str, **payload) -> None:
        self.events.append((topic, payload))


class TestLocalEmbed(unittest.TestCase):
    def test_local_embed_properties(self) -> None:
        empty = local_embed("   ")
        self.assertEqual(len(empty), VEC_DIM)
        self.assertTrue(all(v == 0.0 for v in empty))

        v1 = local_embed("aku suka kopi hitam")
        v2 = local_embed("aku suka kopi hitam")
        v_sim = local_embed("suka minum kopi")
        v_diff = local_embed("cuaca mendung hujan deras")

        self.assertEqual(v1, v2)
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v1)), 1.0, places=2)
        self.assertGreater(_cosine(v1, v_sim), _cosine(v1, v_diff))


class TestMemoryManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._tmp.name) / "test_memory.db")
        await self.store.init()
        self.bus = FakeBus()
        self.mem = MemoryManager(self.store, self.bus)  # type: ignore[arg-type]

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self._tmp.cleanup()

    async def test_upsert_note_inserts_and_deduplicates_case_insensitively(self) -> None:
        self.assertEqual(await self.mem.upsert_note("person:u1", "   "), 0)
        id1 = await self.mem.upsert_note("person:u1", "Suka kopi", evidence="chat")
        id2 = await self.mem.upsert_note("person:u1", "suka kopi", evidence="chat2")
        self.assertGreater(id1, 0)
        self.assertEqual(id1, id2)
        self.assertEqual(len(await self.mem.get_notes("person:u1")), 1)
        self.assertEqual(len([e for e in self.bus.events if e[0] == "memory.note_added"]), 1)

    async def test_auto_capture_saves_explicit_self_fact(self) -> None:
        await self.mem.auto_capture_user_facts("u1", "Budi", "aku suka banget kopi")
        notes = await self.mem.get_notes("person:u1")
        self.assertEqual(len(notes), 1)
        self.assertIn("kopi", notes[0]["fact"].lower())

    async def test_auto_capture_saves_plain_aku_suka_kopi(self) -> None:
        await self.mem.auto_capture_user_facts("u1", "Budi", "aku suka kopi")
        await self.mem.auto_capture_user_facts("u1", "Budi", "gue suka ramen")
        await self.mem.auto_capture_user_facts("u1", "Budi", "aku nggak suka kopi")
        notes = await self.mem.get_notes("person:u1")
        facts = [n["fact"].lower() for n in notes]
        self.assertEqual(len(facts), 3)
        self.assertIn("suka/hobi: kopi", facts)
        self.assertIn("suka/hobi: ramen", facts)
        self.assertIn("kurang suka/hindari: kopi", facts)

    async def test_auto_capture_ignores_other_person_and_question_about_bot(self) -> None:
        await self.mem.auto_capture_user_facts("u1", "Andi", "si Budi suka kucing")
        await self.mem.auto_capture_user_facts("u1", "Andi", "si Budi suka banget kucing")
        await self.mem.auto_capture_user_facts("u1", "Andi", "kamu suka apa?")
        await self.mem.auto_capture_user_facts("u1", "Andi", "kamu suka kopi")
        self.assertEqual(await self.mem.get_notes("person:u1"), [])

    async def test_auto_capture_ignores_question_or_joke(self) -> None:
        for q in (
            "aku suka banget dimarahin ya?",
            "aku suka kopi gak?",
            "aku suka kopi kan?",
            "aku suka kopi nggak?",
        ):
            await self.mem.auto_capture_user_facts("u1", "Andi", q)
        self.assertEqual(await self.mem.get_notes("person:u1"), [])

    async def test_recall_ranks_relevant_first_respects_scope_and_handles_fts5_keywords(self) -> None:
        await self.mem.remember("Budi suka minum kopi arabika pagi hari", scope="global")
        await self.mem.remember("Cuaca hari ini cerah berawan", scope="global")
        await self.mem.remember("rahasia besar kopi spesial milik orang lain", scope="person:other")

        res = await self.mem.recall("kopi arabika", scopes=("global", "person:me"), limit=5)
        self.assertTrue(res)
        self.assertIn("kopi arabika", res[0]["content"])
        self.assertFalse(any(r["scope"] == "person:other" for r in res))

        fts_res = await self.mem.recall("not and near", scopes=("global",))
        self.assertIsInstance(fts_res, list)

    async def test_recall_quotes_fts5_keywords_and_double_quotes(self) -> None:
        mid = await self.mem.remember("Budi suka minum kopi arabika pagi hari", scope="global", importance=0.5)
        # Age the memory 200 days so recency + importance alone (< 0.16) cannot pass without FTS5 kw_bonus.
        await self.store.write("UPDATE memories SET created_at = created_at - ? WHERE id = ?", (200 * 86400, mid))

        res_lower = await self.mem.recall("not and near kopi", scopes=("global",))
        self.assertTrue(res_lower, "Expected FTS5 match for 'not and near kopi'")
        self.assertIn("kopi arabika", res_lower[0]["content"])

        res_upper = await self.mem.recall("NOT AND NEAR kopi", scopes=("global",))
        self.assertTrue(res_upper, "Expected FTS5 match for 'NOT AND NEAR kopi'")
        self.assertIn("kopi arabika", res_upper[0]["content"])

        res_quoted = await self.mem.recall('katanya "kopi" enak', scopes=("global",))
        self.assertTrue(res_quoted, "Expected FTS5 match for query containing double quotes")
        self.assertIn("kopi arabika", res_quoted[0]["content"])

        self.assertEqual(_quote_fts_term('ko"pi'), '"ko""pi"')

    async def test_delete_notes_hides_note_and_removes_twins(self) -> None:
        await self.mem.auto_capture_user_facts("1", "Orenne", "panggil aku Orenne")
        notes = await self.mem.get_notes("person:1")
        self.assertEqual(len(notes), 1)
        self.assertEqual(len(await self.mem.list_all_notes()), 1)
        self.assertTrue(await self.mem.recall("Orenne", scopes=("person:1",)))

        await self.mem.remember("cuaca cerah hari ini", scope="person:1")
        line_id = notes[0]["line_id"]
        self.assertEqual(await self.mem.delete_notes([line_id]), 1)
        self.assertEqual(await self.mem.get_notes("person:1"), [])
        self.assertEqual(await self.mem.list_all_notes(), [])
        recalled = await self.mem.recall("Orenne", scopes=("person:1",))
        self.assertFalse(any("Orenne" in r["content"] for r in recalled))
        self.assertIn(("memory.notes_deleted", {"count": 1}), self.bus.events)
        self.assertEqual(await self.mem.delete_notes([line_id]), 0)

        cur = await self.store.conn.execute("SELECT content FROM memories WHERE scope = 'person:1'")
        self.assertEqual([r["content"] for r in await cur.fetchall()], ["cuaca cerah hari ini"])

        g_id = await self.mem.upsert_note("self", "Suka teh hijau")
        await self.mem.remember("Suka teh hijau", scope="global")
        self.assertEqual(await self.mem.delete_notes([g_id]), 1)
        cur = await self.store.conn.execute("SELECT content FROM memories WHERE scope = 'global'")
        self.assertEqual(await cur.fetchall(), [])

    async def test_delete_memories_removes_fts_row(self) -> None:
        mid = await self.mem.remember("Fakta unik xylophoneunik99", scope="global")
        await self.mem.remember("Fakta lain biasa", scope="global")
        cur = await self.store.conn.execute(
            'SELECT rowid FROM memories_fts WHERE memories_fts MATCH \'"xylophoneunik99"\''
        )
        self.assertEqual(len(await cur.fetchall()), 1)

        self.assertEqual(await self.mem.delete_memories([mid]), 1)
        cur = await self.store.conn.execute(
            'SELECT rowid FROM memories_fts WHERE memories_fts MATCH \'"xylophoneunik99"\''
        )
        self.assertEqual(await cur.fetchall(), [])
        self.assertIn(("memory.memories_deleted", {"count": 1}), self.bus.events)

    async def test_upsert_note_allows_readding_soft_deleted_fact(self) -> None:
        id1 = await self.mem.upsert_note("person:u1", "Suka matcha")
        self.assertEqual(await self.mem.delete_notes([id1]), 1)
        id2 = await self.mem.upsert_note("person:u1", "Suka matcha")
        self.assertGreater(id2, id1)
        notes = await self.mem.get_notes("person:u1")
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["line_id"], id2)

    async def test_delete_unknown_and_empty_ids_return_zero(self) -> None:
        self.assertEqual(await self.mem.delete_notes([99999]), 0)
        self.assertEqual(await self.mem.delete_memories([99999]), 0)
        self.assertEqual(await self.mem.delete_notes([]), 0)
        self.assertEqual(await self.mem.delete_memories([]), 0)
        self.assertFalse(any(t in ("memory.notes_deleted", "memory.memories_deleted") for t, _ in self.bus.events))

    async def test_delete_notes_removes_remember_note_person_and_topic_twins_only(self) -> None:
        toolbox = ToolBox(self.mem, types.SimpleNamespace(check=AsyncMock(return_value=(True, "ok"))))  # type: ignore[arg-type]
        await toolbox.execute(
            "remember_note",
            {"fact": "Suka kopi susu gula aren", "target": "user"},
            room_id="r1",
            user_id="u1",
            user_name="Budi",
        )
        await toolbox.execute(
            "remember_note",
            {"fact": "Suka lagu anime malam", "target": "anime"},
            room_id="r1",
            user_id="u1",
            user_name="Budi",
        )
        await self.mem.remember("Budi: Suka kopi susu gula aren", scope="global", subject_id="u1")

        # Unrelated memories that must NOT be deleted
        await self.mem.remember("Budi bilang Suka kopi susu gula aren enak banget", scope="person:u1")
        await self.mem.remember("Kemarin bahas Suka lagu anime malam di voice chat", scope="global")
        await self.mem.remember("Andi: Suka kopi susu gula aren", scope="person:u2")

        all_notes = await self.mem.list_all_notes()
        person_id = next(n["line_id"] for n in all_notes if n["doc_key"] == "person:u1")
        topic_id = next(n["line_id"] for n in all_notes if n["doc_key"] == "topic:anime")

        self.assertEqual(await self.mem.delete_notes([person_id, topic_id]), 2)

        cur = await self.store.conn.execute("SELECT scope, content FROM memories ORDER BY id")
        remaining = [(r["scope"], r["content"]) for r in await cur.fetchall()]
        self.assertEqual(
            remaining,
            [
                ("person:u1", "Budi bilang Suka kopi susu gula aren enak banget"),
                ("global", "Kemarin bahas Suka lagu anime malam di voice chat"),
                ("person:u2", "Andi: Suka kopi susu gula aren"),
            ],
        )

        recalled_person = await self.mem.recall("kopi susu gula aren", scopes=("global", "person:u1"))
        self.assertFalse(any(r["content"] == "Budi: Suka kopi susu gula aren" for r in recalled_person))
        recalled_topic = await self.mem.recall("lagu anime malam", scopes=("global", "person:u1"))
        self.assertFalse(any(r["content"] == "Budi — Suka lagu anime malam" for r in recalled_topic))

    async def test_remember_note_resolves_person_targets_and_keeps_topics(self) -> None:
        toolbox = ToolBox(self.mem, types.SimpleNamespace(check=AsyncMock(return_value=(True, "ok"))))  # type: ignore[arg-type]
        await self.store.write(
            "INSERT INTO heart_person (user_id, display_name, nickname, last_seen) VALUES (?, ?, ?, ?)",
            ("100", "ther", "ther_nick", 1.0),
        )
        await self.store.write(
            "INSERT INTO heart_person (user_id, display_name, nickname, last_seen) VALUES (?, ?, ?, ?)",
            ("200", "Andi", "andi_n", 1.0),
        )

        # Speaker name, @Name, username, nickname, self words -> person:100
        for idx, tgt in enumerate(("ther", "@Ther", "@ther_dev", "ther_nick", "user", "orang", "aku", "saya", "gue", "")):
            fact = f"Fakta unik nomor {idx} buat ther"
            out = await toolbox.execute(
                "remember_note",
                {"fact": fact, "target": tgt},
                room_id="r1",
                user_id="100",
                user_name="ther",
                username="ther_dev",
            )
            self.assertIn("person:100", out, msg=f"target={tgt!r} -> {out!r}")

        notes_100 = await self.mem.get_notes("person:100", limit=20)
        self.assertEqual(len(notes_100), 10)

        # Unique match for another person in heart_person -> person:200
        out_andi = await toolbox.execute(
            "remember_note",
            {"fact": "Suka main valorant malam", "target": "Andi"},
            room_id="r1",
            user_id="100",
            user_name="ther",
        )
        self.assertIn("person:200", out_andi)
        self.assertEqual(len(await self.mem.get_notes("person:200")), 1)
        cur = await self.store.conn.execute(
            "SELECT scope, subject_id FROM memories WHERE content LIKE '%Suka main valorant malam%'"
        )
        mem_row = await cur.fetchone()
        self.assertIsNotNone(mem_row)
        self.assertEqual((mem_row["scope"], mem_row["subject_id"]), ("person:200", "200"))

        # Unknown topic -> topic:yorushika
        out_topic = await toolbox.execute(
            "remember_note",
            {"fact": "Band Jepang favorit banyak orang", "target": "Yorushika"},
            room_id="r1",
            user_id="100",
            user_name="ther",
        )
        self.assertIn("topic:yorushika", out_topic)
        self.assertEqual(len(await self.mem.get_notes("topic:yorushika")), 1)

        # Ambiguous name (two people named "ther") -> topic:ther
        await self.store.write(
            "INSERT INTO heart_person (user_id, display_name, nickname, last_seen) VALUES (?, ?, ?, ?)",
            ("101", "ther", "", 1.0),
        )
        for uid, uname in (("100", "ther"), ("300", "Budi")):
            out_ambig = await toolbox.execute(
                "remember_note",
                {"fact": f"Catatan ambigu dari {uid}", "target": "ther"},
                room_id="r1",
                user_id=uid,
                user_name=uname,
            )
            self.assertIn("topic:ther", out_ambig)

    async def test_migrate_person_topic_notes_moves_dedups_backups_and_is_idempotent(self) -> None:
        backups_dir = Path(self._tmp.name) / "backups"
        await self.store.write(
            "INSERT INTO heart_person (user_id, display_name, nickname, last_seen) VALUES (?, ?, ?, ?)",
            ("100", "ther", "ther_nick", 1.0),
        )
        await self.store.write(
            "INSERT INTO heart_person (user_id, display_name, nickname, last_seen) VALUES (?, ?, ?, ?)",
            ("201", "alex", "", 1.0),
        )
        await self.store.write(
            "INSERT INTO heart_person (user_id, display_name, nickname, last_seen) VALUES (?, ?, ?, ?)",
            ("202", "alex", "", 1.0),
        )

        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            ("person:100", "Suka kopi", "chat", 10.0),
        )
        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            ("topic:ther", "Suka ramen", "chat", 20.0),
        )
        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            ("topic:ther", "suka kopi", "chat", 21.0),
        )
        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            ("topic:ther_nick", "Suka ramen", "chat", 15.0),
        )
        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            ("topic:yorushika", "Band Jepang", "chat", 30.0),
        )
        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            ("topic:alex", "Suka game", "chat", 40.0),
        )
        await self.mem.remember("ther: Suka ramen", scope="person:100", subject_id="100")

        with self.assertLogs("core.store", level="INFO") as logs:
            res = await migrate_person_topic_notes(self.store, backups_dir=backups_dir)

        self.assertEqual(res, {"moved": 1, "duplicates": 2, "left_as_topic": 2})
        self.assertEqual(len(list(backups_dir.glob("pachinko-*.db"))), 1)

        log_text = "\n".join(logs.output)
        self.assertIn("ther", log_text)
        self.assertIn("yorushika", log_text)
        self.assertIn("alex", log_text)
        self.assertNotIn("Suka ramen", log_text)
        self.assertNotIn("Suka kopi", log_text)

        person_notes = await self.mem.get_notes("person:100", limit=10)
        self.assertEqual([n["fact"] for n in person_notes], ["Suka ramen", "Suka kopi"])
        self.assertEqual(person_notes[0]["updated_at"], 20.0)

        cur = await self.store.conn.execute("SELECT doc_key, fact, active FROM notebook ORDER BY line_id")
        rows = [(r["doc_key"], r["fact"], r["active"]) for r in await cur.fetchall()]
        self.assertIn(("topic:ther", "suka kopi", 0), rows)
        self.assertIn(("topic:ther_nick", "Suka ramen", 0), rows)
        self.assertIn(("topic:yorushika", "Band Jepang", 1), rows)
        self.assertIn(("topic:alex", "Suka game", 1), rows)

        cur_mem = await self.store.conn.execute("SELECT scope, content FROM memories")
        self.assertEqual([(r["scope"], r["content"]) for r in await cur_mem.fetchall()], [("person:100", "ther: Suka ramen")])
        self.assertIsNotNone(await self.store.get_kv(PERSON_TOPIC_MIGRATION_KEY))

        # Second run is a no-op
        res2 = await migrate_person_topic_notes(self.store, backups_dir=backups_dir)
        self.assertIsNone(res2)
        self.assertEqual(len(list(backups_dir.glob("pachinko-*.db"))), 1)

    async def test_auto_capture_and_remember_note_save_one_fact_in_any_order(self) -> None:
        toolbox = ToolBox(self.mem, types.SimpleNamespace(check=AsyncMock(return_value=(True, "ok"))))  # type: ignore[arg-type]
        uid = "957075726648111175"

        for order in ("auto_first", "tool_first"):
            with self.subTest(order=order):
                await self.store.write("DELETE FROM notebook")
                await self.store.write("DELETE FROM memories")

                if order == "auto_first":
                    await self.mem.auto_capture_user_facts(uid, "Narenee", "panggil aku Orenne")
                    await toolbox.execute(
                        "remember_note",
                        {"fact": "Minta dipanggil Orenne", "target": "user"},
                        room_id="r1",
                        user_id=uid,
                        user_name="Narenee",
                    )
                else:
                    await toolbox.execute(
                        "remember_note",
                        {"fact": "Minta dipanggil Orenne", "target": "user"},
                        room_id="r1",
                        user_id=uid,
                        user_name="Narenee",
                    )
                    await self.mem.auto_capture_user_facts(uid, "Narenee", "panggil aku Orenne")

                notes = await self.mem.get_notes(f"person:{uid}")
                self.assertEqual(len(notes), 1)
                cur = await self.store.conn.execute(
                    "SELECT content FROM memories WHERE scope = ?", (f"person:{uid}",)
                )
                mems = await cur.fetchall()
                self.assertEqual(len(mems), 1)

                self.assertEqual(await self.mem.delete_notes([notes[0]["line_id"]]), 1)
                cur = await self.store.conn.execute(
                    "SELECT content FROM memories WHERE scope = ?", (f"person:{uid}",)
                )
                self.assertEqual(await cur.fetchall(), [])

        await self.mem.auto_capture_user_facts(uid, "Narenee", "panggil aku Orenne")
        await toolbox.execute(
            "remember_note",
            {"fact": "Minta dipanggil Orenne", "target": "user"},
            room_id="r1",
            user_id=uid,
            user_name="Narenee",
        )
        await toolbox.execute(
            "remember_note",
            {"fact": "Suka kopi susu gula aren", "target": "user"},
            room_id="r1",
            user_id=uid,
            user_name="Narenee",
        )
        self.assertEqual(len(await self.mem.get_notes(f"person:{uid}")), 2)
        cur = await self.store.conn.execute(
            "SELECT content FROM memories WHERE scope = ?", (f"person:{uid}",)
        )
        self.assertEqual(len(await cur.fetchall()), 2)

    async def test_remember_skips_same_fact_within_30_days(self) -> None:
        self.assertEqual(
            fact_key("Narenee: Minta dipanggil Orenne"),
            fact_key("Narenee — Minta dipanggil: Orenne"),
        )
        id1 = await self.mem.remember("Narenee: Suka kopi", scope="person:1")
        id2 = await self.mem.remember("Narenee — suka kopi!", scope="person:1")
        self.assertEqual(id1, id2)

        cur = await self.store.conn.execute("SELECT id FROM memories WHERE scope = 'person:1'")
        self.assertEqual(len(await cur.fetchall()), 1)

        await self.mem.remember("Narenee — suka kopi!", scope="person:2")
        cur2 = await self.store.conn.execute("SELECT id FROM memories WHERE scope = 'person:2'")
        self.assertEqual(len(await cur2.fetchall()), 1)

        await self.store.write(
            "UPDATE memories SET created_at = created_at - ? WHERE id = ?",
            (31 * 86400, id1),
        )
        id3 = await self.mem.remember("Narenee — suka kopi!", scope="person:1")
        self.assertGreater(id3, id1)
        cur = await self.store.conn.execute("SELECT id FROM memories WHERE scope = 'person:1'")
        self.assertEqual(len(await cur.fetchall()), 2)

    async def test_migrate_fact_duplicates_keeps_newest_and_is_idempotent(self) -> None:
        backups_dir = Path(self._tmp.name) / "backups"
        uid = "957075726648111175"
        person_key = f"person:{uid}"

        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            (person_key, "Minta dipanggil: Orenne", "chat", 10.0),
        )
        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            (person_key, "Minta dipanggil Orenne", "chat", 11.0),
        )
        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            (person_key, "Suka kopi", "chat", 5.0),
        )
        await self.store.write(
            "INSERT INTO notebook (doc_key, fact, evidence, active, updated_at) VALUES (?, ?, ?, 1, ?)",
            ("topic:anime", "Minta dipanggil Orenne", "chat", 3.0),
        )

        await self.store.write(
            "INSERT INTO memories (scope, subject_id, content, importance, vec_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (person_key, uid, "Narenee — Minta dipanggil: Orenne", 0.75, "[]", 10.0),
        )
        await self.store.write(
            "INSERT INTO memories (scope, subject_id, content, importance, vec_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (person_key, uid, "Narenee: Minta dipanggil Orenne", 0.8, "[]", 11.0),
        )
        await self.store.write(
            "INSERT INTO memories (scope, subject_id, content, importance, vec_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (person_key, uid, "Narenee: Suka kopi", 0.8, "[]", 5.0),
        )
        await self.store.write(
            "INSERT INTO memories (scope, subject_id, content, importance, vec_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("person:other", "other", "Narenee: Minta dipanggil Orenne", 0.8, "[]", 12.0),
        )

        with self.assertLogs("core.store", level="INFO") as logs:
            res = await migrate_fact_duplicates(self.store, backups_dir=backups_dir)

        self.assertEqual(res, {"notes_deactivated": 1, "memories_deleted": 1})
        self.assertEqual(len(list(backups_dir.glob("pachinko-*.db"))), 1)
        self.assertNotIn("Orenne", "\n".join(logs.output))

        person_notes = await self.mem.get_notes(person_key, limit=10)
        self.assertEqual([n["fact"] for n in person_notes], ["Minta dipanggil Orenne", "Suka kopi"])
        self.assertEqual(len(await self.mem.get_notes("topic:anime", limit=10)), 1)

        cur = await self.store.conn.execute("SELECT content FROM memories WHERE scope = ?", (person_key,))
        self.assertEqual(
            {r["content"] for r in await cur.fetchall()},
            {"Narenee: Minta dipanggil Orenne", "Narenee: Suka kopi"},
        )
        cur_other = await self.store.conn.execute("SELECT content FROM memories WHERE scope = 'person:other'")
        self.assertEqual(len(await cur_other.fetchall()), 1)
        self.assertIsNotNone(await self.store.get_kv(FACT_DEDUPE_MIGRATION_KEY))

        res2 = await migrate_fact_duplicates(self.store, backups_dir=backups_dir)
        self.assertIsNone(res2)
        self.assertEqual(len(list(backups_dir.glob("pachinko-*.db"))), 1)

        orenne_note_id = next(n["line_id"] for n in person_notes if n["fact"] == "Minta dipanggil Orenne")
        self.assertEqual(await self.mem.delete_notes([orenne_note_id]), 1)
        cur = await self.store.conn.execute("SELECT content FROM memories WHERE scope = ?", (person_key,))
        self.assertEqual([r["content"] for r in await cur.fetchall()], ["Narenee: Suka kopi"])

    async def test_api_memory_includes_person_display_name(self) -> None:
        orig_token, orig_host, orig_port = config.panel_token, config.panel_host, config.panel_port
        config.panel_token = ""
        config.panel_host = "127.0.0.1"
        config.panel_port = 7878
        try:
            await self.store.write(
                "INSERT INTO heart_person (user_id, display_name, last_seen) VALUES (?, ?, ?)",
                ("1250", "ther", 1.0),
            )
            await self.mem.upsert_note("person:1250", "Suka kopi susu")
            await self.mem.upsert_note("person:9999", "Tidak ada di heart_person")
            await self.mem.upsert_note("topic:yorushika", "Band Jepang")

            prof = types.SimpleNamespace(
                profile_id="default",
                display_name="Pachinko",
                temperament="tsundere",
                default_channel_mode="quiet",
                identity_md="",
                style_md="",
                examples_md="",
                lore_entries=[],
            )
            app = create_web_app(
                store=self.store,
                bus=EventBus(),
                profiles=types.SimpleNamespace(current=prof),
                heart=types.SimpleNamespace(state_summary=lambda _t: {}),
                memory=self.mem,
                gate=types.SimpleNamespace(paused=False),
                brain=types.SimpleNamespace(),
                singing=types.SimpleNamespace(snapshot=lambda: {"current": None, "queue": []}),
            )
            client = TestClient(app, base_url="http://127.0.0.1:7878")
            resp = client.get("/api/memory")
            self.assertEqual(resp.status_code, 200)
            by_key = {n["doc_key"]: n for n in resp.json()["notebook"]}
            self.assertEqual(by_key["person:1250"].get("display_name"), "ther")
            self.assertIsNone(by_key["person:9999"].get("display_name"))
            self.assertIsNone(by_key["topic:yorushika"].get("display_name"))
        finally:
            config.panel_token = orig_token
            config.panel_host = orig_host
            config.panel_port = orig_port


if __name__ == "__main__":
    unittest.main()
