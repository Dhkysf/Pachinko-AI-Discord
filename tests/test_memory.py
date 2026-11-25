"""Offline unit tests for memory/manager.py."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from core.store import Store
from memory.manager import VEC_DIM, MemoryManager, _cosine, _quote_fts_term, local_embed


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


if __name__ == "__main__":
    unittest.main()
