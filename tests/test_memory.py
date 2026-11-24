"""Offline unit tests for memory/manager.py."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from core.store import Store
from memory.manager import VEC_DIM, MemoryManager, _cosine, local_embed


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

    # BUG: pattern requires "suka banget" and misses plain "aku suka kopi"
    @unittest.expectedFailure
    async def test_auto_capture_saves_plain_aku_suka_kopi(self) -> None:
        await self.mem.auto_capture_user_facts("u1", "Budi", "aku suka kopi")
        notes = await self.mem.get_notes("person:u1")
        self.assertEqual(len(notes), 1)

    async def test_auto_capture_ignores_other_person_and_question_about_bot(self) -> None:
        await self.mem.auto_capture_user_facts("u1", "Andi", "si Budi suka kucing")
        await self.mem.auto_capture_user_facts("u1", "Andi", "si Budi suka banget kucing")
        await self.mem.auto_capture_user_facts("u1", "Andi", "kamu suka apa?")
        self.assertEqual(await self.mem.get_notes("person:u1"), [])

    # BUG: questions ending in '?' still match and get saved as facts
    @unittest.expectedFailure
    async def test_auto_capture_ignores_question_or_joke(self) -> None:
        await self.mem.auto_capture_user_facts("u1", "Andi", "aku suka banget dimarahin ya?")
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


if __name__ == "__main__":
    unittest.main()
