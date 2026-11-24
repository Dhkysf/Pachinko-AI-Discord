"""Concurrency and transaction atomicity tests for Store and EconomyAndPlay."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.store import Store
from play.economy import EconomyAndPlay


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, topic: str, **payload) -> None:
        self.events.append((topic, payload))


class FakeHeart:
    def __init__(self) -> None:
        self.touch_calls: list[tuple[str, str, str]] = []
        self.gift_calls: list[tuple[str, str, int]] = []

    async def touch_interaction(self, user_id: str, display_name: str, modality: str = "text") -> dict:
        self.touch_calls.append((str(user_id), display_name, modality))
        return {"bond": 10.0, "tier": "Orang Asing"}

    async def add_gift_bonus(self, user_id: str, display_name: str, amount: int) -> dict:
        self.gift_calls.append((str(user_id), display_name, amount))
        return {"bond": 10.0, "tier": "Orang Asing"}


class TestEconomyAtomicity(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test_economy.db"
        self.store = Store(self.db_path)
        await self.store.init()
        self.bus = FakeBus()
        self.heart = FakeHeart()
        self.eco = EconomyAndPlay(self.store, self.bus, self.heart)  # type: ignore[arg-type]

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self._tmp.cleanup()

    async def test_concurrent_get_wallet_first_use(self) -> None:
        w1, w2 = await asyncio.gather(
            self.eco.get_wallet("u0", "Zoe"),
            self.eco.get_wallet("u0", "Zoe"),
        )
        self.assertEqual(w1["balance"], 100)
        self.assertEqual(w2["balance"], 100)

    async def test_concurrent_daily_checkin_single_reward(self) -> None:
        r1, r2 = await asyncio.gather(
            self.eco.daily_checkin("u1", "Alice"),
            self.eco.daily_checkin("u1", "Alice"),
        )
        ok_res = [r for r in (r1, r2) if r["ok"]]
        fail_res = [r for r in (r1, r2) if not r["ok"]]
        self.assertEqual(len(ok_res), 1)
        self.assertEqual(len(fail_res), 1)
        self.assertEqual(ok_res[0]["reward"], 95)
        self.assertEqual(ok_res[0]["streak"], 1)
        self.assertEqual(ok_res[0]["balance"], 195)
        self.assertEqual(fail_res[0]["message"], "Lo udah ambil `/daily` hari ini! Balik lagi besok ya.")

        w = await self.eco.get_wallet("u1", "Alice")
        self.assertEqual(w["balance"], 195)
        cur = await self.store.conn.execute("SELECT COUNT(*) FROM economy_ledger WHERE user_id = ?", ("u1",))
        self.assertEqual((await cur.fetchone())[0], 1)
        self.assertEqual(len(self.heart.touch_calls), 1)
        self.assertEqual(len(self.bus.events), 1)

    async def test_concurrent_gift_exceeding_balance(self) -> None:
        r1, r2 = await asyncio.gather(
            self.eco.gift_to_character("u2", "Bob", 70),
            self.eco.gift_to_character("u2", "Bob", 70),
        )
        ok_res = [r for r in (r1, r2) if r["ok"]]
        fail_res = [r for r in (r1, r2) if not r["ok"]]
        self.assertEqual(len(ok_res), 1)
        self.assertEqual(len(fail_res), 1)
        self.assertEqual(ok_res[0]["balance"], 30)
        self.assertEqual(fail_res[0]["message"], "Saldo lo cuma **30 koin**.")

        w = await self.eco.get_wallet("u2", "Bob")
        self.assertEqual(w["balance"], 30)
        self.assertGreaterEqual(w["balance"], 0)
        cur = await self.store.conn.execute("SELECT COUNT(*) FROM economy_ledger WHERE user_id = ?", ("u2",))
        self.assertEqual((await cur.fetchone())[0], 1)
        self.assertEqual(len(self.heart.gift_calls), 1)

    async def test_concurrent_draw_card_coins_for_one(self) -> None:
        r1, r2 = await asyncio.gather(
            self.eco.draw_card("u3", "Cara", cost=60),
            self.eco.draw_card("u3", "Cara", cost=60),
        )
        ok_res = [r for r in (r1, r2) if r["ok"]]
        fail_res = [r for r in (r1, r2) if not r["ok"]]
        self.assertEqual(len(ok_res), 1)
        self.assertEqual(len(fail_res), 1)
        self.assertEqual(ok_res[0]["balance"], 40)
        self.assertEqual(fail_res[0]["message"], "Koin kurang! Butuh **60 koin**, saldo lo **40 koin**.")

        w = await self.eco.get_wallet("u3", "Cara")
        self.assertEqual(w["balance"], 40)
        cur = await self.store.conn.execute("SELECT COUNT(*) FROM collection_cards WHERE user_id = ?", ("u3",))
        self.assertEqual((await cur.fetchone())[0], 1)
        cur = await self.store.conn.execute("SELECT COUNT(*) FROM economy_ledger WHERE user_id = ?", ("u3",))
        self.assertEqual((await cur.fetchone())[0], 1)

    async def test_draw_card_returns_true_new_balance(self) -> None:
        await self.eco.get_wallet("u4", "Dan")
        _, res = await asyncio.gather(
            self.eco.add_coins("u4", "Dan", 50, "bonus"),
            self.eco.draw_card("u4", "Dan", cost=60),
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["balance"], 90)
        w = await self.eco.get_wallet("u4", "Dan")
        self.assertEqual(w["balance"], 90)
        gacha_events = [p for topic, p in self.bus.events if topic == "economy.tx" and p.get("reason") == "gacha_card"]
        self.assertEqual(len(gacha_events), 1)
        self.assertEqual(gacha_events[0]["balance"], 90)

    async def test_write_tx_exception_leaves_no_partial_rows(self) -> None:
        await self.eco.get_wallet("u5", "Fay")
        with self.assertRaises(RuntimeError):
            async with self.store.write_tx() as conn:
                await conn.execute("UPDATE economy_wallets SET balance = 999 WHERE user_id = ?", ("u5",))
                await conn.execute(
                    "INSERT INTO economy_ledger (user_id, delta, reason, ts) VALUES (?, ?, ?, ?)",
                    ("u5", 899, "partial", 1.0),
                )
                raise RuntimeError("boom")

        w = await self.eco.get_wallet("u5", "Fay")
        self.assertEqual(w["balance"], 100)
        cur = await self.store.conn.execute("SELECT COUNT(*) FROM economy_ledger WHERE user_id = ?", ("u5",))
        self.assertEqual((await cur.fetchone())[0], 0)

    async def test_concurrent_set_kv_does_not_commit_economy_op_halfway(self) -> None:
        await self.eco.get_wallet("u6", "Eve")
        orig_execute = self.store.conn.execute
        entered = asyncio.Event()
        kv_done_before_raise = False
        draw_raised = False

        async def wrapped_execute(sql: str, parameters=()):
            nonlocal draw_raised
            if "INSERT INTO collection_cards" in sql:
                entered.set()
                await asyncio.sleep(0.02)
                draw_raised = True
                raise RuntimeError("card insert failed")
            return await orig_execute(sql, parameters)

        async def concurrent_kv() -> None:
            nonlocal kv_done_before_raise
            await entered.wait()
            await self.store.set_kv("unrelated_key", "saved")
            kv_done_before_raise = not draw_raised

        with patch.object(self.store.conn, "execute", side_effect=wrapped_execute):
            results = await asyncio.gather(
                self.eco.draw_card("u6", "Eve", cost=60),
                concurrent_kv(),
                return_exceptions=True,
            )

        self.assertIsInstance(results[0], RuntimeError)
        self.assertIsNone(results[1])
        self.assertFalse(kv_done_before_raise)
        self.assertEqual(await self.store.get_kv("unrelated_key"), "saved")

        w = await self.eco.get_wallet("u6", "Eve")
        self.assertEqual(w["balance"], 100)
        cur = await self.store.conn.execute("SELECT COUNT(*) FROM economy_ledger WHERE user_id = ?", ("u6",))
        self.assertEqual((await cur.fetchone())[0], 0)
        cur = await self.store.conn.execute("SELECT COUNT(*) FROM collection_cards WHERE user_id = ?", ("u6",))
        self.assertEqual((await cur.fetchone())[0], 0)


if __name__ == "__main__":
    unittest.main()
