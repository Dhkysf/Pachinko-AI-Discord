"""Offline unit tests for heart/engine.py."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.store import Store
from heart.engine import EMOTION_KEYS, HeartEngine, bond_tier_name, extract_tags


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, topic: str, **payload) -> None:
        self.events.append((topic, payload))


class TestHeartPure(unittest.TestCase):
    def test_extract_tags_cases(self) -> None:
        clean, emo, feel = extract_tags("<emo:happy>Halo! <feel>self: joy weak | user: bond +</feel>")
        self.assertEqual((clean, emo, feel), ("Halo!", "happy", "self: joy weak | user: bond +"))

        clean2, emo2, feel2 = extract_tags("<emo:shy>Eh makasih")
        self.assertEqual((clean2, emo2, feel2), ("Eh makasih", "shy", None))

        clean3, emo3, feel3 = extract_tags("Halo biasa")
        self.assertEqual((clean3, emo3, feel3), ("Halo biasa", None, None))

    def test_extract_tags_truncated_trailing_tag(self) -> None:
        clean_feel, _, _ = extract_tags("Halo! <feel>self: joy weak")
        self.assertEqual(clean_feel, "Halo!")
        clean_emo, _, _ = extract_tags("Halo! <emo:hap")
        self.assertEqual(clean_emo, "Halo!")
        clean_lt, _, _ = extract_tags("Halo! <")
        self.assertEqual(clean_lt, "Halo!")
        clean_mid, _, _ = extract_tags("Nilai 3 < 5 ya")
        self.assertEqual(clean_mid, "Nilai 3 < 5 ya")

    def test_bond_tier_name_boundaries(self) -> None:
        cases = [
            (-100.0, "Bermusuhan"),
            (-20.1, "Bermusuhan"),
            (-20.0, "Jaga Jarak"),
            (-0.1, "Jaga Jarak"),
            (0.0, "Orang Asing"),
            (24.9, "Orang Asing"),
            (25.0, "Kenalan Baru"),
            (74.9, "Kenalan Baru"),
            (75.0, "Teman Ngobrol"),
            (159.9, "Teman Ngobrol"),
            (160.0, "Teman Akrab"),
            (299.9, "Teman Akrab"),
            (300.0, "Teman Dekat"),
            (499.9, "Teman Dekat"),
            (500.0, "Sobat Kental"),
            (749.9, "Sobat Kental"),
            (750.0, "Orang Kepercayaan"),
            (919.9, "Orang Kepercayaan"),
            (920.0, "Tak Terpisahkan"),
            (1000.0, "Tak Terpisahkan"),
        ]
        for bond, expected in cases:
            self.assertEqual(bond_tier_name(bond), expected, msg=f"bond={bond}")


class TestHeartEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._tmp.name) / "test_heart.db")
        await self.store.init()
        self.bus = FakeBus()
        self.engine = HeartEngine(self.store, self.bus, half_life_mins=20.0)  # type: ignore[arg-type]

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self._tmp.cleanup()

    def test_emotion_decay_halves_after_one_half_life(self) -> None:
        clock = [10000.0]
        with patch("heart.engine.time.time", side_effect=lambda: clock[0]):
            self.engine.last_update = clock[0]
            self.engine.emotions["joy"] = 0.8
            clock[0] += 20 * 60.0
            self.engine._apply_decay()
            self.assertAlmostEqual(self.engine.emotions["joy"], 0.4, places=3)

            clock[0] += 200 * 60.0
            self.engine._apply_decay()
            self.assertEqual(self.engine.emotions["joy"], 0.0)

    async def test_geometric_daily_damping_and_asymmetric_steps(self) -> None:
        await self.engine.apply_feel_block("user: bond +", None, "u1", "Budi")
        b1 = (await self.engine.get_person("u1", "Budi"))["bond"]
        await self.engine.apply_feel_block("user: bond +", None, "u1", "Budi")
        b2 = (await self.engine.get_person("u1", "Budi"))["bond"]
        await self.engine.apply_feel_block("user: bond +", None, "u1", "Budi")
        b3 = (await self.engine.get_person("u1", "Budi"))["bond"]

        d1, d2, d3 = b1, b2 - b1, b3 - b2
        self.assertGreater(d1, d2)
        self.assertGreater(d2, d3)
        self.assertGreater(d3, 0.0)
        self.assertAlmostEqual(d2 / d1, 0.75, places=2)
        self.assertAlmostEqual(d3 / d2, 0.75, places=2)

        await self.engine.apply_feel_block("user: bond +", None, "u_pos", "Pos")
        pos_delta = (await self.engine.get_person("u_pos", "Pos"))["bond"]
        await self.engine.apply_feel_block("user: bond -", None, "u_neg", "Neg")
        neg_delta = (await self.engine.get_person("u_neg", "Neg"))["bond"]
        self.assertGreater(abs(neg_delta), pos_delta)

    async def test_values_stay_within_ranges(self) -> None:
        for _ in range(8):
            await self.engine.apply_feel_block(
                "self: joy strong | user: bond +, trust +, warmth +, ease +",
                "happy",
                "u_max",
                "Max",
            )
            await self.engine.touch_interaction("u_max", "Max", modality="voice")

        for k in EMOTION_KEYS:
            self.assertGreaterEqual(self.engine.emotions[k], 0.0)
            self.assertLessEqual(self.engine.emotions[k], 1.0)
        self.assertGreaterEqual(self.engine.valence, -1.0)
        self.assertLessEqual(self.engine.valence, 1.0)
        self.assertGreaterEqual(self.engine.arousal, 0.0)
        self.assertLessEqual(self.engine.arousal, 1.0)
        self.assertGreaterEqual(self.engine.energy, 0.15)
        self.assertLessEqual(self.engine.energy, 1.0)

        p = await self.engine.get_person("u_max", "Max")
        self.assertLessEqual(p["bond"], 1000.0)
        self.assertGreaterEqual(p["bond"], -100.0)
        self.assertLessEqual(p["trust"], 100.0)
        self.assertLessEqual(p["warmth"], 1.0)
        self.assertLessEqual(p["ease"], 1.0)


if __name__ == "__main__":
    unittest.main()
