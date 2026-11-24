"""Offline unit tests for brain/gate.py."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brain.gate import RISK_LEVELS, ActionGate
from config import config
from core.store import Store


class FakeBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, topic: str, **payload) -> None:
        self.events.append((topic, payload))


class TestActionGate(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._orig_owner_ids = config.owner_ids
        config.owner_ids = (999,)
        self._tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self._tmp.name) / "test_gate.db")
        await self.store.init()
        self.bus = FakeBus()
        self.gate = ActionGate(self.store, self.bus)  # type: ignore[arg-type]
        self.gate.paused = False

    async def asyncTearDown(self) -> None:
        config.owner_ids = self._orig_owner_ids
        await self.store.close()
        self._tmp.cleanup()

    async def test_risk_levels_per_tool(self) -> None:
        self.assertEqual(RISK_LEVELS["web_search"], "low")
        self.assertEqual(RISK_LEVELS["remember_note"], "low")
        self.assertEqual(RISK_LEVELS["set_reminder"], "medium")
        self.assertEqual(RISK_LEVELS["queue_song"], "medium")

        ok1, _ = await self.gate.check("web_search", {}, requester_id="42", requester_name="Budi")
        ok2, _ = await self.gate.check("web_search", {}, requester_id="42", requester_name="Budi")
        self.assertTrue(ok1)
        self.assertTrue(ok2)

    async def test_unknown_tool_rejected(self) -> None:
        with self.assertLogs("brain.gate", level="WARNING"):
            ok, msg = await self.gate.check("delete_everything", {}, requester_id="42", requester_name="Budi")
        self.assertFalse(ok)
        self.assertIn("tidak dikenali", msg)

    async def test_medium_risk_cooldown_for_non_owner(self) -> None:
        clock = [1000.0]
        with patch("brain.gate.time.time", side_effect=lambda: clock[0]):
            ok1, _ = await self.gate.check("set_reminder", {}, requester_id="42", requester_name="Budi")
            ok2, msg2 = await self.gate.check("set_reminder", {}, requester_id="42", requester_name="Budi")
            clock[0] += 8.1
            ok3, _ = await self.gate.check("set_reminder", {}, requester_id="42", requester_name="Budi")

        self.assertTrue(ok1)
        self.assertFalse(ok2)
        self.assertIn("Wait a moment", msg2)
        self.assertTrue(ok3)

    async def test_owner_bypasses_cooldown(self) -> None:
        clock = [1000.0]
        with patch("brain.gate.time.time", side_effect=lambda: clock[0]):
            ok1, _ = await self.gate.check("set_reminder", {}, requester_id="999", requester_name="Owner")
            ok2, _ = await self.gate.check("set_reminder", {}, requester_id="999", requester_name="Owner")

        self.assertTrue(ok1)
        self.assertTrue(ok2)

    async def test_high_risk_requires_owner_privileges(self) -> None:
        with patch.dict(RISK_LEVELS, {"dangerous_op": "high"}):
            ok_user, msg = await self.gate.check(
                "dangerous_op", {}, requester_id="42", requester_name="Budi"
            )
            ok_owner, _ = await self.gate.check(
                "dangerous_op", {}, requester_id="999", requester_name="Owner"
            )
        self.assertFalse(ok_user)
        self.assertIn("owner privileges", msg)
        self.assertTrue(any(t == "gate.decision" and p.get("reason") == "owner_only" for t, p in self.bus.events))
        self.assertTrue(ok_owner)

    async def test_paused_gate_blocks_autonomous_and_persists(self) -> None:
        await self.gate.set_paused(True)
        self.assertTrue(any(t == "gate.pause_changed" and p["paused"] is True for t, p in self.bus.events))

        ok_auto, _ = await self.gate.check(
            "web_search", {}, requester_id="42", requester_name="Budi", is_autonomous=True
        )
        ok_direct, _ = await self.gate.check(
            "web_search", {}, requester_id="42", requester_name="Budi", is_autonomous=False
        )
        self.assertFalse(ok_auto)
        self.assertTrue(ok_direct)

        gate2 = ActionGate(self.store, self.bus)  # type: ignore[arg-type]
        await gate2.load()
        self.assertTrue(gate2.paused)


if __name__ == "__main__":
    unittest.main()
