"""Offline unit tests for brain/triage.py."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from brain.profile import CharacterProfile
from brain.triage import TriageEngine


def _profile() -> CharacterProfile:
    return CharacterProfile(
        profile_id="test",
        display_name="Pachinko",
        nicknames=["pachinko", "pachi"],
        wake_aliases={"pacinko": "Pachinko", "paciko": "Pachinko"},
        language="id",
        interests=[],
        temperament="tsundere",
        half_life_mins=25.0,
        default_channel_mode="quiet",
        filler_phrases=[],
        backchannel_words=set(),
        identity_md="",
        style_md="",
        examples_md="",
    )


class TestTriage(unittest.TestCase):
    def setUp(self) -> None:
        self.triage = TriageEngine()
        self.prof = _profile()

    def test_wake_aliases_and_is_addressed(self) -> None:
        fixed = self.triage.normalize_wake_words("hai Pacinko, apa kabar?", self.prof)
        self.assertEqual(fixed, "hai Pachinko, apa kabar?")
        self.assertTrue(self.triage.is_addressed(fixed, self.prof))
        self.assertFalse(self.triage.is_addressed("hai pacinkos apa kabar", self.prof))

        dec = self.triage.decide_text(
            room_id="r1",
            text=fixed,
            mode="mention-only",
            profile=self.prof,
            mentioned=False,
            is_dm=False,
            autonomy_paused=False,
            user_opted_out=False,
        )
        self.assertTrue(dec.should_wake)
        self.assertEqual(dec.reason, "called_name")
        self.assertTrue(dec.direct)

    def test_mention_and_dm_always_addressed(self) -> None:
        dec_mention = self.triage.decide_text(
            room_id="r1",
            text="hi",
            mode="mention-only",
            profile=self.prof,
            mentioned=True,
            is_dm=False,
            autonomy_paused=True,
            user_opted_out=True,
        )
        self.assertTrue(dec_mention.should_wake)
        self.assertEqual(dec_mention.reason, "direct_mention")
        self.assertTrue(dec_mention.direct)

        dec_dm = self.triage.decide_text(
            room_id="dm1",
            text="hi",
            mode="off",
            profile=self.prof,
            mentioned=False,
            is_dm=True,
            autonomy_paused=True,
            user_opted_out=True,
        )
        self.assertTrue(dec_dm.should_wake)
        self.assertEqual(dec_dm.reason, "direct_mention")
        self.assertTrue(dec_dm.direct)

    def test_mention_only_never_ambient(self) -> None:
        dec = self.triage.decide_text(
            room_id="r1",
            text="hari ini cuacanya lumayan enak ya",
            mode="mention-only",
            profile=self.prof,
            mentioned=False,
            is_dm=False,
            autonomy_paused=False,
            user_opted_out=False,
        )
        self.assertFalse(dec.should_wake)
        self.assertEqual(dec.reason, "ignored_mode_ceiling")

    def test_off_mode_blocks_channel_even_when_addressed_except_dm(self) -> None:
        dec = self.triage.decide_text(
            room_id="r1",
            text="Pachinko halo",
            mode="off",
            profile=self.prof,
            mentioned=True,
            is_dm=False,
            autonomy_paused=False,
            user_opted_out=False,
        )
        self.assertFalse(dec.should_wake)
        self.assertEqual(dec.reason, "ignored_channel_off")

    def test_hourly_ambient_caps_and_cooldown_per_mode(self) -> None:
        clock = [10000.0]
        with patch("brain.triage.time.time", side_effect=lambda: clock[0]):
            for mode, cap in (("quiet", 2), ("normal", 6), ("lively", 14)):
                engine = TriageEngine()
                room = f"room_{mode}"
                min_gap = 3600.0 / (cap * 2.0)

                first = engine.decide_text(
                    room_id=room,
                    text="obrolan santai di channel",
                    mode=mode,
                    profile=self.prof,
                    mentioned=False,
                    is_dm=False,
                    autonomy_paused=False,
                    user_opted_out=False,
                )
                self.assertTrue(first.should_wake, msg=mode)
                self.assertEqual(first.reason, "ambient_glance")
                self.assertFalse(first.direct)

                too_soon = engine.decide_text(
                    room_id=room,
                    text="obrolan santai lagi",
                    mode=mode,
                    profile=self.prof,
                    mentioned=False,
                    is_dm=False,
                    autonomy_paused=False,
                    user_opted_out=False,
                )
                self.assertFalse(too_soon.should_wake, msg=mode)
                self.assertEqual(too_soon.reason, "ignored_cooldown")

                for i in range(1, cap):
                    clock[0] += min_gap + 1.0
                    d = engine.decide_text(
                        room_id=room,
                        text=f"pesan ambient ke {i}",
                        mode=mode,
                        profile=self.prof,
                        mentioned=False,
                        is_dm=False,
                        autonomy_paused=False,
                        user_opted_out=False,
                    )
                    self.assertTrue(d.should_wake, msg=f"{mode} #{i}")

                clock[0] += min_gap + 1.0
                capped = engine.decide_text(
                    room_id=room,
                    text="pesan melebihi batas jam",
                    mode=mode,
                    profile=self.prof,
                    mentioned=False,
                    is_dm=False,
                    autonomy_paused=False,
                    user_opted_out=False,
                )
                self.assertFalse(capped.should_wake, msg=mode)
                self.assertEqual(capped.reason, "ignored_hourly_cap")

                # Advancing past 1 hour from the first wake expires it
                clock[0] += 3601.0
                after_hour = engine.decide_text(
                    room_id=room,
                    text="pesan setelah satu jam",
                    mode=mode,
                    profile=self.prof,
                    mentioned=False,
                    is_dm=False,
                    autonomy_paused=False,
                    user_opted_out=False,
                )
                self.assertTrue(after_hour.should_wake, msg=mode)

    def test_opted_out_and_paused_autonomy_block_ambient_not_direct(self) -> None:
        d_opt = self.triage.decide_text(
            room_id="r1",
            text="ngobrol biasa aja nih",
            mode="lively",
            profile=self.prof,
            mentioned=False,
            is_dm=False,
            autonomy_paused=False,
            user_opted_out=True,
        )
        self.assertFalse(d_opt.should_wake)
        self.assertEqual(d_opt.reason, "ignored_privacy_opt_out")

        d_paused = self.triage.decide_text(
            room_id="r1",
            text="ngobrol biasa aja nih",
            mode="lively",
            profile=self.prof,
            mentioned=False,
            is_dm=False,
            autonomy_paused=True,
            user_opted_out=False,
        )
        self.assertFalse(d_paused.should_wake)
        self.assertEqual(d_paused.reason, "ignored_autonomy_paused")

        d_direct = self.triage.decide_text(
            room_id="r1",
            text="Pachi tolong dong",
            mode="lively",
            profile=self.prof,
            mentioned=False,
            is_dm=False,
            autonomy_paused=True,
            user_opted_out=True,
        )
        self.assertTrue(d_direct.should_wake)
        self.assertEqual(d_direct.reason, "called_name")


if __name__ == "__main__":
    unittest.main()
