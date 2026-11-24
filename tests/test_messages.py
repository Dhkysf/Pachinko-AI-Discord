"""Tests for strict chat-template message assembly and normalization."""
from __future__ import annotations

import copy
import datetime as dt
import os
import unittest
from unittest.mock import patch

from brain.profile import CharacterProfile
from brain.prompt import (
    _COMPACT_LANGUAGE_RULE,
    _LANGUAGE_RULE,
    _RUNNER_RULES,
    _VOICE_EXTRA_RULES,
    assemble_messages,
)
from config import AppConfig


def _make_profile() -> CharacterProfile:
    return CharacterProfile(
        profile_id="test",
        display_name="Pachinko",
        nicknames=["pachinko"],
        wake_aliases={},
        language="id",
        interests=["musik"],
        temperament="tsundere",
        half_life_mins=25.0,
        default_channel_mode="text",
        filler_phrases=["Hmm?"],
        backchannel_words={"hmm", "iya"},
        identity_md="Kamu adalah Pachinko, AI yang ekspresif.",
        style_md="Gaya bicara santai dan hangat.",
        examples_md="User: Halo\nPachinko: Hai juga!",
        lore_entries=[
            {"keywords": ["kopi"], "content": "Pachinko suka kopi susu.", "priority": 2, "scope": "both"}
        ],
        kaomoji={"happy": ["(≧▽≦)", "(^‿^)"]},
    )


def _make_args(modality: str = "text", user_text: str = "Halo Pachinko, lagi minum kopi ya?") -> dict:
    return {
        "profile": _make_profile(),
        "directives": "Selalu jawab singkat.",
        "heart_state": {"dominant": "happy", "valence": 0.4, "energy": 0.7, "active_emotions": {"joy": 0.6}},
        "person": {"tier": "Teman", "bond": 120.0, "trust": 60.0, "warmth": 0.25, "impression": "Ramah"},
        "person_notes": [{"fact": "Suka main game"}],
        "recalled_memories": [{"content": "Kemarin bahas musik"}],
        "history": [
            {"role": "user", "speaker_name": "Budi", "content": "Pagi Pachinko!"},
            {"role": "assistant", "content": "Pagi juga Budi!", "interrupted": False},
        ],
        "speaker_name": "Budi",
        "user_text": user_text,
        "modality": modality,
        "wake_reason": "direct_mention" if modality == "text" else "voice_utterance",
    }


class TestAssembleMessages(unittest.TestCase):
    def _assert_strict_roles(self, messages: list[dict]) -> None:
        sys_indices = [i for i, m in enumerate(messages) if m["role"] == "system"]
        self.assertEqual(sys_indices, [0], f"Expected single system message at index 0, got {sys_indices}")
        for i in range(len(messages) - 1):
            self.assertNotEqual(
                messages[i]["role"],
                messages[i + 1]["role"],
                f"Adjacent messages share role '{messages[i]['role']}' at indices {i} and {i + 1}",
            )

    def test_text_turn_strict_roles_and_content(self) -> None:
        with patch("brain.prompt.config.llm_compact_prompt", False):
            msgs = assemble_messages(**_make_args(modality="text"))
        self._assert_strict_roles(msgs)

        self.assertTrue(msgs[0]["content"].startswith(_LANGUAGE_RULE))
        self.assertIn("Kamu adalah Pachinko", msgs[0]["content"])
        self.assertIn("Gaya bicara santai", msgs[0]["content"])
        self.assertIn("User: Halo", msgs[0]["content"])
        self.assertIn("Selalu jawab singkat.", msgs[0]["content"])

        all_text = "\n".join(m["content"] for m in msgs)
        self.assertIn("Suka main game", all_text)
        self.assertIn("Kemarin bahas musik", all_text)
        self.assertIn("Pagi Pachinko!", all_text)
        self.assertIn("Pagi juga Budi!", all_text)

        final_user = msgs[-1]
        self.assertEqual(final_user["role"], "user")
        content = final_user["content"]
        for tag in ("<now>", "<style_note>", "<turn "):
            self.assertIn(tag, content)
        self.assertIn("Pachinko suka kopi susu.", content)
        self.assertIn("Halo Pachinko, lagi minum kopi ya?", content)
        self.assertLess(content.index("<turn "), content.index("Halo Pachinko, lagi minum kopi ya?"))

    def test_voice_turn_strict_roles_and_content(self) -> None:
        with patch("brain.prompt.config.llm_compact_prompt", True):
            msgs = assemble_messages(**_make_args(modality="voice"))
        self._assert_strict_roles(msgs)

        final_user = msgs[-1]
        self.assertEqual(final_user["role"], "user")
        content = final_user["content"]
        self.assertIn(_VOICE_EXTRA_RULES, content)
        self.assertIn("Halo Pachinko, lagi minum kopi ya?", content)
        self.assertLess(content.index("<style_note>"), content.index("Halo Pachinko, lagi minum kopi ya?"))
        self.assertIn("1-2 kalimat, nyambung sama pesan terakhir. Kalau nggak ngerti slang-nya, tanya balik.", content)
        self.assertTrue(content.endswith("Kalau user minta pengingat, cari info, atau putar lagu, panggil tool-nya dulu."))

    def test_compact_prompt_default_and_structure(self) -> None:
        env = dict(os.environ)
        env.pop("LLM_COMPACT_PROMPT", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(AppConfig().llm_compact_prompt)

        with patch("brain.prompt.config.llm_compact_prompt", True):
            msgs = assemble_messages(**_make_args(modality="text"))
        self._assert_strict_roles(msgs)
        self.assertTrue(msgs[0]["content"].startswith(_COMPACT_LANGUAGE_RULE))
        all_text = "\n".join(m["content"] for m in msgs)
        self.assertNotIn("<feel>", all_text)
        self.assertNotIn("Bond:", all_text)
        self.assertNotIn("Trust:", all_text)
        self.assertNotIn("Warmth:", all_text)
        self.assertNotIn("valence=", all_text)
        self.assertNotIn("energy=", all_text)
        self.assertIn("hubungan: teman ngobrol", all_text)
        self.assertIn("mood: lagi seneng", all_text)
        self.assertIn(
            "Jawab sebagai Pachinko pakai bahasa Indonesia santai, 1-3 kalimat, nyambung sama pesan terakhir. Kalau nggak ngerti slang-nya, tanya balik.",
            msgs[-1]["content"],
        )
        self.assertTrue(
            msgs[-1]["content"].endswith("Kalau user minta pengingat, cari info, atau putar lagu, panggil tool-nya dulu.")
        )

    def test_legacy_prompt_byte_for_byte_when_compact_false(self) -> None:
        args = _make_args(modality="text")
        args["profile"].kaomoji = {"happy": ["(≧▽≦)"]}
        fixed_now = dt.datetime(2026, 1, 15, 20, 30)

        class _FixedDateTime:
            @classmethod
            def now(cls) -> dt.datetime:
                return fixed_now

        expected = [
            {
                "role": "system",
                "content": (
                    f"{_LANGUAGE_RULE}\n\n"
                    "Kamu adalah Pachinko, AI yang ekspresif.\n\n"
                    "Gaya bicara santai dan hangat.\n\n"
                    f"{_RUNNER_RULES}\n\n"
                    "# Server Owner Directives\nSelalu jawab singkat.\n\n"
                    "<examples>\nUser: Halo\nPachinko: Hai juga!\n</examples>"
                ),
            },
            {
                "role": "user",
                "content": (
                    '<data source="person_card" title="Budi">\n'
                    "Name: Budi\n"
                    "Relationship Status: Teman (Bond: 120.0/1000, Trust: 60/100, Warmth: +0.25)\n"
                    "Impression: Ramah\n"
                    "Notes about Budi:\n"
                    "- Suka main game\n"
                    "</data>\n\n"
                    '<data source="recalled_memory" title="Recalled Memories">\n'
                    "- Kemarin bahas musik\n"
                    "</data>"
                ),
            },
            {"role": "assistant", "content": "Oke, aku inget konteks lawan bicara dan catatan di atas."},
            {"role": "user", "content": "[Budi]: Pagi Pachinko!"},
            {"role": "assistant", "content": "Pagi juga Budi!"},
            {
                "role": "user",
                "content": (
                    "<now>\n"
                    "Time: Thursday, 15 January 2026 20:30 | Channel: TEXT\n"
                    "Pachinko's Mood: expression=happy, valence=+0.40, energy=0.70 (joy:0.60)\n"
                    "</now>\n\n"
                    "<style_note>\n"
                    "Relevant info:\n"
                    "- Pachinko suka kopi susu.\n"
                    "Optional kaomoji (use sparingly, not in every sentence): (≧▽≦)\n"
                    "</style_note>\n\n"
                    '<turn reason="direct_mention">Reply directly as Pachinko without writing a \'[Pachinko]:\' prefix.</turn>\n\n'
                    "[Budi]:\n"
                    '<data source="chat Budi" title="text">\n'
                    "Halo Pachinko, lagi minum kopi ya?\n"
                    "</data>"
                ),
            },
        ]

        with patch("brain.prompt.config.llm_compact_prompt", False), patch("brain.prompt.dt.datetime", _FixedDateTime):
            actual = assemble_messages(**args)
        self.assertEqual(actual, expected)

    def test_system_message_byte_identical_between_turns(self) -> None:
        args1 = _make_args(modality="text", user_text="Halo")
        args2 = _make_args(modality="voice", user_text="Apa kabar?")
        args2["heart_state"] = {"dominant": "sad", "valence": -0.5, "energy": 0.2, "active_emotions": {}}
        args2["person"] = {"tier": "Sahabat", "bond": 900.0, "trust": 95.0, "warmth": 0.9, "impression": "Akrab"}
        args2["history"] = []

        msgs1 = assemble_messages(**args1)
        msgs2 = assemble_messages(**args2)
        self.assertEqual(msgs1[0], msgs2[0])

    def test_assistant_messages_are_indonesian_not_english(self) -> None:
        args = _make_args(modality="text")
        args["history"] = [
            {"role": "user", "speaker_name": "Budi", "content": "Pagi!"},
            {"role": "assistant", "content": "Sebentar ya...", "interrupted": True},
        ]
        msgs = assemble_messages(**args)
        assistant_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
        self.assertIn("Oke, aku inget konteks lawan bicara dan catatan di atas.", assistant_msgs)
        self.assertTrue(
            any(c.endswith(" [terpotong karena lawan bicara menyela]") for c in assistant_msgs),
            f"Expected Indonesian interruption suffix in {assistant_msgs}",
        )
        for content in assistant_msgs:
            self.assertNotIn("I remember the speaker context and notes above.", content)
            self.assertNotIn(" [interrupted because the speaker cut in]", content)


class TestNormalizeMessages(unittest.TestCase):
    def test_folds_later_system_into_next_user(self) -> None:
        from brain.llm import normalize_messages

        msgs = [
            {"role": "system", "content": "sys0"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "sys_tail"},
            {"role": "user", "content": "u2"},
        ]
        norm = normalize_messages(msgs)
        self.assertEqual(
            norm,
            [
                {"role": "system", "content": "sys0"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "sys_tail\n\nu2"},
            ],
        )

    def test_merges_adjacent_same_role(self) -> None:
        from brain.llm import normalize_messages

        msgs = [
            {"role": "system", "content": "sys0"},
            {"role": "assistant", "content": "a1"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "[Tool Result `t1`]:\nout1"},
            {"role": "user", "content": "[Tool Result `t2`]:\nout2"},
        ]
        norm = normalize_messages(msgs)
        self.assertEqual(
            norm,
            [
                {"role": "system", "content": "sys0"},
                {"role": "assistant", "content": "a1\n\na2"},
                {"role": "user", "content": "[Tool Result `t1`]:\nout1\n\n[Tool Result `t2`]:\nout2"},
            ],
        )

    def test_trailing_system_never_dropped(self) -> None:
        from brain.llm import normalize_messages

        msgs = [
            {"role": "system", "content": "sys0"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "system", "content": "sys_tail"},
        ]
        norm = normalize_messages(msgs)
        self.assertEqual(
            norm,
            [
                {"role": "system", "content": "sys0"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "sys_tail"},
            ],
        )

    def test_does_not_mutate_input(self) -> None:
        from brain.llm import normalize_messages

        msgs = [
            {"role": "system", "content": "sys0"},
            {"role": "user", "content": "u1"},
            {"role": "system", "content": "sys_tail"},
            {"role": "user", "content": "u2"},
        ]
        snapshot = copy.deepcopy(msgs)
        _ = normalize_messages(msgs)
        self.assertEqual(msgs, snapshot)

    def test_empty_and_single_system(self) -> None:
        from brain.llm import normalize_messages

        self.assertEqual(normalize_messages([]), [])
        self.assertEqual(
            normalize_messages([{"role": "system", "content": "sys0"}, {"role": "user", "content": ""}]),
            [{"role": "system", "content": "sys0"}, {"role": "user", "content": ""}],
        )


if __name__ == "__main__":
    unittest.main()
