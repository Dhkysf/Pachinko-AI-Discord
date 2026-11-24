"""Tests for strict chat-template message assembly and normalization."""
from __future__ import annotations

import copy
import unittest

from brain.profile import CharacterProfile
from brain.prompt import _VOICE_EXTRA_RULES, assemble_messages


def _make_profile() -> CharacterProfile:
    return CharacterProfile(
        profile_id="test",
        display_name="Aira",
        nicknames=["aira"],
        wake_aliases={},
        language="id",
        interests=["musik"],
        temperament="tsundere",
        half_life_mins=25.0,
        default_channel_mode="text",
        filler_phrases=["Hmm?"],
        backchannel_words={"hmm", "iya"},
        voice_cfg={},
        identity_md="Kamu adalah Aira, AI yang ekspresif.",
        style_md="Gaya bicara santai dan hangat.",
        examples_md="User: Halo\nAira: Hai juga!",
        lore_entries=[
            {"keywords": ["kopi"], "content": "Aira suka kopi susu.", "priority": 2, "scope": "both"}
        ],
        kaomoji={"happy": ["(≧▽≦)", "(^‿^)"]},
    )


def _make_args(modality: str = "text", user_text: str = "Halo Aira, lagi minum kopi ya?") -> dict:
    return {
        "profile": _make_profile(),
        "directives": "Selalu jawab singkat.",
        "heart_state": {"dominant": "happy", "valence": 0.4, "energy": 0.7, "active_emotions": {"joy": 0.6}},
        "person": {"tier": "Teman", "bond": 120.0, "trust": 60.0, "warmth": 0.25, "impression": "Ramah"},
        "person_notes": [{"fact": "Suka main game"}],
        "recalled_memories": [{"content": "Kemarin bahas musik"}],
        "history": [
            {"role": "user", "speaker_name": "Budi", "content": "Pagi Aira!"},
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
        msgs = assemble_messages(**_make_args(modality="text"))
        self._assert_strict_roles(msgs)

        self.assertIn("Kamu adalah Aira", msgs[0]["content"])
        self.assertIn("Gaya bicara santai", msgs[0]["content"])
        self.assertIn("User: Halo", msgs[0]["content"])
        self.assertIn("Selalu jawab singkat.", msgs[0]["content"])

        all_text = "\n".join(m["content"] for m in msgs)
        self.assertIn("Suka main game", all_text)
        self.assertIn("Kemarin bahas musik", all_text)
        self.assertIn("Pagi Aira!", all_text)
        self.assertIn("Pagi juga Budi!", all_text)

        final_user = msgs[-1]
        self.assertEqual(final_user["role"], "user")
        content = final_user["content"]
        for tag in ("<now>", "<style_note>", "<turn "):
            self.assertIn(tag, content)
        self.assertIn("Aira suka kopi susu.", content)
        self.assertIn("Halo Aira, lagi minum kopi ya?", content)
        self.assertLess(content.index("<turn "), content.index("Halo Aira, lagi minum kopi ya?"))

    def test_voice_turn_strict_roles_and_content(self) -> None:
        msgs = assemble_messages(**_make_args(modality="voice"))
        self._assert_strict_roles(msgs)

        final_user = msgs[-1]
        self.assertEqual(final_user["role"], "user")
        content = final_user["content"]
        self.assertIn(_VOICE_EXTRA_RULES, content)
        self.assertIn("Halo Aira, lagi minum kopi ya?", content)
        self.assertLess(content.index("<style_note>"), content.index("Halo Aira, lagi minum kopi ya?"))

    def test_system_message_byte_identical_between_turns(self) -> None:
        args1 = _make_args(modality="text", user_text="Halo")
        args2 = _make_args(modality="voice", user_text="Apa kabar?")
        args2["heart_state"] = {"dominant": "sad", "valence": -0.5, "energy": 0.2, "active_emotions": {}}
        args2["person"] = {"tier": "Sahabat", "bond": 900.0, "trust": 95.0, "warmth": 0.9, "impression": "Akrab"}
        args2["history"] = []

        msgs1 = assemble_messages(**args1)
        msgs2 = assemble_messages(**args2)
        self.assertEqual(msgs1[0], msgs2[0])


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
            {"role": "user", "content": "[Hasil Tool `t1`]:\nout1"},
            {"role": "user", "content": "[Hasil Tool `t2`]:\nout2"},
        ]
        norm = normalize_messages(msgs)
        self.assertEqual(
            norm,
            [
                {"role": "system", "content": "sys0"},
                {"role": "assistant", "content": "a1\n\na2"},
                {"role": "user", "content": "[Hasil Tool `t1`]:\nout1\n\n[Hasil Tool `t2`]:\nout2"},
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
