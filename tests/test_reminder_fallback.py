"""Offline unit tests for the reminder fallback parser and BrainRunner safety net."""
from __future__ import annotations

import types
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from brain.llm import LLMResponse, LLMToolCall
from brain.profile import CharacterProfile
from brain.runner import BrainRunner, parse_reminder_request


def _fake_profile() -> CharacterProfile:
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
        identity_md="Kamu adalah Pachinko.",
        style_md="Santai.",
        examples_md="",
        lore_entries=[],
        kaomoji={"happy": ["(^‿^)"]},
    )


class TestReminderFallback(unittest.IsolatedAsyncioTestCase):
    def _make_runner(self, llm: Any) -> tuple[BrainRunner, list[tuple[str, dict[str, Any]]]]:
        calls: list[tuple[str, dict[str, Any]]] = []

        async def fake_execute(name: str, raw_args: dict[str, Any], **kwargs: Any) -> str:
            calls.append((name, dict(raw_args)))
            return "Reminder set"

        store = types.SimpleNamespace(get_kv=AsyncMock(return_value=""))
        bus = types.SimpleNamespace(emit=AsyncMock())
        profiles = types.SimpleNamespace(current=_fake_profile())
        heart = types.SimpleNamespace(
            touch_interaction=AsyncMock(
                return_value={"tier": "Teman", "bond": 10.0, "trust": 10.0, "warmth": 0.0, "impression": ""}
            ),
            state_summary=MagicMock(return_value={"dominant": "neutral", "valence": 0.0, "energy": 0.5, "active_emotions": {}}),
            apply_feel_block=AsyncMock(return_value="neutral"),
            dominant_emotion=MagicMock(return_value="neutral"),
        )
        memory = types.SimpleNamespace(
            auto_capture_user_facts=AsyncMock(),
            get_notes=AsyncMock(return_value=[]),
            recall=AsyncMock(return_value=[]),
            recent_transcript=AsyncMock(return_value=[]),
            add_transcript=AsyncMock(return_value=1),
        )
        tools = types.SimpleNamespace(execute=fake_execute)
        runner = BrainRunner(
            store=store, bus=bus, profiles=profiles, heart=heart, memory=memory, tools=tools, llm=llm  # type: ignore[arg-type]
        )
        return runner, calls

    def test_parse_reminder_request_cases(self) -> None:
        self.assertEqual(parse_reminder_request("ingatkan aku minum 1 menit lagi"), (1.0, "minum"))
        res_half = parse_reminder_request("ingetin gue setengah jam lagi buat makan")
        self.assertIsNotNone(res_half)
        assert res_half is not None
        self.assertEqual(res_half[0], 30.0)
        self.assertIn("makan", res_half[1].lower())

        self.assertEqual(parse_reminder_request("ingetin aku sejam lagi"), (60.0, "Pengingat!"))
        self.assertEqual(parse_reminder_request("pasang timer 30 detik"), (0.5, "Pengingat!"))
        self.assertEqual(parse_reminder_request("ingatin lima belas menit lagi"), (15.0, "Pengingat!"))
        self.assertIsNone(parse_reminder_request("ingatkan aku besok ya"))
        self.assertIsNone(parse_reminder_request("aku suka minum 1 menit lagi"))

    async def test_text_turn_reminder_fallback_when_model_skips_tool(self) -> None:
        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(return_value=LLMResponse(text="Oke siap!", tool_calls=[])),
            supports_tools=MagicMock(return_value=True),
        )
        runner, calls = self._make_runner(llm)
        await runner.run_text_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="ingatkan aku minum 1 menit lagi")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "set_reminder")
        self.assertEqual(calls[0][1]["minutes"], 1.0)
        self.assertIn("minum", calls[0][1]["note"].lower())

    async def test_text_turn_no_duplicate_when_model_calls_tool(self) -> None:
        responses = [
            LLMResponse(text="", tool_calls=[LLMToolCall("set_reminder", {"minutes": 1, "note": "minum"})]),
            LLMResponse(text="Sip, udah aku pasang pengingatnya!", tool_calls=[]),
        ]
        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(side_effect=responses),
            supports_tools=MagicMock(return_value=True),
        )
        runner, calls = self._make_runner(llm)
        await runner.run_text_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="ingatkan aku minum 1 menit lagi")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "set_reminder")

    async def test_text_turn_no_fallback_without_time_or_verb(self) -> None:
        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(return_value=LLMResponse(text="Oke!", tool_calls=[])),
            supports_tools=MagicMock(return_value=True),
        )
        runner, calls = self._make_runner(llm)
        await runner.run_text_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="ingatkan aku besok ya")
        await runner.run_text_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="aku suka minum 1 menit lagi")
        self.assertEqual(calls, [])

    async def test_voice_turn_reminder_fallback_and_no_duplicate(self) -> None:
        async def empty_stream(*args: Any, **kwargs: Any):
            if False:
                yield ""

        llm_skip = types.SimpleNamespace(
            complete_with_tools=AsyncMock(return_value=LLMResponse(text="Oke!", tool_calls=[])),
            stream_text=empty_stream,
            supports_tools=MagicMock(return_value=True),
        )
        runner_skip, calls_skip = self._make_runner(llm_skip)
        _ = [x async for x in runner_skip.stream_voice_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="ingatkan aku minum 1 menit lagi")]
        self.assertEqual(len(calls_skip), 1)
        self.assertEqual(calls_skip[0][0], "set_reminder")

        async def ok_stream(*args: Any, **kwargs: Any):
            yield "Sip!"

        llm_call = types.SimpleNamespace(
            complete_with_tools=AsyncMock(
                return_value=LLMResponse(text="", tool_calls=[LLMToolCall("set_reminder", {"minutes": 1, "note": "minum"})])
            ),
            stream_text=ok_stream,
            supports_tools=MagicMock(return_value=True),
        )
        runner_call, calls_call = self._make_runner(llm_call)
        _ = [x async for x in runner_call.stream_voice_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="ingatkan aku minum 1 menit lagi")]
        self.assertEqual(len(calls_call), 1)
        self.assertEqual(calls_call[0][0], "set_reminder")


if __name__ == "__main__":
    unittest.main()
