"""Offline tests for explicit language requests, register guard, and silent direct reply handling."""
from __future__ import annotations

import json
import os
import types
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from brain.llm import LLMClient, LLMResponse
from brain.profile import CharacterProfile
from brain.prompt import (
    _COMPACT_LANGUAGE_RULE,
    _LANGUAGE_RULE,
    assemble_messages,
    normalize_register,
)
from brain.runner import _LLM_ERROR_REPLY, BrainRunner
from config import AppConfig


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


def _prompt_args(user_text: str, modality: str = "text") -> dict[str, Any]:
    return {
        "profile": _fake_profile(),
        "directives": "",
        "heart_state": {"dominant": "neutral", "valence": 0.0, "energy": 0.7, "active_emotions": {}},
        "person": {"tier": "Teman", "bond": 100.0, "trust": 50.0, "warmth": 0.2, "impression": "Ramah"},
        "person_notes": [],
        "recalled_memories": [],
        "history": [],
        "speaker_name": "Budi",
        "user_text": user_text,
        "modality": modality,
        "wake_reason": "direct_mention" if modality == "text" else "voice_utterance",
    }


def _make_runner(llm: Any) -> BrainRunner:
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
    tools = types.SimpleNamespace(execute=AsyncMock(return_value="ok"))
    return BrainRunner(
        store=store, bus=bus, profiles=profiles, heart=heart, memory=memory, tools=tools, llm=llm  # type: ignore[arg-type]
    )


def _make_cfg(**overrides: Any) -> AppConfig:
    with patch.dict(os.environ, {}, clear=True):
        cfg = AppConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestRequestedLanguageAndPrompt(unittest.TestCase):
    def test_requested_language_positive_and_negative(self) -> None:
        from brain.prompt import requested_language

        positives = (
            ("jelasin RAM pakai bahasa Jepang", "Jepang"),
            ("Jelaskan apa itu RAM pakai bahasa Jepang", "Jepang"),
            ("pake bahasa inggris", "Inggris"),
            ("dalam bahasa Korea", "Korea"),
            ("explain RAM in bahasa Jepang", "Jepang"),
            ("explain RAM in English", "English"),
            ("answer in Japanese", "Japanese"),
            ("translate ke Jepang", "Jepang"),
            ("translate ke bahasa Spanyol", "Spanyol"),
            ("terjemahin ke Inggris", "Inggris"),
            ("jawab pakai Jepang", "Jepang"),
            ("jawab pake Inggris", "Inggris"),
        )
        for text, expected in positives:
            self.assertEqual(requested_language(text), expected, msg=f"Failed for {text!r}")

        negatives = (
            "",
            "Halo Pachinko",
            "jelasin pakai bahasa Indonesia",
            "jawab pakai bahasa santai",
            "in general this is fine",
            "aku suka bahasa Jepang",
            "jawab pakai poin",
        )
        for text in negatives:
            self.assertIsNone(requested_language(text), msg=f"Unexpected match for {text!r}")

    def test_assemble_messages_overrides_language_rule_and_reminder(self) -> None:
        expected_line = "User minta jawaban dalam bahasa Jepang. Jawab dalam bahasa itu, tetap dengan gaya Pachinko."
        for compact in (True, False):
            for modality in ("text", "voice"):
                with patch("brain.prompt.config.llm_compact_prompt", compact):
                    msgs = assemble_messages(**_prompt_args("jelasin RAM pakai bahasa Jepang", modality=modality))
                self.assertTrue(msgs[0]["content"].startswith(expected_line))
                self.assertTrue(msgs[-1]["content"].endswith(expected_line))
                joined = "\n".join(m["content"] for m in msgs)
                self.assertNotIn(_LANGUAGE_RULE, joined)
                self.assertNotIn(_COMPACT_LANGUAGE_RULE, joined)
                self.assertNotIn("bahasa Indonesia", joined)
                self.assertNotIn("always indonesian", joined.lower())
                self.assertNotIn("Pakai aku/kamu", joined)

    def test_assemble_messages_normal_indonesian_unchanged(self) -> None:
        with patch("brain.prompt.config.llm_compact_prompt", True):
            msgs = assemble_messages(**_prompt_args("Halo Pachinko, lagi minum kopi ya?", modality="text"))
        self.assertTrue(msgs[0]["content"].startswith(_COMPACT_LANGUAGE_RULE))
        self.assertIn("Pakai aku/kamu, jangan gue/lo/saya.", msgs[-1]["content"])

    def test_normalize_register_skips_non_indonesian_and_keeps_indonesian(self) -> None:
        self.assertEqual(normalize_register("Lo siento, no puedo ayudarte.", "aku"), "Lo siento, no puedo ayudarte.")
        self.assertEqual(normalize_register("Lo siento", "aku"), "Lo siento")
        self.assertEqual(normalize_register("I love lo-fi beats while coding", "aku"), "I love lo-fi beats while coding")
        self.assertEqual(normalize_register("Gue lagi capek banget nih", "aku"), "Aku lagi capek banget nih")


class TestRunnerLanguageAndSilentReplies(unittest.IsolatedAsyncioTestCase):
    async def test_run_text_turn_skips_register_normalization_on_language_request(self) -> None:
        seen_msgs: list[list[dict[str, Any]]] = []

        async def fake_complete(messages: list[dict[str, Any]], tools: Any = None, fast_voice: bool = False, **kw: Any):
            seen_msgs.append(messages)
            return LLMResponse(text="RAMは一時的なメモリだよ、lo siento!", tool_calls=[])

        llm = types.SimpleNamespace(
            complete_with_tools=fake_complete,
            supports_tools=MagicMock(return_value=True),
        )
        runner = _make_runner(llm)
        res = await runner.run_text_turn(
            room_id="r1",
            user_id="u1",
            user_name="Budi",
            user_text="jelasin RAM pakai bahasa Jepang",
            wake_reason="direct_mention",
        )
        self.assertFalse(res.silent)
        self.assertEqual(res.text, "RAMは一時的なメモリだよ、lo siento!")
        self.assertIn("bahasa Jepang", seen_msgs[0][0]["content"])

    async def test_stream_voice_turn_skips_register_normalization_on_language_request(self) -> None:
        seen_msgs: list[list[dict[str, Any]]] = []

        async def fake_stream(messages: list[dict[str, Any]], fast_voice: bool = True):
            seen_msgs.append(messages)
            yield "RAM wa memori desu, lo siento."

        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(),
            stream_text=fake_stream,
            supports_tools=MagicMock(return_value=True),
        )
        runner = _make_runner(llm)
        items = [
            x
            async for x in runner.stream_voice_turn(
                room_id="r1",
                user_id="u1",
                user_name="Budi",
                user_text="jelasin RAM pakai bahasa Jepang",
            )
        ]
        self.assertEqual(items, [("RAM wa memori desu, lo siento.", "neutral", [])])
        self.assertIn("bahasa Jepang", seen_msgs[0][0]["content"])

    async def test_llm_reasoning_never_returned_and_max_tokens_scale(self) -> None:
        seen_payloads: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            seen_payloads.append(body)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": "",
                                "reasoning_content": "secret chain of thought",
                            },
                        }
                    ]
                },
            )

        client = LLMClient()
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cfg = _make_cfg(
                llm_provider="ninerouter",
                ninerouter_api_key="k",
                ninerouter_model="deepseek-r1",
                llm_compact_prompt=True,
                gemini_api_key="",
            )
            with patch("brain.llm.config", cfg):
                r1 = await client.complete_with_tools([{"role": "user", "content": "hi"}], max_tokens_scale=1)
                r2 = await client.complete_with_tools([{"role": "user", "content": "hi"}], max_tokens_scale=2)
            self.assertEqual(r1.text, "")
            self.assertEqual(r1.finish_reason, "length")
            self.assertTrue(r1.has_reasoning)
            self.assertEqual(r1.provider, "ninerouter")
            self.assertEqual(r1.model, "deepseek-r1")
            self.assertEqual(seen_payloads[0]["max_tokens"], 300)
            self.assertEqual(seen_payloads[1]["max_tokens"], 600)
        finally:
            await client.close()

    async def test_ollama_unclosed_think_and_thinking_field_stripped(self) -> None:
        req = httpx.Request("POST", "http://localhost:11434/api/chat")
        resp = httpx.Response(
            200,
            json={
                "done_reason": "length",
                "message": {"content": "<think>cutoff thinking", "thinking": "more"},
            },
            request=req,
        )
        client = LLMClient()
        client._http.post = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        try:
            out = await client._ollama_once([{"role": "user", "content": "hi"}], tools=None, fast_voice=False)
            self.assertEqual(out.text, "")
            self.assertEqual(out.finish_reason, "length")
            self.assertTrue(out.has_reasoning)
            self.assertEqual(out.provider, "ollama")
        finally:
            await client.close()

    async def test_direct_empty_length_reasoning_warns_retries_once_and_falls_back(self) -> None:
        seen_payloads: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "", "reasoning_content": "private reasoning"},
                        }
                    ]
                },
            )

        client = LLMClient()
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cfg = _make_cfg(
                llm_provider="ninerouter",
                ninerouter_api_key="k",
                ninerouter_model="deepseek-r1",
                llm_compact_prompt=True,
                gemini_api_key="",
            )
            runner = _make_runner(client)
            with patch("brain.llm.config", cfg), self.assertLogs("brain.runner", level="WARNING") as cm:
                res = await runner.run_text_turn(
                    room_id="r1",
                    user_id="u1",
                    user_name="Budi",
                    user_text="SECRET_USER_PROMPT_XYZ",
                    wake_reason="direct_mention",
                )
            self.assertEqual(len(seen_payloads), 2)
            self.assertEqual(seen_payloads[0]["max_tokens"], 300)
            self.assertEqual(seen_payloads[1]["max_tokens"], 600)
            self.assertEqual(len(cm.output), 1)
            warn_msg = cm.output[0]
            self.assertIn("ninerouter", warn_msg)
            self.assertIn("deepseek-r1", warn_msg)
            self.assertIn("length", warn_msg)
            self.assertIn("has_reasoning=True", warn_msg)
            self.assertNotIn("SECRET_USER_PROMPT_XYZ", warn_msg)
            self.assertFalse(res.silent)
            self.assertEqual(res.text, _LLM_ERROR_REPLY)
            self.assertNotIn("private reasoning", res.text)
        finally:
            await client.close()

    async def test_direct_silence_retries_and_succeeds(self) -> None:
        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(
                side_effect=[
                    LLMResponse(text="<silence/>", finish_reason="stop", has_reasoning=False, provider="ollama", model="m1"),
                    LLMResponse(text="Hai Budi!", finish_reason="stop", has_reasoning=False, provider="ollama", model="m1"),
                ]
            ),
            supports_tools=MagicMock(return_value=True),
        )
        runner = _make_runner(llm)
        with self.assertLogs("brain.runner", level="WARNING") as cm:
            res = await runner.run_text_turn(
                room_id="r1",
                user_id="u1",
                user_name="Budi",
                user_text="Halo",
                wake_reason="slash_say",
            )
        self.assertEqual(len(cm.output), 1)
        self.assertEqual(llm.complete_with_tools.await_count, 2)
        self.assertFalse(res.silent)
        self.assertEqual(res.text, "Hai Budi!")

    async def test_ambient_empty_stays_silent_without_warning_or_retry(self) -> None:
        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(return_value=LLMResponse(text="<silence/>", finish_reason="stop")),
            supports_tools=MagicMock(return_value=True),
        )
        runner = _make_runner(llm)
        with self.assertNoLogs("brain.runner", level="WARNING"):
            res = await runner.run_text_turn(
                room_id="r1",
                user_id="u1",
                user_name="Budi",
                user_text="random chat",
                wake_reason="ambient_glance",
            )
        self.assertEqual(llm.complete_with_tools.await_count, 1)
        self.assertTrue(res.silent)
        self.assertEqual(res.text, "")


if __name__ == "__main__":
    unittest.main()
