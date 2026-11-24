"""Unit tests for _filter_stream, _ollama_stream, _openai_stream, and stream_voice_turn."""
from __future__ import annotations

import json
import types
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx

from brain.llm import LLMClient, LLMResponse, _filter_stream
from brain.profile import CharacterProfile
from brain.runner import BrainRunner


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


class _FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self) -> _FakeStreamResponse:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("POST", "http://localhost")
            raise httpx.HTTPStatusError("err", request=req, response=httpx.Response(self.status_code, request=req))

    async def aiter_lines(self):
        for line in self._lines:
            yield line


async def _aiter(items: list[str]):
    for item in items:
        yield item


class TestLLMStreamAndVoiceTurn(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = LLMClient()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_filter_stream_think_single_and_split_chunks(self) -> None:
        out_single = "".join([t async for t in _filter_stream(_aiter(["<think>ok</think>Halo", " dunia!"]))])
        self.assertEqual(out_single, "Halo dunia!")

        out_split = "".join([t async for t in _filter_stream(_aiter(["<th", "ink>ok</th", "ink>Halo dunia!"]))])
        self.assertEqual(out_split, "Halo dunia!")

        out_mid = "".join([t async for t in _filter_stream(_aiter(["Hai <thi", "nk>rahasia panjang</thi", "nk> dunia!"]))])
        self.assertEqual(out_mid, "Hai  dunia!")

    async def test_ollama_stream_think_normal_and_400_retry(self) -> None:
        for chunks in (
            ["<think>ok</think>Halo", " dunia!"],
            ["<th", "ink>ok</th", "ink>Halo dunia!"],
        ):
            lines = [json.dumps({"message": {"content": c}}) for c in chunks]
            self.client._http.stream = MagicMock(return_value=_FakeStreamResponse(lines, status_code=200))  # type: ignore[method-assign]
            out = "".join([t async for t in self.client._ollama_stream([{"role": "user", "content": "hi"}], True)])
            self.assertEqual(out, "Halo dunia!")

            seen_payloads: list[dict[str, Any]] = []

            def fake_stream(method: str, url: str, json: dict[str, Any], **kwargs: Any) -> _FakeStreamResponse:
                seen_payloads.append(dict(json))
                if len(seen_payloads) == 1:
                    return _FakeStreamResponse([], status_code=400)
                return _FakeStreamResponse(lines, status_code=200)

            self.client._http.stream = fake_stream  # type: ignore[method-assign]
            out_retry = "".join([t async for t in self.client._ollama_stream([{"role": "user", "content": "hi"}], True)])
            self.assertEqual(out_retry, "Halo dunia!")
            self.assertEqual(len(seen_payloads), 2)
            self.assertIn("think", seen_payloads[0])
            self.assertNotIn("think", seen_payloads[1])

    async def test_openai_stream_skips_empty_choices_null_delta_and_garbage_json(self) -> None:
        lines = [
            "",
            ": keep-alive",
            'data: {"choices": []}',
            'data: {"usage": {"total_tokens": 5}}',
            'data: {"choices": [{"delta": null}]}',
            'data: {"choices": [{"delta": {"content": null}}]}',
            "data: not-valid-json",
            "data: still-not-json",
            f'data: {json.dumps({"choices": [{"delta": {"content": "Halo "}}]})}',
            f'data: {json.dumps({"choices": [{"delta": {"content": "dunia!"}}]})}',
            "data: [DONE]",
        ]
        self.client._http.stream = MagicMock(return_value=_FakeStreamResponse(lines))  # type: ignore[method-assign]
        with self.assertLogs("llm", level="DEBUG") as cm:
            out = "".join([t async for t in self.client._openai_stream([{"role": "user", "content": "hi"}], True)])
        self.assertEqual(out, "Halo dunia!")
        self.assertEqual(len(cm.output), 1)

    def _make_runner(self, llm: Any) -> BrainRunner:
        store = types.SimpleNamespace(get_kv=AsyncMock(return_value=""))
        bus = types.SimpleNamespace(emit=AsyncMock())
        profiles = types.SimpleNamespace(current=_fake_profile())
        heart = types.SimpleNamespace(
            touch_interaction=AsyncMock(return_value={"tier": "Teman", "bond": 10.0, "trust": 10.0, "warmth": 0.0, "impression": ""}),
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

    async def test_stream_voice_turn_precheck_text_without_tool_calls_makes_one_llm_call(self) -> None:
        stream_calls = 0

        async def fail_if_called(*args: Any, **kwargs: Any):
            nonlocal stream_calls
            stream_calls += 1
            yield "should not be called"

        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(
                return_value=LLMResponse(
                    text="<emo:happy>Oke siap! Nanti aku ingetin ya. <feel>self: joy weak</feel>",
                    tool_calls=[],
                )
            ),
            stream_text=fail_if_called,
            supports_tools=MagicMock(return_value=True),
        )
        runner = self._make_runner(llm)
        items = [
            x
            async for x in runner.stream_voice_turn(
                room_id="r1", user_id="u1", user_name="Budi", user_text="ingetin aku minum"
            )
        ]
        self.assertEqual(llm.complete_with_tools.await_count, 1)
        self.assertEqual(stream_calls, 0)
        self.assertEqual(
            items,
            [("Oke siap!", "neutral", []), ("Nanti aku ingetin ya.", "neutral", [])],
        )
        runner.heart.apply_feel_block.assert_awaited_once_with(
            "self: joy weak", "happy", "u1", "Budi", "tsundere"
        )

    async def test_stream_voice_turn_precheck_silence_without_tool_calls_makes_one_llm_call(self) -> None:
        stream_calls = 0

        async def fail_if_called(*args: Any, **kwargs: Any):
            nonlocal stream_calls
            stream_calls += 1
            yield "should not be called"

        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(return_value=LLMResponse(text="<silence/>", tool_calls=[])),
            stream_text=fail_if_called,
            supports_tools=MagicMock(return_value=True),
        )
        runner = self._make_runner(llm)
        items = [
            x
            async for x in runner.stream_voice_turn(
                room_id="r1", user_id="u1", user_name="Budi", user_text="ingetin aku minum"
            )
        ]
        self.assertEqual(llm.complete_with_tools.await_count, 1)
        self.assertEqual(stream_calls, 0)
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
