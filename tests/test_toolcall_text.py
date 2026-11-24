"""Tests for leaked tool-call text parsing, stream stripping, no-tools retry, and LLM error handling."""
from __future__ import annotations

import json
import types
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from brain.llm import LLMClient, LLMToolCall
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


def _make_httpx_response(status_code: int, payload: dict[str, Any] | str) -> httpx.Response:
    req = httpx.Request("POST", "http://localhost:11434/api/chat")
    if isinstance(payload, str):
        return httpx.Response(status_code, text=payload, request=req)
    return httpx.Response(status_code, json=payload, request=req)


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


class TestToolCallTextOnce(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = LLMClient()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def _call_ollama(self, content: str):
        resp = _make_httpx_response(200, {"message": {"content": content}})
        self.client._http.post = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        return await self.client._ollama_once([{"role": "user", "content": "hi"}], tools=None, fast_voice=False)

    async def _call_openai(self, content: str):
        resp = _make_httpx_response(200, {"choices": [{"message": {"content": content}}]})
        self.client._http.post = AsyncMock(return_value=resp)  # type: ignore[method-assign]
        return await self.client._openai_once([{"role": "user", "content": "hi"}], tools=None, fast_voice=False)

    async def test_tagged_tool_call_ollama_and_openai(self) -> None:
        raw = 'Bentar ya <tool_call>{"name": "remember_note", "arguments": {"fact": "suka kopi"}}</tool_call> oke!'
        for fn in (self._call_ollama, self._call_openai):
            res = await fn(raw)
            self.assertEqual(res.tool_calls, [LLMToolCall("remember_note", {"fact": "suka kopi"})])
            self.assertNotIn("tool_call", res.text)
            self.assertNotIn("remember_note", res.text)
            self.assertIn("Bentar ya", res.text)
            self.assertIn("oke!", res.text)

    async def test_call_without_opening_tag(self) -> None:
        raw = 'Oke {"name": "remember_note", "arguments": {"fact": "suka kopi"}}</tool_call>'
        res = await self._call_ollama(raw)
        self.assertEqual(res.tool_calls, [LLMToolCall("remember_note", {"fact": "suka kopi"})])
        self.assertEqual(res.text, "Oke")

    async def test_call_inside_code_fence(self) -> None:
        raw = 'Sebentar\n```json\n{"name": "web_search", "arguments": {"query": "cuaca jakarta"}}\n```'
        res = await self._call_ollama(raw)
        self.assertEqual(res.tool_calls, [LLMToolCall("web_search", {"query": "cuaca jakarta"})])
        self.assertEqual(res.text, "Sebentar")

    async def test_normal_text_mentioning_name_preserved(self) -> None:
        raw = 'Hai! Apa "name" kamu? JSON contoh: {"name": "Pachinko"}'
        res = await self._call_ollama(raw)
        self.assertEqual(res.tool_calls, [])
        self.assertEqual(res.text, raw)

    async def test_unknown_tool_name_and_non_dict_args_stripped_never_executed(self) -> None:
        raw1 = 'Coba <tool_call>{"name": "delete_db", "arguments": {"all": true}}</tool_call> ya'
        res1 = await self._call_ollama(raw1)
        self.assertEqual(res1.tool_calls, [])
        self.assertEqual(res1.text, "Coba  ya")

        raw2 = 'Hai {"name": "delete_db", "arguments": {"all": true}}'
        res2 = await self._call_ollama(raw2)
        self.assertEqual(res2.tool_calls, [])
        self.assertEqual(res2.text, "Hai")

        raw3 = '<tool_call>{"name": "web_search", "arguments": [1, 2]}</tool_call>'
        res3 = await self._call_ollama(raw3)
        self.assertEqual(res3.tool_calls, [])
        self.assertEqual(res3.text, "")


class TestToolCallStream(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = LLMClient()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_ollama_stream_split_in_tag(self) -> None:
        chunks = [
            "Halo <to",
            'ol_call>{"name":"remember_note","arguments":{"fact":"x"}}</to',
            "ol_call> dunia",
        ]
        lines = [json.dumps({"message": {"content": c}}) for c in chunks]
        self.client._http.stream = MagicMock(return_value=_FakeStreamResponse(lines))  # type: ignore[method-assign]
        out = "".join([t async for t in self.client._ollama_stream([{"role": "user", "content": "hi"}], True)])
        self.assertEqual(out, "Halo  dunia")

    async def test_openai_stream_split_bare_json_with_braces(self) -> None:
        chunks = [
            "Oke {",
            '"na',
            'me": "web_search", "arguments": {"query": "a}b"}}',
            "</tool_call> sip",
        ]
        lines = [f'data: {json.dumps({"choices": [{"delta": {"content": c}}]})}' for c in chunks] + ["data: [DONE]"]
        self.client._http.stream = MagicMock(return_value=_FakeStreamResponse(lines))  # type: ignore[method-assign]
        out = "".join([t async for t in self.client._openai_stream([{"role": "user", "content": "hi"}], True)])
        self.assertEqual(out, "Oke  sip")

    async def test_stream_normal_text_with_name_intact(self) -> None:
        chunks = ['Apa "na', 'me" kamu? ', "{bukan}"]
        lines = [json.dumps({"message": {"content": c}}) for c in chunks]
        self.client._http.stream = MagicMock(return_value=_FakeStreamResponse(lines))  # type: ignore[method-assign]
        out = "".join([t async for t in self.client._ollama_stream([{"role": "user", "content": "hi"}], True)])
        self.assertEqual(out, 'Apa "name" kamu? {bukan}')


class TestNoToolsRetryAndErrors(unittest.IsolatedAsyncioTestCase):
    async def test_ollama_does_not_support_tools_retries_once_and_remembers(self) -> None:
        client = LLMClient()
        try:
            seen_payloads: list[dict[str, Any]] = []

            async def fake_post(url: str, json: dict[str, Any], **kwargs: Any) -> httpx.Response:
                seen_payloads.append(dict(json))
                if "tools" in json:
                    return _make_httpx_response(
                        400, {"error": "registry.ollama.ai/library/hivemind-8k does not support tools"}
                    )
                return _make_httpx_response(200, {"message": {"content": "Halo!"}})

            client._http.post = fake_post  # type: ignore[method-assign]
            dummy_tools = [{"type": "function", "function": {"name": "web_search"}}]

            with patch("brain.llm.config.ollama_model", "hivemind-8k"):
                with self.assertLogs("llm", level="WARNING") as cm:
                    res1 = await client._ollama_once([{"role": "user", "content": "hi"}], tools=dummy_tools, fast_voice=False)
                self.assertEqual(res1.text, "Halo!")
                self.assertEqual(len(seen_payloads), 2)
                self.assertIn("tools", seen_payloads[0])
                self.assertNotIn("tools", seen_payloads[1])
                self.assertFalse(client.supports_tools())
                self.assertEqual(len(cm.output), 1)

                # Second call must skip tools without retrying or logging again
                res2 = await client._ollama_once([{"role": "user", "content": "hi"}], tools=dummy_tools, fast_voice=False)
                self.assertEqual(res2.text, "Halo!")
                self.assertEqual(len(seen_payloads), 3)
                self.assertNotIn("tools", seen_payloads[2])
        finally:
            await client.close()

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

    async def test_run_text_turn_llm_failure_returns_fallback(self) -> None:
        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(side_effect=httpx.ConnectTimeout("timeout")),
            supports_tools=MagicMock(return_value=True),
        )
        runner = self._make_runner(llm)
        with self.assertLogs("brain.runner", level="ERROR"):
            res = await runner.run_text_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="Halo")
        self.assertFalse(res.silent)
        self.assertEqual(res.text, "Otakku lagi error, coba lagi bentar ya")
        # User message was recorded, but error fallback was not saved as assistant transcript
        self.assertEqual(runner.memory.add_transcript.await_count, 1)

    async def test_stream_voice_turn_llm_failure_is_quiet_and_skips_tools_when_unsupported(self) -> None:
        async def failing_stream(*args: Any, **kwargs: Any):
            raise httpx.ConnectTimeout("timeout")
            yield ""  # pragma: no cover

        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(),
            stream_text=failing_stream,
            supports_tools=MagicMock(return_value=False),
        )
        runner = self._make_runner(llm)
        with self.assertLogs("brain.runner", level="ERROR"):
            items = [x async for x in runner.stream_voice_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="ingetin aku")]
        self.assertEqual(items, [])
        llm.complete_with_tools.assert_not_called()

    async def test_run_text_turn_empty_after_stripping_is_silent(self) -> None:
        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(return_value=types.SimpleNamespace(text="", tool_calls=[])),
            supports_tools=MagicMock(return_value=True),
        )
        runner = self._make_runner(llm)
        res = await runner.run_text_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="Halo")
        self.assertTrue(res.silent)
        self.assertEqual(res.text, "")

    async def test_run_text_turn_tool_round_no_empty_assistant_and_single_round(self) -> None:
        import copy
        from brain.llm import LLMResponse
        from brain.tools import TOOL_SCHEMAS

        calls: list[tuple[list[dict[str, Any]], Any]] = []

        async def fake_complete(messages: list[dict[str, Any]], tools: Any = None, fast_voice: bool = False):
            calls.append((copy.deepcopy(messages), tools))
            if len(calls) == 1:
                return LLMResponse(text="", tool_calls=[LLMToolCall("set_reminder", {"minutes": 1, "note": "minum"})])
            return LLMResponse(
                text="Siap, udah aku ingetin ya!",
                tool_calls=[LLMToolCall("web_search", {"query": "ignored"})],
            )

        llm = types.SimpleNamespace(
            complete_with_tools=fake_complete,
            supports_tools=MagicMock(return_value=True),
        )
        runner = self._make_runner(llm)
        res = await runner.run_text_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="ingatkan aku minum")
        self.assertEqual(res.text, "Siap, udah aku ingetin ya!")
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][1], TOOL_SCHEMAS)
        self.assertIsNone(calls[1][1])
        assistant_msgs = [m["content"] for m in calls[1][0] if m["role"] == "assistant"]
        self.assertTrue(all(c.strip() != "" for c in assistant_msgs))
        self.assertIn("[memanggil tool: set_reminder]", assistant_msgs)

    async def test_stream_voice_turn_tool_round_no_empty_assistant(self) -> None:
        import copy
        from brain.llm import LLMResponse

        seen_messages: list[dict[str, Any]] = []

        async def fake_stream(messages: list[dict[str, Any]], fast_voice: bool = True):
            seen_messages.extend(copy.deepcopy(messages))
            yield "Siap, udah aku ingetin ya!"

        llm = types.SimpleNamespace(
            complete_with_tools=AsyncMock(
                return_value=LLMResponse(text="", tool_calls=[LLMToolCall("set_reminder", {"minutes": 1, "note": "minum"})])
            ),
            stream_text=fake_stream,
            supports_tools=MagicMock(return_value=True),
        )
        runner = self._make_runner(llm)
        items = [x async for x in runner.stream_voice_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="ingatkan aku minum 1 menit lagi")]
        self.assertTrue(items)
        assistant_msgs = [m["content"] for m in seen_messages if m["role"] == "assistant"]
        self.assertTrue(all(c.strip() != "" for c in assistant_msgs))
        self.assertIn("[memanggil tool: set_reminder]", assistant_msgs)


class TestToolBoxAllowlist(unittest.IsolatedAsyncioTestCase):
    async def test_web_read_removed_and_unknown_tools_rejected(self) -> None:
        from brain.gate import RISK_LEVELS
        from brain.tools import ToolBox

        self.assertFalse(hasattr(ToolBox, "_web_read"))
        self.assertNotIn("web_read", RISK_LEVELS)

        gate = types.SimpleNamespace(check=AsyncMock(return_value=(True, "ok")))
        box = ToolBox(memory=types.SimpleNamespace(), gate=gate)  # type: ignore[arg-type]

        for bad_name, args in (
            ("web_read", {"url": "http://127.0.0.1:7878/"}),
            ("delete_db", {}),
        ):
            with self.assertLogs("brain.tools", level="WARNING"):
                out = await box.execute(bad_name, args, room_id="r1", user_id="u1", user_name="Budi")
            self.assertTrue(out.startswith("Unknown tool"), f"Expected Unknown tool for {bad_name}, got {out!r}")
        gate.check.assert_not_called()


class TestRememberNoteGuard(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_event_logs_scores_and_short_notes(self) -> None:
        from brain.tools import ToolBox

        memory = types.SimpleNamespace(
            upsert_note=AsyncMock(),
            remember=AsyncMock(),
        )
        gate = types.SimpleNamespace(check=AsyncMock(return_value=(True, "ok")))
        box = ToolBox(memory=memory, gate=gate)  # type: ignore[arg-type]

        rejected = (
            "Glizzy Glazer asked about my current drink",
            'Narenee said "patata"',
            "baru kenal. Bond 16.2/1000",
            "ok",
        )
        for fact in rejected:
            with self.assertLogs("brain.tools", level="INFO"):
                out = await box.execute("remember_note", {"fact": fact}, room_id="r1", user_id="u1", user_name="Budi")
            self.assertTrue(out.startswith("Skipped note:"), f"Expected Skipped note for {fact!r}, got {out!r}")
        memory.upsert_note.assert_not_called()
        memory.remember.assert_not_called()

    async def test_accepts_valid_indonesian_facts(self) -> None:
        from brain.tools import ToolBox

        memory = types.SimpleNamespace(
            upsert_note=AsyncMock(return_value=1),
            remember=AsyncMock(return_value=1),
        )
        gate = types.SimpleNamespace(check=AsyncMock(return_value=(True, "ok")))
        box = ToolBox(memory=memory, gate=gate)  # type: ignore[arg-type]

        for fact in ("Suka kopi susu gula aren", "Kuliah di UBSI jurusan informatika"):
            out = await box.execute("remember_note", {"fact": fact}, room_id="r1", user_id="u1", user_name="Budi")
            self.assertTrue(out.startswith("Saved to notebook"), f"Expected Saved for {fact!r}, got {out!r}")
        self.assertEqual(memory.upsert_note.await_count, 2)
        self.assertEqual(memory.remember.await_count, 2)


if __name__ == "__main__":
    unittest.main()
