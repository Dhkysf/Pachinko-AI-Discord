"""Offline unit tests for 9router, per-run env overrides, tool support, Gemini tools, and health checks."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import config as cfg_mod
from brain.llm import LLMClient, LLMToolCall
from brain.profile import CharacterProfile
from brain.runner import BrainRunner
from config import AppConfig
from core.runtime import write_status
from core.safety import redact_secrets


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


def _make_cfg(**overrides: Any) -> AppConfig:
    with patch.dict(os.environ, {}, clear=True):
        cfg = AppConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestConfigAndEnvOverride(unittest.TestCase):
    def test_active_model_per_provider(self) -> None:
        cfg = _make_cfg(
            ollama_model="qwen3:8b",
            ninerouter_model="nr/claude-sonnet",
            gemini_model="gemini-3.8-flash",
            openai_model="local-qwen",
        )
        for provider, expected in (
            ("ollama", "qwen3:8b"),
            ("ninerouter", "nr/claude-sonnet"),
            ("gemini", "gemini-3.8-flash"),
            ("openai_compat", "local-qwen"),
        ):
            cfg.llm_provider = provider
            self.assertEqual(cfg.active_model(), expected)

        with patch.object(cfg_mod, "config", cfg):
            cfg.llm_provider = "ninerouter"
            self.assertEqual(cfg_mod.active_model(), "nr/claude-sonnet")

    def test_env_vars_override_dotenv_in_subprocess(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            (tmp_dir / ".env").write_text(
                "LLM_PROVIDER=ollama\nNINEROUTER_MODEL=from-dotenv\n",
                encoding="utf-8",
            )
            code = (
                "import sys; "
                f"sys.path.insert(0, {str(repo_root)!r}); "
                "from config import config; "
                "print(config.llm_provider, config.ninerouter_model, config.active_model())"
            )
            env = os.environ.copy()
            env["LLM_PROVIDER"] = "ninerouter"
            env["NINEROUTER_MODEL"] = "x"
            res = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(tmp_dir),
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(res.stdout.strip(), "ninerouter x x")


class TestNineRouterAndOpenAICompat(unittest.IsolatedAsyncioTestCase):
    async def test_ninerouter_request_url_bearer_and_model_and_stream(self) -> None:
        seen: list[tuple[str, str, str, dict[str, Any]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            seen.append((
                request.method,
                str(request.url),
                request.headers.get("authorization", ""),
                body,
            ))
            if body.get("stream"):
                sse = (
                    f'data: {json.dumps({"choices": [{"delta": {"content": "Halo "}}]})}\n'
                    f'data: {json.dumps({"choices": [{"delta": {"content": "9router!"}}]})}\n'
                    "data: [DONE]\n"
                )
                return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Siap!", "tool_calls": []}}]},
            )

        client = LLMClient()
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cfg = _make_cfg(
                llm_provider="ninerouter",
                ninerouter_base_url="http://127.0.0.1:20128/v1/",
                ninerouter_api_key="nr-secret-key-123",
                ninerouter_model="gpt-4o-mini",
                gemini_api_key="",
            )
            with patch("brain.llm.config", cfg):
                resp = await client.complete_with_tools([{"role": "user", "content": "hai"}], tools=None)
                self.assertEqual(resp.text, "Siap!")
                streamed = "".join([t async for t in client.stream_text([{"role": "user", "content": "hai"}])])
                self.assertEqual(streamed, "Halo 9router!")

            self.assertEqual(len(seen), 2)
            self.assertEqual(seen[0][1], "http://127.0.0.1:20128/v1/chat/completions")
            self.assertEqual(seen[0][2], "Bearer nr-secret-key-123")
            self.assertEqual(seen[0][3]["model"], "gpt-4o-mini")
            self.assertEqual(seen[1][1], "http://127.0.0.1:20128/v1/chat/completions")
            self.assertEqual(seen[1][2], "Bearer nr-secret-key-123")
            self.assertTrue(seen[1][3]["stream"])
        finally:
            await client.close()

    async def test_openai_compat_payload_byte_identical(self) -> None:
        raw_bodies: list[bytes] = []
        auths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            raw_bodies.append(request.content)
            auths.append(request.headers.get("authorization", ""))
            body = json.loads(request.content.decode("utf-8"))
            if body.get("stream"):
                return httpx.Response(200, text="data: [DONE]\n")
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        client = LLMClient()
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cfg = _make_cfg(
                llm_provider="openai_compat",
                openai_base_url="http://127.0.0.1:1234/v1",
                openai_api_key="lm-studio",
                openai_model="qwen3-8b",
                llm_temperature=0.7,
                llm_compact_prompt=True,
                gemini_api_key="",
            )
            tools = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]
            msgs = [{"role": "user", "content": "halo"}]
            with patch("brain.llm.config", cfg):
                await client.complete_with_tools(msgs, tools=tools, fast_voice=False)
                _ = [t async for t in client.stream_text(msgs, fast_voice=True)]

            expected_once = json.dumps(
                {
                    "model": "qwen3-8b",
                    "messages": [{"role": "user", "content": "halo"}],
                    "temperature": 0.7,
                    "max_tokens": 300,
                    "tools": tools,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            expected_stream = json.dumps(
                {
                    "model": "qwen3-8b",
                    "messages": [{"role": "user", "content": "halo"}],
                    "temperature": 0.7,
                    "max_tokens": 280,
                    "stream": True,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            self.assertEqual(raw_bodies[0], expected_once)
            self.assertEqual(raw_bodies[1], expected_stream)
            self.assertEqual(auths, ["Bearer lm-studio", "Bearer lm-studio"])
        finally:
            await client.close()

    async def test_tools_rejected_400_and_422_retries_once_and_remembers(self) -> None:
        for status_code, err_body in (
            (400, {"error": "This model does not support tools"}),
            (422, {"detail": "tool_choice / function calling is not supported"}),
        ):
            seen_payloads: list[dict[str, Any]] = []

            def handler(request: httpx.Request, code: int = status_code, body: dict[str, Any] = err_body) -> httpx.Response:
                payload = json.loads(request.content.decode("utf-8"))
                seen_payloads.append(payload)
                if "tools" in payload:
                    return httpx.Response(code, json=body)
                return httpx.Response(200, json={"choices": [{"message": {"content": "Halo!"}}]})

            client = LLMClient()
            await client._http.aclose()
            client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                cfg = _make_cfg(
                    llm_provider="ninerouter",
                    ninerouter_api_key="k",
                    ninerouter_model="no-tool-model",
                    gemini_api_key="",
                )
                dummy_tools = [{"type": "function", "function": {"name": "web_search"}}]
                with patch("brain.llm.config", cfg):
                    with self.assertLogs("llm", level="WARNING") as cm:
                        r1 = await client.complete_with_tools([{"role": "user", "content": "hi"}], tools=dummy_tools)
                    self.assertEqual(r1.text, "Halo!")
                    self.assertEqual(len(seen_payloads), 2)
                    self.assertIn("tools", seen_payloads[0])
                    self.assertNotIn("tools", seen_payloads[1])
                    self.assertFalse(client.supports_tools())
                    self.assertEqual(len(cm.output), 1)

                    r2 = await client.complete_with_tools([{"role": "user", "content": "hi"}], tools=dummy_tools)
                    self.assertEqual(r2.text, "Halo!")
                    self.assertEqual(len(seen_payloads), 3)
                    self.assertNotIn("tools", seen_payloads[2])

                    # Different model still supports tools
                    cfg.ninerouter_model = "other-model"
                    self.assertTrue(client.supports_tools())
            finally:
                await client.close()

    async def test_unrelated_400_does_not_retry_or_disable_tools(self) -> None:
        seen = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen
            seen += 1
            return httpx.Response(400, json={"error": "context length exceeded"})

        client = LLMClient()
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cfg = _make_cfg(
                llm_provider="ninerouter",
                ninerouter_api_key="k",
                ninerouter_model="m1",
                gemini_api_key="",
            )
            with patch("brain.llm.config", cfg):
                with self.assertRaises(httpx.HTTPStatusError):
                    await client.complete_with_tools(
                        [{"role": "user", "content": "hi"}],
                        tools=[{"type": "function", "function": {"name": "web_search"}}],
                    )
            self.assertEqual(seen, 1)
            self.assertTrue(client.supports_tools())
        finally:
            await client.close()


class TestErrorsAndFallback(unittest.IsolatedAsyncioTestCase):
    async def test_ninerouter_401_403_429_connect_error_logs_and_never_leaks_key(self) -> None:
        secret = "super-secret-9r-key-xyz"
        cases: list[tuple[Any, str]] = [
            (httpx.Response(401, json={"error": f"bad key {secret}"}), "9router API key missing or wrong"),
            (httpx.Response(403, text=f"forbidden {secret}"), "9router API key missing or wrong"),
            (httpx.Response(429, text="too many requests"), "rate limited"),
            ("connect_error", "9router is not running at http://127.0.0.1:20128/v1"),
        ]
        for outcome, expected_msg in cases:
            def handler(request: httpx.Request, out: Any = outcome) -> httpx.Response:
                if out == "connect_error":
                    raise httpx.ConnectError("Connection refused", request=request)
                return out

            client = LLMClient()
            await client._http.aclose()
            client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                cfg = _make_cfg(
                    llm_provider="ninerouter",
                    ninerouter_base_url="http://127.0.0.1:20128/v1",
                    ninerouter_api_key=secret,
                    ninerouter_model="m1",
                    gemini_api_key="",
                )
                with patch("brain.llm.config", cfg):
                    with self.assertLogs("llm", level="WARNING") as cm:
                        with self.assertRaises(Exception):
                            await client.complete_with_tools([{"role": "user", "content": "hi"}])
                joined = "\n".join(cm.output)
                self.assertIn(expected_msg, joined)
                self.assertNotIn(secret, joined)
            finally:
                await client.close()

    async def test_runner_returns_indonesian_error_on_ninerouter_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        client = LLMClient()
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cfg = _make_cfg(llm_provider="ninerouter", ninerouter_api_key="k", ninerouter_model="m1", gemini_api_key="")
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
            runner = BrainRunner(
                store=store, bus=bus, profiles=profiles, heart=heart, memory=memory, tools=tools, llm=client  # type: ignore[arg-type]
            )
            with patch("brain.llm.config", cfg):
                res = await runner.run_text_turn(room_id="r1", user_id="u1", user_name="Budi", user_text="Halo")
            self.assertFalse(res.silent)
            self.assertEqual(res.text, "Otakku lagi error, coba lagi bentar ya")
        finally:
            await client.close()

    async def test_gemini_fallback_only_when_key_set_and_not_gemini_primary(self) -> None:
        client = LLMClient()
        try:
            cfg = _make_cfg(llm_provider="ninerouter", gemini_api_key="gem-key", gemini_model="g1")
            with patch("brain.llm.config", cfg), \
                 patch.object(client, "_openai_once", AsyncMock(side_effect=RuntimeError("down"))), \
                 patch.object(client, "_gemini_once", AsyncMock(return_value=types.SimpleNamespace(text="from-gem", tool_calls=[]))) as gem_mock:
                res = await client.complete_with_tools([{"role": "user", "content": "hi"}])
                self.assertEqual(res.text, "from-gem")
                gem_mock.assert_awaited_once()

            cfg.llm_provider = "gemini"
            with patch("brain.llm.config", cfg), \
                 patch.object(client, "_gemini_once", AsyncMock(side_effect=RuntimeError("gem-down"))) as gem_mock2:
                with self.assertRaises(RuntimeError):
                    await client.complete_with_tools([{"role": "user", "content": "hi"}])
                self.assertEqual(gem_mock2.await_count, 1)
        finally:
            await client.close()


class TestGeminiTools(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_function_calling_and_followup(self) -> None:
        calls: list[dict[str, Any]] = []

        class _FakeObj:
            def __init__(self, **kw: Any) -> None:
                self.__dict__.update(kw)

        class _FakeTypes:
            Content = _FakeObj
            Part = _FakeObj
            FunctionDeclaration = _FakeObj
            Tool = _FakeObj
            GenerateContentConfig = _FakeObj

        async def fake_generate_content(*, model: str, contents: Any, config: Any) -> Any:
            calls.append({"model": model, "contents": contents, "config": config})
            if len(calls) == 1:
                fc_part = _FakeObj(text=None, function_call=_FakeObj(name="set_reminder", args={"minutes": 1, "note": "minum"}))
                return _FakeObj(candidates=[_FakeObj(content=_FakeObj(parts=[fc_part]))], text=None)
            txt_part = _FakeObj(text="Sip, udah aku pasang pengingatnya!", function_call=None)
            return _FakeObj(candidates=[_FakeObj(content=_FakeObj(parts=[txt_part]))], text="Sip, udah aku pasang pengingatnya!")

        fake_client = types.SimpleNamespace(
            aio=types.SimpleNamespace(models=types.SimpleNamespace(generate_content=fake_generate_content), aclose=AsyncMock()),
            close=MagicMock(),
        )
        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]
        fake_genai.types = _FakeTypes  # type: ignore[attr-defined]
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai  # type: ignore[attr-defined]

        llm = LLMClient()
        cfg = _make_cfg(
            llm_provider="gemini",
            gemini_api_key="test-gem-key",
            gemini_model="gemini-3.8-flash",
            gemini_fallback_models=(),
        )
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "set_reminder",
                    "description": "Set reminder",
                    "parameters": {"type": "object", "properties": {"minutes": {"type": "number"}}},
                },
            }
        ]
        try:
            with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai, "google.genai.types": _FakeTypes}), \
                 patch("brain.llm.config", cfg):
                r1 = await llm.complete_with_tools([{"role": "user", "content": "ingetin minum 1 menit lagi"}], tools=tool_schemas)
                self.assertEqual(r1.tool_calls, [LLMToolCall("set_reminder", {"minutes": 1, "note": "minum"})])
                self.assertTrue(getattr(calls[0]["config"], "tools", None))

                r2 = await llm.complete_with_tools(
                    [{"role": "user", "content": "ingetin minum 1 menit lagi"}, {"role": "user", "content": "[Tool Result]: OK"}],
                    tools=None,
                )
                self.assertEqual(r2.text, "Sip, udah aku pasang pengingatnya!")
                self.assertIsNone(getattr(calls[1]["config"], "tools", None))
        finally:
            await llm.close()


class TestHealthStatusAndSafety(unittest.IsolatedAsyncioTestCase):
    async def test_check_health_per_provider(self) -> None:
        seen_gets: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_gets.append((str(request.url), request.headers.get("authorization", "")))
            return httpx.Response(200, json={"data": []})

        client = LLMClient()
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cfg = _make_cfg(
                ollama_base_url="http://127.0.0.1:11434",
                ninerouter_base_url="http://127.0.0.1:20128/v1",
                ninerouter_api_key="nr-key",
                openai_base_url="http://127.0.0.1:1234/v1",
                openai_api_key="oa-key",
            )
            with patch("brain.llm.config", cfg):
                cfg.llm_provider = "ollama"
                client.last_ok = False
                client._last_probe_ts = 0.0
                self.assertTrue(await client.check_health(60.0))

                cfg.llm_provider = "ninerouter"
                client.last_ok = False
                client._last_probe_ts = 0.0
                self.assertTrue(await client.check_health(60.0))

                cfg.llm_provider = "openai_compat"
                client.last_ok = False
                client._last_probe_ts = 0.0
                self.assertTrue(await client.check_health(60.0))

                cfg.llm_provider = "gemini"
                client.last_ok = False
                client._last_probe_ts = 0.0
                self.assertFalse(await client.check_health(60.0))
                client.last_ok = True
                self.assertTrue(await client.check_health(60.0))

            self.assertEqual(
                seen_gets,
                [
                    ("http://127.0.0.1:11434/api/tags", ""),
                    ("http://127.0.0.1:20128/v1/models", "Bearer nr-key"),
                    ("http://127.0.0.1:1234/v1/models", "Bearer oa-key"),
                ],
            )
        finally:
            await client.close()

    def test_write_status_includes_llm_model_and_keeps_ollama_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "status.json"
            write_status(
                p,
                state="running",
                started=100,
                discord_ready=True,
                voice_sessions=0,
                llm_ok=True,
                llm_provider="ninerouter",
                llm_model="gpt-4o-mini",
                ollama_model="qwen3:8b",
            )
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["llm_provider"], "ninerouter")
            self.assertEqual(data["llm_model"], "gpt-4o-mini")
            self.assertEqual(data["ollama_model"], "qwen3:8b")

    def test_redact_secrets_redacts_ninerouter_api_key(self) -> None:
        out = redact_secrets("NINEROUTER_API_KEY=secret_9r_value")
        self.assertNotIn("secret_9r_value", out)
        self.assertIn("[REDACTED]", out)


if __name__ == "__main__":
    unittest.main()
