"""Offline unit tests for 9router, per-run env overrides, tool support, Ollama fallback, and health checks."""
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
    def test_defaults_and_legacy_gemini_provider_warning(self) -> None:
        cfg = _make_cfg()
        self.assertEqual(cfg.llm_provider, "ninerouter")
        self.assertEqual(cfg.llm_fallback, "ollama")
        self.assertFalse(hasattr(cfg, "gemini_api_key"))

        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "AIzaIgnored123", "GEMINI_MODEL": "ignored"},
            clear=True,
        ):
            with self.assertLogs("config", level="WARNING") as cm:
                cfg_legacy = AppConfig()
        self.assertEqual(cfg_legacy.llm_provider, "ninerouter")
        self.assertEqual(len(cm.output), 1)
        self.assertIn("ninerouter", cm.output[0])

    def test_active_model_per_provider(self) -> None:
        cfg = _make_cfg(
            ollama_model="qwen3:8b",
            ninerouter_model="nr/claude-sonnet",
            openai_model="local-qwen",
        )
        for provider, expected in (
            ("ollama", "qwen3:8b"),
            ("ninerouter", "nr/claude-sonnet"),
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
                llm_fallback="none",
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
                llm_fallback="none",
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
                    llm_fallback="none",
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
                llm_fallback="none",
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
                    llm_fallback="none",
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
            cfg = _make_cfg(llm_provider="ninerouter", ninerouter_api_key="k", ninerouter_model="m1", llm_fallback="none")
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

    async def test_ollama_fallback_on_errors_cooldown_and_recovery(self) -> None:
        for outcome in (
            "connect_error",
            "timeout",
            httpx.Response(503, text="unavailable"),
            httpx.Response(429, text="rate limit"),
            httpx.Response(401, text="bad key"),
            httpx.Response(403, text="forbidden"),
        ):
            now = 1000.0
            nr_calls = 0
            ollama_calls = 0
            seen_ollama_msgs: list[list[dict[str, Any]]] = []
            seen_ollama_tools: list[Any] = []
            nr_should_fail = True

            def handler(request: httpx.Request, out: Any = outcome) -> httpx.Response:
                nonlocal nr_calls, ollama_calls
                url = str(request.url)
                body = json.loads(request.content.decode("utf-8"))
                if url.startswith("http://127.0.0.1:20128/"):
                    nr_calls += 1
                    if nr_should_fail:
                        if out == "connect_error":
                            raise httpx.ConnectError("Connection refused", request=request)
                        if out == "timeout":
                            raise httpx.ReadTimeout("Timed out", request=request)
                        return out
                    return httpx.Response(200, json={"choices": [{"message": {"content": "from-9router"}}]})
                ollama_calls += 1
                seen_ollama_msgs.append(body["messages"])
                seen_ollama_tools.append(body.get("tools"))
                return httpx.Response(200, json={"message": {"content": "from-ollama"}})

            client = LLMClient(clock=lambda: now, cooldown_sec=120.0)
            await client._http.aclose()
            client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                cfg = _make_cfg(
                    llm_provider="ninerouter",
                    llm_fallback="ollama",
                    ninerouter_base_url="http://127.0.0.1:20128/v1",
                    ninerouter_api_key="nr-key",
                    ninerouter_model="gpt-4o-mini",
                    ollama_base_url="http://127.0.0.1:11434",
                    ollama_model="qwen3:8b",
                )
                msgs = [{"role": "user", "content": "halo"}]
                tools = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]
                with patch("brain.llm.config", cfg):
                    with self.assertLogs("llm", level="WARNING") as cm_warn:
                        r1 = await client.complete_with_tools(msgs, tools=tools)
                    self.assertEqual(r1.text, "from-ollama")
                    self.assertEqual(r1.provider, "ollama")
                    self.assertEqual(nr_calls, 1)
                    self.assertEqual(ollama_calls, 1)
                    self.assertEqual(seen_ollama_msgs[0], msgs)
                    self.assertEqual(seen_ollama_tools[0], tools)
                    self.assertEqual(len(cm_warn.output), 1)
                    self.assertEqual(client.status_snapshot(), ("ollama", "qwen3:8b", True))

                    # Second message within 2 min goes straight to Ollama without calling 9router
                    now += 60.0
                    r2 = await client.complete_with_tools(msgs, tools=tools)
                    self.assertEqual(r2.text, "from-ollama")
                    self.assertEqual(nr_calls, 1)
                    self.assertEqual(ollama_calls, 2)

                    # After 2 min cooldown expires, 9router is tried again and logs 1 recovery info
                    now += 61.0
                    nr_should_fail = False
                    with self.assertLogs("llm", level="INFO") as cm_info:
                        r3 = await client.complete_with_tools(msgs, tools=tools)
                    self.assertEqual(r3.text, "from-9router")
                    self.assertEqual(nr_calls, 2)
                    self.assertEqual(ollama_calls, 2)
                    self.assertEqual(len(cm_info.output), 1)
                    self.assertEqual(client.status_snapshot(), ("ninerouter", "gpt-4o-mini", False))
            finally:
                await client.close()

    async def test_stream_text_falls_back_to_ollama_and_respects_cooldown(self) -> None:
        now = 500.0
        nr_calls = 0
        ollama_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal nr_calls, ollama_calls
            if str(request.url).startswith("http://127.0.0.1:20128/"):
                nr_calls += 1
                raise httpx.ConnectError("Connection refused", request=request)
            ollama_calls += 1
            ndjson = json.dumps({"message": {"content": "Halo dari Ollama!"}}) + "\n"
            return httpx.Response(200, text=ndjson)

        client = LLMClient(clock=lambda: now, cooldown_sec=120.0)
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cfg = _make_cfg(
                llm_provider="ninerouter",
                llm_fallback="ollama",
                ninerouter_api_key="k",
                ninerouter_model="m1",
                ollama_model="qwen3:8b",
            )
            with patch("brain.llm.config", cfg):
                out1 = "".join([t async for t in client.stream_text([{"role": "user", "content": "hi"}])])
                out2 = "".join([t async for t in client.stream_text([{"role": "user", "content": "hi"}])])
            self.assertEqual(out1, "Halo dari Ollama!")
            self.assertEqual(out2, "Halo dari Ollama!")
            self.assertEqual(nr_calls, 1)
            self.assertEqual(ollama_calls, 2)
        finally:
            await client.close()

    async def test_400_tools_error_does_not_trigger_ollama_fallback(self) -> None:
        ollama_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal ollama_calls
            if str(request.url).startswith("http://127.0.0.1:11434/"):
                ollama_calls += 1
                return httpx.Response(200, json={"message": {"content": "ollama"}})
            body = json.loads(request.content.decode("utf-8"))
            if "tools" in body:
                return httpx.Response(400, json={"error": "tools not supported"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok-no-tools"}}]})

        client = LLMClient()
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            cfg = _make_cfg(llm_provider="ninerouter", llm_fallback="ollama", ninerouter_api_key="k", ninerouter_model="m1")
            with patch("brain.llm.config", cfg):
                res = await client.complete_with_tools(
                    [{"role": "user", "content": "hi"}],
                    tools=[{"type": "function", "function": {"name": "web_search"}}],
                )
            self.assertEqual(res.text, "ok-no-tools")
            self.assertEqual(ollama_calls, 0)
        finally:
            await client.close()


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
                llm_last_provider="ollama",
                llm_last_model="qwen3:8b",
                llm_fallback_active=True,
            )
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["llm_provider"], "ninerouter")
            self.assertEqual(data["llm_model"], "gpt-4o-mini")
            self.assertEqual(data["ollama_model"], "qwen3:8b")
            self.assertEqual(data["llm_last_provider"], "ollama")
            self.assertEqual(data["llm_last_model"], "qwen3:8b")
            self.assertTrue(data["llm_fallback_active"])

    def test_redact_secrets_redacts_ninerouter_api_key(self) -> None:
        out = redact_secrets("NINEROUTER_API_KEY=secret_9r_value")
        self.assertNotIn("secret_9r_value", out)
        self.assertIn("[REDACTED]", out)


if __name__ == "__main__":
    unittest.main()
