"""Tests for configurable LLM_TEMPERATURE across config and LLM payloads."""
from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from brain.llm import LLMClient
from config import AppConfig


def _make_httpx_response(payload: dict[str, Any]) -> httpx.Response:
    req = httpx.Request("POST", "http://localhost")
    return httpx.Response(200, json=payload, request=req)


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.status_code = 200

    async def __aenter__(self) -> _FakeStreamResponse:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class TestLLMTemperature(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.client = LLMClient()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    def _cfg_with_env(self, value: str | None) -> AppConfig:
        env = dict(os.environ)
        if value is None:
            env.pop("LLM_TEMPERATURE", None)
        else:
            env["LLM_TEMPERATURE"] = value
        with patch.dict(os.environ, env, clear=True):
            return AppConfig()

    async def _capture_payloads(self, cfg: AppConfig) -> dict[str, dict[str, Any]]:
        captured: dict[str, dict[str, Any]] = {}
        msgs = [{"role": "user", "content": "hi"}]

        async def fake_post(url: str, json: dict[str, Any], **kwargs: Any) -> httpx.Response:
            if url.endswith("/api/chat"):
                captured["ollama_once"] = json
                return _make_httpx_response({"message": {"content": "ok"}})
            captured["openai_once"] = json
            return _make_httpx_response({"choices": [{"message": {"content": "ok"}}]})

        def fake_stream(method: str, url: str, json: dict[str, Any], **kwargs: Any) -> _FakeStreamResponse:
            if url.endswith("/api/chat"):
                captured["ollama_stream"] = json
                return _FakeStreamResponse([json_dumps({"message": {"content": "ok"}})])
            captured["openai_stream"] = json
            return _FakeStreamResponse([f'data: {json_dumps({"choices": [{"delta": {"content": "ok"}}]})}', "data: [DONE]"])

        json_dumps = json.dumps
        self.client._http.post = fake_post  # type: ignore[method-assign]
        self.client._http.stream = fake_stream  # type: ignore[method-assign]

        with patch("brain.llm.config", cfg):
            await self.client._ollama_once(msgs, tools=None, fast_voice=False)
            _ = [t async for t in self.client._ollama_stream(msgs, fast_voice=True)]
            await self.client._openai_once(msgs, tools=None, fast_voice=False)
            _ = [t async for t in self.client._openai_stream(msgs, fast_voice=True)]

        return captured

    async def test_unset_and_empty_omit_temperature(self) -> None:
        for val in (None, ""):
            cfg = self._cfg_with_env(val)
            self.assertIsNone(cfg.llm_temperature)
            payloads = await self._capture_payloads(cfg)
            self.assertNotIn("temperature", payloads["ollama_once"]["options"])
            self.assertNotIn("temperature", payloads["ollama_stream"]["options"])
            self.assertNotIn("temperature", payloads["openai_once"])
            self.assertNotIn("temperature", payloads["openai_stream"])

    async def test_compact_prompt_caps_text_num_predict_to_300(self) -> None:
        cfg_compact = self._cfg_with_env(None)
        cfg_compact.llm_compact_prompt = True
        p_compact = await self._capture_payloads(cfg_compact)
        self.assertEqual(p_compact["ollama_once"]["options"]["num_predict"], 300)
        self.assertEqual(p_compact["ollama_stream"]["options"]["num_predict"], 280)
        self.assertEqual(p_compact["openai_once"]["max_tokens"], 300)
        self.assertEqual(p_compact["openai_stream"]["max_tokens"], 280)

        cfg_legacy = self._cfg_with_env(None)
        cfg_legacy.llm_compact_prompt = False
        p_legacy = await self._capture_payloads(cfg_legacy)
        self.assertEqual(p_legacy["ollama_once"]["options"]["num_predict"], 650)
        self.assertEqual(p_legacy["openai_once"]["max_tokens"], 650)

    async def test_valid_temperature_sent_exact(self) -> None:
        cfg = self._cfg_with_env("0.6")
        self.assertEqual(cfg.llm_temperature, 0.6)
        payloads = await self._capture_payloads(cfg)
        self.assertEqual(payloads["ollama_once"]["options"]["temperature"], 0.6)
        self.assertEqual(payloads["ollama_stream"]["options"]["temperature"], 0.6)
        self.assertEqual(payloads["openai_once"]["temperature"], 0.6)
        self.assertEqual(payloads["openai_stream"]["temperature"], 0.6)

    async def test_zero_temperature_falsy_boundary_still_sent(self) -> None:
        cfg = self._cfg_with_env("0.0")
        self.assertEqual(cfg.llm_temperature, 0.0)
        payloads = await self._capture_payloads(cfg)
        self.assertEqual(payloads["ollama_once"]["options"]["temperature"], 0.0)
        self.assertEqual(payloads["ollama_stream"]["options"]["temperature"], 0.0)
        self.assertEqual(payloads["openai_once"]["temperature"], 0.0)
        self.assertEqual(payloads["openai_stream"]["temperature"], 0.0)

    async def test_invalid_or_out_of_range_warns_and_omits_temperature(self) -> None:
        for bad in ("abc", "-0.1", "2.5"):
            with self.assertLogs("config", level="WARNING") as cm:
                cfg = self._cfg_with_env(bad)
            self.assertEqual(len(cm.output), 1)
            self.assertIsNone(cfg.llm_temperature)
            payloads = await self._capture_payloads(cfg)
            self.assertNotIn("temperature", payloads["ollama_once"]["options"])
            self.assertNotIn("temperature", payloads["ollama_stream"]["options"])
            self.assertNotIn("temperature", payloads["openai_once"])
            self.assertNotIn("temperature", payloads["openai_stream"])

    async def test_compact_prompt_caps_text_num_predict_to_300(self) -> None:
        cfg_compact = self._cfg_with_env(None)
        cfg_compact.llm_compact_prompt = True
        p_compact = await self._capture_payloads(cfg_compact)
        self.assertEqual(p_compact["ollama_once"]["options"]["num_predict"], 300)
        self.assertEqual(p_compact["ollama_stream"]["options"]["num_predict"], 280)
        self.assertEqual(p_compact["openai_once"]["max_tokens"], 300)
        self.assertEqual(p_compact["openai_stream"]["max_tokens"], 280)

        cfg_legacy = self._cfg_with_env(None)
        cfg_legacy.llm_compact_prompt = False
        p_legacy = await self._capture_payloads(cfg_legacy)
        self.assertEqual(p_legacy["ollama_once"]["options"]["num_predict"], 650)
        self.assertEqual(p_legacy["openai_once"]["max_tokens"], 650)


if __name__ == "__main__":
    unittest.main()
