"""Offline unit tests for lazy cloud client reuse and shutdown."""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import main
from brain.llm import LLMClient
from config import config
from speech.stt import STTEngine


class TestCloudClientReuse(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_client_reused_and_closed(self) -> None:
        created: list[MagicMock] = []

        def fake_client_cls(*, api_key: str) -> MagicMock:
            inst = MagicMock()
            inst.api_key = api_key
            inst.aio = types.SimpleNamespace(aclose=AsyncMock())
            inst.close = MagicMock()
            created.append(inst)
            return inst

        fake_genai = types.ModuleType("google.genai")
        fake_genai.Client = fake_client_cls  # type: ignore[attr-defined]
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai  # type: ignore[attr-defined]

        llm = LLMClient()
        with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_genai}), \
             patch.object(config, "gemini_api_key", "test-gem-key"):
            c1 = llm._get_gemini_client()
            c2 = llm._get_gemini_client()
            self.assertIs(c1, c2)
            self.assertEqual(len(created), 1)
            await llm.close()
            c1.aio.aclose.assert_awaited_once()
            c1.close.assert_called_once()
            self.assertIsNone(llm._gemini_client)

    async def test_gemini_empty_key_raises_and_close_is_safe(self) -> None:
        llm = LLMClient()
        with patch.object(config, "gemini_api_key", ""):
            with self.assertRaises(RuntimeError):
                llm._get_gemini_client()
        await llm.close()

    async def test_groq_client_reused_and_closed(self) -> None:
        created: list[MagicMock] = []

        def fake_async_groq(*, api_key: str) -> MagicMock:
            inst = MagicMock()
            inst.api_key = api_key
            inst.close = AsyncMock()
            created.append(inst)
            return inst

        fake_groq = types.ModuleType("groq")
        fake_groq.AsyncGroq = fake_async_groq  # type: ignore[attr-defined]

        stt = STTEngine()
        with patch.dict(sys.modules, {"groq": fake_groq}), \
             patch.object(config, "groq_api_key", "test-groq-key"):
            c1 = stt._get_groq_client()
            c2 = stt._get_groq_client()
            self.assertIs(c1, c2)
            self.assertEqual(len(created), 1)
            await stt.close()
            c1.close.assert_awaited_once()
            self.assertIsNone(stt._groq_client)

    async def test_groq_empty_key_returns_none_and_close_is_safe(self) -> None:
        stt = STTEngine()
        with patch.object(config, "groq_api_key", ""):
            self.assertIsNone(stt._get_groq_client())
        await stt.close()

    async def test_shutdown_all_closes_stt_before_store(self) -> None:
        calls: list[str] = []
        stt = MagicMock()
        stt.close = AsyncMock(side_effect=lambda: calls.append("stt.close"))
        store = MagicMock()
        store.close = AsyncMock(side_effect=lambda: calls.append("store.close"))
        await main.shutdown_all(stt=stt, store=store)
        self.assertEqual(calls, ["stt.close", "store.close"])


if __name__ == "__main__":
    unittest.main()
