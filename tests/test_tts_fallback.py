"""Offline unit tests for TTS fallback order, ElevenLabs cooldown, and filler cache keys."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import speech.tts as tts_mod
from config import AppConfig, config
from speech.tts import TTSEngine


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestTTSFallbackConfig(unittest.TestCase):
    def test_tts_fallback_default_and_override(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_FALLBACK", None)
            cfg = AppConfig()
            self.assertEqual(cfg.tts_fallback, ("edge", "piper"))

        with patch.dict(os.environ, {"TTS_FALLBACK": "piper"}, clear=False):
            cfg = AppConfig()
            self.assertEqual(cfg.tts_fallback, ("piper",))


class TestTTSFallbackAndCooldown(unittest.IsolatedAsyncioTestCase):
    def _make_engine(self) -> TTSEngine:
        eng = TTSEngine()
        return eng

    async def test_elevenlabs_402_falls_back_to_edge_with_cooldown_and_recovery(self) -> None:
        clock = _FakeClock(1000.0)
        el_calls = 0
        edge_calls: list[tuple[str, str]] = []

        async def fake_post(url: str, **kwargs) -> httpx.Response:
            nonlocal el_calls
            el_calls += 1
            if el_calls == 1:
                return httpx.Response(
                    402,
                    json={
                        "detail": {"status": "quota_exceeded", "message": "Quota exceeded"},
                        "secret_leak": "sk-secret-key-999",
                        "extra_raw_field": "should-not-be-logged",
                    },
                )
            return httpx.Response(200, content=b"EL_MP3")

        async def fake_edge(text: str, emotion: str = "neutral") -> bytes:
            edge_calls.append((text, emotion))
            return b"EDGE_MP3"

        eng = self._make_engine()
        try:
            with (
                patch.object(config, "tts_provider", "elevenlabs"),
                patch.object(config, "tts_fallback", ("edge", "piper")),
                patch.object(config, "elevenlabs_api_key", "sk-secret-key-999"),
                patch.object(config, "elevenlabs_voice_id", "voice-123"),
                patch.object(config, "edge_tts_voice", "id-ID-GadisNeural"),
                patch.object(config, "rvc_enabled", False),
                patch.object(eng._http, "post", side_effect=fake_post),
                patch.object(eng, "_synth_edge", side_effect=fake_edge),
                patch.object(tts_mod, "audio_bytes_to_pcm48k", side_effect=lambda data, fmt: data),
                patch.object(tts_mod.time, "monotonic", side_effect=clock),
                self.assertLogs(tts_mod.log, level="INFO") as logs,
            ):
                # 1. First sentence: ElevenLabs 402 -> Edge used, cooldown started, 1 warning logged
                pcm1 = await eng.synthesize_pcm("Halo satu", emotion="happy")
                self.assertEqual(pcm1, b"EDGE_MP3")
                self.assertEqual(el_calls, 1)
                self.assertEqual(edge_calls, [("Halo satu", "happy")])

                # 2. Next sentence during cooldown (< 600s): ElevenLabs skipped, no second warning
                clock.advance(120.0)
                pcm2 = await eng.synthesize_pcm("Halo dua", emotion="sad")
                self.assertEqual(pcm2, b"EDGE_MP3")
                self.assertEqual(el_calls, 1)
                self.assertEqual(edge_calls, [("Halo satu", "happy"), ("Halo dua", "sad")])

                # 3. After 10 min (+601s from start): ElevenLabs tried again, info logged, succeeds
                clock.advance(481.0)
                pcm3 = await eng.synthesize_pcm("Halo tiga", emotion="neutral")
                self.assertEqual(pcm3, b"EL_MP3")
                self.assertEqual(el_calls, 2)

            warnings = [r for r in logs.records if r.levelname == "WARNING"]
            infos = [r for r in logs.records if r.levelname == "INFO"]
            self.assertEqual(len(warnings), 1)
            self.assertIn("ElevenLabs quota or key problem, using Edge for 10 min", warnings[0].getMessage())
            self.assertNotIn("sk-secret-key-999", warnings[0].getMessage())
            self.assertNotIn("should-not-be-logged", warnings[0].getMessage())
            self.assertEqual(len(infos), 1)
        finally:
            await eng.close()

    async def test_elevenlabs_401_429_cooldown_and_network_no_cooldown(self) -> None:
        for status in (401, 429):
            clock = _FakeClock(2000.0)
            el_calls = 0

            async def fake_post(url: str, **kwargs) -> httpx.Response:
                nonlocal el_calls
                el_calls += 1
                return httpx.Response(status, json={"detail": {"message": f"err {status}"}})

            eng = self._make_engine()
            try:
                with (
                    patch.object(config, "tts_provider", "elevenlabs"),
                    patch.object(config, "tts_fallback", ("edge", "piper")),
                    patch.object(config, "elevenlabs_api_key", "sk-test"),
                    patch.object(config, "elevenlabs_voice_id", "voice-1"),
                    patch.object(config, "rvc_enabled", False),
                    patch.object(eng._http, "post", side_effect=fake_post),
                    patch.object(eng, "_synth_edge", AsyncMock(return_value=b"EDGE")),
                    patch.object(tts_mod, "audio_bytes_to_pcm48k", side_effect=lambda data, fmt: data),
                    patch.object(tts_mod.time, "monotonic", side_effect=clock),
                ):
                    await eng.synthesize_pcm("Satu")
                    await eng.synthesize_pcm("Dua")
                    self.assertEqual(el_calls, 1)
            finally:
                await eng.close()

        # Transient network error: falls back to Edge, does NOT start cooldown
        el_net_calls = 0

        async def fake_net_err(url: str, **kwargs) -> httpx.Response:
            nonlocal el_net_calls
            el_net_calls += 1
            raise httpx.ConnectTimeout("timed out")

        eng = self._make_engine()
        try:
            with (
                patch.object(config, "tts_provider", "elevenlabs"),
                patch.object(config, "tts_fallback", ("edge", "piper")),
                patch.object(config, "elevenlabs_api_key", "sk-test"),
                patch.object(config, "elevenlabs_voice_id", "voice-1"),
                patch.object(config, "rvc_enabled", False),
                patch.object(eng._http, "post", side_effect=fake_net_err),
                patch.object(eng, "_synth_edge", AsyncMock(return_value=b"EDGE")),
                patch.object(tts_mod, "audio_bytes_to_pcm48k", side_effect=lambda data, fmt: data),
            ):
                await eng.synthesize_pcm("Satu")
                await eng.synthesize_pcm("Dua")
                self.assertEqual(el_net_calls, 2)
        finally:
            await eng.close()

    async def test_edge_also_failing_falls_back_to_piper(self) -> None:
        eng = self._make_engine()
        try:
            with (
                patch.object(config, "tts_provider", "elevenlabs"),
                patch.object(config, "tts_fallback", ("edge", "piper")),
                patch.object(config, "elevenlabs_api_key", "sk-test"),
                patch.object(config, "elevenlabs_voice_id", "voice-1"),
                patch.object(config, "rvc_enabled", False),
                patch.object(eng, "_synth_elevenlabs", AsyncMock(side_effect=RuntimeError("boom"))),
                patch.object(eng, "_synth_edge", AsyncMock(side_effect=RuntimeError("edge offline"))),
                patch.object(eng, "_synth_piper", AsyncMock(return_value=b"PIPER_WAV")),
                patch.object(tts_mod, "audio_bytes_to_pcm48k", side_effect=lambda data, fmt: data),
            ):
                pcm = await eng.synthesize_pcm("Halo")
                self.assertEqual(pcm, b"PIPER_WAV")
        finally:
            await eng.close()

    async def test_tts_fallback_piper_skips_edge(self) -> None:
        eng = self._make_engine()
        edge_mock = AsyncMock(return_value=b"EDGE")
        piper_mock = AsyncMock(return_value=b"PIPER")
        try:
            with (
                patch.object(config, "tts_provider", "elevenlabs"),
                patch.object(config, "tts_fallback", ("piper",)),
                patch.object(config, "elevenlabs_api_key", "sk-test"),
                patch.object(config, "elevenlabs_voice_id", "voice-1"),
                patch.object(config, "rvc_enabled", False),
                patch.object(eng, "_synth_elevenlabs", AsyncMock(side_effect=RuntimeError("402"))),
                patch.object(eng, "_synth_edge", edge_mock),
                patch.object(eng, "_synth_piper", piper_mock),
                patch.object(tts_mod, "audio_bytes_to_pcm48k", side_effect=lambda data, fmt: data),
            ):
                pcm = await eng.synthesize_pcm("Halo")
                self.assertEqual(pcm, b"PIPER")
                edge_mock.assert_not_called()
                piper_mock.assert_awaited_once()
        finally:
            await eng.close()

    async def test_main_edge_fallback_is_piper_only_even_if_piper_omitted(self) -> None:
        eng = self._make_engine()
        edge_mock = AsyncMock(side_effect=RuntimeError("edge down"))
        piper_mock = AsyncMock(return_value=b"PIPER")
        try:
            for fb in (("edge", "piper"), ("edge",)):
                edge_mock.reset_mock()
                piper_mock.reset_mock()
                with (
                    patch.object(config, "tts_provider", "edge"),
                    patch.object(config, "tts_fallback", fb),
                    patch.object(config, "rvc_enabled", False),
                    patch.object(eng, "_synth_edge", edge_mock),
                    patch.object(eng, "_synth_piper", piper_mock),
                    patch.object(tts_mod, "audio_bytes_to_pcm48k", side_effect=lambda data, fmt: data),
                ):
                    pcm = await eng.synthesize_pcm("Halo")
                    self.assertEqual(pcm, b"PIPER")
                    self.assertEqual(edge_mock.await_count, 1)
                    self.assertEqual(piper_mock.await_count, 1)
        finally:
            await eng.close()

    async def test_filler_cache_keys_differ_per_actual_provider(self) -> None:
        clock = _FakeClock(5000.0)
        el_calls = 0

        async def fake_post(url: str, **kwargs) -> httpx.Response:
            nonlocal el_calls
            el_calls += 1
            if el_calls == 1:
                return httpx.Response(402, json={"detail": {"message": "quota exceeded"}})
            return httpx.Response(200, content=b"EL_FILLER")

        eng = self._make_engine()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_dir = Path(tmpdir)
                edge_mock = AsyncMock(return_value=b"EDGE_FILLER")
                with (
                    patch.object(tts_mod, "FILLER_CACHE_DIR", cache_dir),
                    patch.object(config, "tts_provider", "elevenlabs"),
                    patch.object(config, "tts_fallback", ("edge", "piper")),
                    patch.object(config, "elevenlabs_api_key", "sk-test"),
                    patch.object(config, "elevenlabs_voice_id", "voice-1"),
                    patch.object(config, "edge_tts_voice", "id-ID-GadisNeural"),
                    patch.object(config, "rvc_enabled", False),
                    patch.object(eng._http, "post", side_effect=fake_post),
                    patch.object(eng, "_synth_edge", edge_mock),
                    patch.object(tts_mod, "audio_bytes_to_pcm48k", side_effect=lambda data, fmt: data),
                    patch.object(tts_mod.time, "monotonic", side_effect=clock),
                ):
                    # 1. ElevenLabs 402 -> Edge filler cached under Edge key
                    p1 = await eng.get_filler_pcm("Hmm?")
                    self.assertEqual(p1, b"EDGE_FILLER")
                    self.assertEqual(edge_mock.await_count, 1)

                    # 2. During cooldown -> reuses Edge cache without calling Edge again
                    clock.advance(60.0)
                    p2 = await eng.get_filler_pcm("Hmm?")
                    self.assertEqual(p2, b"EDGE_FILLER")
                    self.assertEqual(edge_mock.await_count, 1)

                    # 3. After cooldown -> ElevenLabs succeeds and caches under its own distinct file
                    clock.advance(600.0)
                    p3 = await eng.get_filler_pcm("Hmm?")
                    self.assertEqual(p3, b"EL_FILLER")
                    self.assertEqual(el_calls, 2)

                    files = list(cache_dir.glob("*.pcm"))
                    self.assertEqual(len(files), 2)
        finally:
            await eng.close()


if __name__ == "__main__":
    unittest.main()
