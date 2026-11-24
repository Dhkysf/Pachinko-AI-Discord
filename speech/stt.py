"""Local Speech-to-Text Engine (`faster-whisper` + optional Groq) & Self-Echo Guard."""
from __future__ import annotations

import asyncio
import io
import logging
import re
import time
import wave
from collections import deque

import numpy as np

from config import MODELS_DIR, config
from core.vram import vram_guard

log = logging.getLogger("speech.stt")

_HALLUCINATIONS = {
    "terima kasih",
    "terima kasih telah menonton",
    "terima kasih sudah menonton",
    "sampai jumpa di video berikutnya",
    "jangan lupa subscribe",
    "subtitle oleh",
    "thank you",
    "thanks for watching",
    "you",
}


def pcm48_stereo_to_float16k(pcm: bytes) -> np.ndarray:
    """Convert 48kHz stereo int16 -> 16kHz mono float32 [-1..1] directly in memory."""
    arr = np.frombuffer(pcm, dtype=np.int16)
    if arr.size % 2:
        arr = arr[:-1]
    mono = arr.reshape(-1, 2).mean(axis=1)
    mono_16k = mono[::3].astype(np.float32) / 32768.0
    return mono_16k


def pcm48_stereo_to_wav16k(pcm: bytes) -> bytes:
    f32 = pcm48_stereo_to_float16k(pcm)
    i16 = np.clip(f32 * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(i16.tobytes())
    return buf.getvalue()


def _bigrams(text: str) -> set[str]:
    clean = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(clean) < 2:
        return {clean} if clean else set()
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


class SelfEchoGuard:
    """Prevents bot audio leaking into a user's mic (without headset) from being treated as user speech."""

    def __init__(self, tail_seconds: float = 6.5) -> None:
        self.tail_seconds = tail_seconds
        self._recent_spoken: deque[tuple[float, set[str]]] = deque(maxlen=15)

    def record_bot_speech(self, text: str) -> None:
        bg = _bigrams(text)
        if bg:
            self._recent_spoken.append((time.monotonic(), bg))

    def is_echo(self, heard_text: str) -> bool:
        now = time.monotonic()
        cand = _bigrams(heard_text)
        if len(cand) < 4:
            return False
        for ts, ref in list(self._recent_spoken):
            if now - ts > self.tail_seconds:
                continue
            overlap = len(cand & ref) / max(1, len(cand))
            if overlap >= 0.72:
                return True
        return False


class STTEngine:
    def __init__(self) -> None:
        self._whisper_model = None
        self._lock = asyncio.Lock()
        self.echo_guard = SelfEchoGuard()
        vram_guard.register_offload("faster-whisper", self.offload)

    async def offload(self) -> None:
        async with self._lock:
            self._whisper_model = None

    def _ensure_whisper_sync(self):
        if self._whisper_model is None:
            from faster_whisper import WhisperModel

            cache_dir = MODELS_DIR / "whisper"
            cache_dir.mkdir(parents=True, exist_ok=True)
            device = config.whisper_device
            # On Windows + AMD Radeon without CUDA, faster-whisper automatically uses fast multi-thread CTranslate2 CPU
            if device == "auto":
                device = "cpu"
            log.info("Loading Faster-Whisper model '%s' (device=%s, compute=%s)...", config.whisper_model, device, config.whisper_compute_type)
            self._whisper_model = WhisperModel(
                config.whisper_model,
                device=device,
                compute_type=config.whisper_compute_type,
                download_root=str(cache_dir),
                cpu_threads=4,
            )
        return self._whisper_model

    async def transcribe(self, pcm_48k_stereo: bytes, *, hotwords: str = "") -> str:
        provider = config.stt_provider
        try:
            if provider == "groq" and config.groq_api_key:
                text = await self._transcribe_groq(pcm_48k_stereo, hotwords)
            else:
                text = await self._transcribe_local(pcm_48k_stereo, hotwords)
        except Exception as e:
            log.warning("Primary STT (%s) failed: %s", provider, e)
            if provider != "groq" and config.groq_api_key:
                text = await self._transcribe_groq(pcm_48k_stereo, hotwords)
            else:
                return ""

        cleaned = text.strip()
        norm = re.sub(r"[.!?,…]+$", "", cleaned.lower()).strip()
        if not norm or norm in _HALLUCINATIONS:
            return ""
        if self.echo_guard.is_echo(cleaned):
            log.info("STT ignored (detected bot self-echo leak): '%s'", cleaned)
            return ""
        return cleaned

    async def _transcribe_local(self, pcm: bytes, hotwords: str) -> str:
        audio_16k = pcm48_stereo_to_float16k(pcm)

        def _run() -> str:
            model = self._ensure_whisper_sync()
            segments, _ = model.transcribe(
                audio_16k,
                language=config.stt_language or None,
                beam_size=2,
                vad_filter=True,
                initial_prompt=hotwords[:180] if hotwords else None,
                condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments).strip()

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def _transcribe_groq(self, pcm: bytes, hotwords: str) -> str:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=config.groq_api_key)
        res = await client.audio.transcriptions.create(
            file=("speech.wav", pcm48_stereo_to_wav16k(pcm)),
            model=config.groq_stt_model,
            language=config.stt_language or "id",
            prompt=hotwords[:180] if hotwords else None,
            temperature=0.0,
        )
        return (res.text or "").strip()
