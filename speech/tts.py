"""Multi-Backend Text-to-Speech Engine + Auto-Download Piper ONNX + Filler Cache + RVC Hook.

Supports:
1. `piper`      : 100% Local Offline ONNX (`id_ID-news_tts-medium.onnx`, auto-downloaded if missing).
2. `edge`       : Expressive Indonesian Neural Voice (`id-ID-GadisNeural`), 100% free without quota & 0 VRAM,
                  automatically falls back to local `piper` when offline!
3. `elevenlabs` : Optional if the user provides an API key + Voice ID.
+ Optional Local RVC Post-Processing (`RVC_ENABLED=true`) to convert voice timbre to a `.pth`/`.onnx` character model.

All output is converted to **PCM 48kHz Stereo s16le** for real-time pause/duck playback control.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import subprocess
import tempfile
import wave
from pathlib import Path

import httpx

from config import DATA_DIR, MODELS_DIR, config

log = logging.getLogger("speech.tts")

PIPER_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium"
PIPER_DEFAULT_FILES = ("id_ID-news_tts-medium.onnx", "id_ID-news_tts-medium.onnx.json")

FILLER_CACHE_DIR = DATA_DIR / "filler_cache"
FILLER_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def audio_bytes_to_pcm48k(raw_audio: bytes, input_fmt: str = "mp3") -> bytes:
    """Decode any MP3/WAV into raw PCM 48kHz stereo int16 using FFmpeg in memory."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", input_fmt, "-i", "pipe:0",
            "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", "pipe:1",
        ],
        input=raw_audio,
        capture_output=True,
        check=True,
    )
    return proc.stdout


class TTSEngine:
    def __init__(self) -> None:
        self._piper_voice = None
        self._piper_loaded_path: Path | None = None
        self._lock = asyncio.Lock()
        self._http = httpx.AsyncClient(timeout=45.0, follow_redirects=True)

    async def close(self) -> None:
        await self._http.aclose()

    async def ensure_piper_model(self) -> Path:
        model_path = config.piper_model_path
        json_path = Path(str(model_path) + ".json")
        if model_path.exists() and json_path.exists():
            return model_path

        target_dir = MODELS_DIR / "piper"
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in PIPER_DEFAULT_FILES:
            dest = target_dir / fname
            if not dest.exists():
                url = f"{PIPER_HF_BASE}/{fname}"
                log.info("Downloading local Indonesian Piper voice: %s ...", fname)
                r = await self._http.get(url)
                r.raise_for_status()
                dest.write_bytes(r.content)
        return target_dir / PIPER_DEFAULT_FILES[0]

    async def synthesize_pcm(self, text: str, *, emotion: str = "neutral") -> bytes:
        """Generate raw PCM 48kHz stereo int16 from a single sentence."""
        clean = text.strip()
        if not clean:
            return b""

        provider = config.tts_provider
        wav_or_mp3: bytes
        fmt: str

        try:
            if provider == "piper":
                wav_or_mp3, fmt = await self._synth_piper(clean), "wav"
            elif provider == "elevenlabs" and config.elevenlabs_api_key and config.elevenlabs_voice_id:
                wav_or_mp3, fmt = await self._synth_elevenlabs(clean), "mp3"
            else:
                wav_or_mp3, fmt = await self._synth_edge(clean, emotion=emotion), "mp3"
        except Exception as e:
            log.warning("TTS '%s' failed (%s), falling back to local offline Piper...", provider, e)
            wav_or_mp3, fmt = await self._synth_piper(clean), "wav"

        if config.rvc_enabled:
            wav_or_mp3, fmt = await self._apply_rvc_if_available(wav_or_mp3, fmt)

        return await asyncio.to_thread(audio_bytes_to_pcm48k, wav_or_mp3, fmt)

    async def get_filler_pcm(self, phrase: str) -> bytes:
        """Fetch (or generate once and cache on disk) a short 'Hmm?' filler clip with zero delay."""
        key = hashlib.md5(f"{config.tts_provider}:{config.edge_tts_voice}:{phrase}".encode()).hexdigest()[:12]
        cache_file = FILLER_CACHE_DIR / f"{key}.pcm"
        if cache_file.exists():
            return cache_file.read_bytes()
        pcm = await self.synthesize_pcm(phrase, emotion="thinking")
        if pcm:
            cache_file.write_bytes(pcm)
        return pcm

    async def _synth_edge(self, text: str, emotion: str = "neutral") -> bytes:
        import edge_tts

        # Subtle expression variations based on character mood
        pitch = config.edge_tts_pitch
        rate = config.edge_tts_rate
        if emotion in ("happy", "surprised"):
            pitch = "+12Hz"
            rate = "+9%"
        elif emotion in ("sad",):
            pitch = "+2Hz"
            rate = "-4%"
        elif emotion in ("angry", "annoyed"):
            pitch = "+4Hz"
            rate = "+10%"

        comm = edge_tts.Communicate(text, voice=config.edge_tts_voice, rate=rate, pitch=pitch)
        out = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                out.extend(chunk["data"])
        if not out:
            raise RuntimeError("Edge TTS returned empty audio")
        return bytes(out)

    async def _synth_piper(self, text: str) -> bytes:
        model_path = await self.ensure_piper_model()

        def _run() -> bytes:
            from piper import PiperVoice

            if self._piper_voice is None or self._piper_loaded_path != model_path:
                self._piper_voice = PiperVoice.load(str(model_path))
                self._piper_loaded_path = model_path
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                self._piper_voice.synthesize_wav(text, wf)
            return buf.getvalue()

        async with self._lock:
            return await asyncio.to_thread(_run)

    async def _synth_elevenlabs(self, text: str) -> bytes:
        r = await self._http.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{config.elevenlabs_voice_id}",
            params={"output_format": "mp3_44100_128"},
            headers={"xi-api-key": config.elevenlabs_api_key},
            json={
                "text": text,
                "model_id": config.elevenlabs_model,
                "voice_settings": {"stability": 0.45, "similarity_boost": 0.8},
            },
        )
        if r.status_code != 200:
            raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:200]}")
        return r.content

    async def _apply_rvc_if_available(self, audio_bytes: bytes, fmt: str) -> tuple[bytes, str]:
        """If a local RVC / Applio server is active, convert voice timbre to the RVC character."""
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp_in:
                tmp_in.write(audio_bytes)
                in_path = Path(tmp_in.name)
            r = await self._http.post(
                f"{config.rvc_api_url.rstrip('/')}/convert",
                json={
                    "input_path": str(in_path),
                    "model_path": config.rvc_model_path,
                    "f0_up_key": config.rvc_pitch_shift,
                },
                timeout=20.0,
            )
            in_path.unlink(missing_ok=True)
            if r.status_code == 200 and r.content:
                return r.content, "wav"
        except Exception as e:
            log.debug("RVC skipped (%s), using base TTS voice.", e)
        return audio_bytes, fmt
