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
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import httpx

from config import DATA_DIR, MODELS_DIR, config

log = logging.getLogger("speech.tts")

PIPER_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium"
PIPER_DEFAULT_FILES = ("id_ID-news_tts-medium.onnx", "id_ID-news_tts-medium.onnx.json")

FILLER_CACHE_DIR = DATA_DIR / "filler_cache"
FILLER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ELEVENLABS_COOLDOWN_SEC = 600.0
_VALID_PROVIDERS = ("elevenlabs", "edge", "piper")
_PROVIDER_LABELS = {"elevenlabs": "ElevenLabs", "edge": "Edge", "piper": "Piper"}
_QUOTA_RE = re.compile(r"\b(401|402|429)\b|quota|limit|credit|exceeded", re.IGNORECASE)


class _ElevenLabsError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, cooldown: bool) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.cooldown = cooldown


def _elevenlabs_error_summary(resp: httpx.Response) -> str:
    """Return a short, key-safe error summary from an ElevenLabs HTTP response."""
    summary = f"HTTP {resp.status_code}"
    try:
        data = resp.json()
    except Exception:
        return summary
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, dict):
            parts = [str(detail.get("status") or "").strip(), str(detail.get("message") or "").strip()]
            msg = ": ".join(p for p in parts if p)
            if msg:
                summary = f"{summary}: {msg}"
        elif isinstance(detail, str) and detail.strip():
            summary = f"{summary}: {detail.strip()}"
    return summary[:160]


def _safe_err(exc: Exception) -> str:
    msg = str(exc)
    if config.elevenlabs_api_key:
        msg = msg.replace(config.elevenlabs_api_key, "***")
    return msg


def _is_elevenlabs_cooldown_error(exc: Exception) -> bool:
    if isinstance(exc, _ElevenLabsError):
        return exc.cooldown
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 402, 429):
        return True
    return bool(_QUOTA_RE.search(str(exc)))


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
        self._el_cooldown_until: float = 0.0
        self._el_cooldown_active: bool = False

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

    def _provider_order(self) -> list[str]:
        main = (config.tts_provider or "edge").strip().lower()
        order = [main] if main in _VALID_PROVIDERS else ["edge"]
        raw_fb = config.tts_fallback
        items = raw_fb.split(",") if isinstance(raw_fb, str) else raw_fb
        for item in items:
            p = item.strip().lower()
            if p in _VALID_PROVIDERS and p not in order:
                order.append(p)
        if "piper" not in order:
            order.append("piper")
        return order

    def _preferred_active_provider(self) -> str:
        for p in self._provider_order():
            if p == "elevenlabs":
                if not (config.elevenlabs_api_key and config.elevenlabs_voice_id):
                    continue
                if self._el_cooldown_active and time.monotonic() < self._el_cooldown_until:
                    continue
            return p
        return "piper"

    def _filler_cache_key(self, provider: str, phrase: str) -> str:
        if provider == "elevenlabs":
            voice = config.elevenlabs_voice_id
        elif provider == "piper":
            voice = config.piper_model_path.name
        else:
            voice = config.edge_tts_voice
        return hashlib.md5(f"{provider}:{voice}:{phrase}".encode()).hexdigest()[:12]

    async def synthesize_pcm(self, text: str, *, emotion: str = "neutral") -> bytes:
        """Generate raw PCM 48kHz stereo int16 from a single sentence."""
        pcm, _ = await self._synthesize_pcm_with_provider(text, emotion=emotion)
        return pcm

    async def _synthesize_pcm_with_provider(self, text: str, *, emotion: str = "neutral") -> tuple[bytes, str]:
        clean = text.strip()
        if not clean:
            return b"", ""

        order = self._provider_order()
        wav_or_mp3: bytes = b""
        fmt: str = "wav"
        actual_provider: str = "piper"

        for idx, provider in enumerate(order):
            if provider == "elevenlabs":
                if not (config.elevenlabs_api_key and config.elevenlabs_voice_id):
                    continue
                if self._el_cooldown_active:
                    if time.monotonic() < self._el_cooldown_until:
                        continue
                    self._el_cooldown_active = False
                    log.info("ElevenLabs cooldown ended, trying ElevenLabs again.")

            try:
                if provider == "piper":
                    wav_or_mp3, fmt = await self._synth_piper(clean), "wav"
                elif provider == "elevenlabs":
                    wav_or_mp3, fmt = await self._synth_elevenlabs(clean), "mp3"
                else:
                    wav_or_mp3, fmt = await self._synth_edge(clean, emotion=emotion), "mp3"
                actual_provider = provider
                break
            except Exception as e:
                if idx == len(order) - 1:
                    raise
                next_prov = order[idx + 1]
                if provider == "elevenlabs" and _is_elevenlabs_cooldown_error(e):
                    now = time.monotonic()
                    already_active = self._el_cooldown_active and now < self._el_cooldown_until
                    self._el_cooldown_until = now + ELEVENLABS_COOLDOWN_SEC
                    self._el_cooldown_active = True
                    if not already_active:
                        log.warning(
                            "ElevenLabs quota or key problem, using %s for 10 min (%s)",
                            _PROVIDER_LABELS.get(next_prov, next_prov),
                            _safe_err(e),
                        )
                else:
                    log.warning("TTS '%s' failed (%s), falling back to '%s'...", provider, _safe_err(e), next_prov)

        if config.rvc_enabled:
            wav_or_mp3, fmt = await self._apply_rvc_if_available(wav_or_mp3, fmt)

        pcm = await asyncio.to_thread(audio_bytes_to_pcm48k, wav_or_mp3, fmt)
        return pcm, actual_provider

    async def get_filler_pcm(self, phrase: str) -> bytes:
        """Fetch (or generate once and cache on disk) a short 'Hmm?' filler clip with zero delay."""
        expected = self._preferred_active_provider()
        cache_file = FILLER_CACHE_DIR / f"{self._filler_cache_key(expected, phrase)}.pcm"
        if cache_file.exists():
            return cache_file.read_bytes()
        pcm, actual_provider = await self._synthesize_pcm_with_provider(phrase, emotion="thinking")
        if pcm:
            actual_file = FILLER_CACHE_DIR / f"{self._filler_cache_key(actual_provider, phrase)}.pcm"
            actual_file.write_bytes(pcm)
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
            summary = _elevenlabs_error_summary(r)
            cooldown = r.status_code in (401, 402, 429) or bool(_QUOTA_RE.search(summary))
            raise _ElevenLabsError(f"ElevenLabs {summary}", status_code=r.status_code, cooldown=cooldown)
        return r.content

    async def _apply_rvc_if_available(self, audio_bytes: bytes, fmt: str) -> tuple[bytes, str]:
        """If a local RVC / Applio server is active, convert voice timbre to the RVC character."""
        in_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp_in:
                in_path = Path(tmp_in.name)
                tmp_in.write(audio_bytes)
            r = await self._http.post(
                f"{config.rvc_api_url.rstrip('/')}/convert",
                json={
                    "input_path": str(in_path),
                    "model_path": config.rvc_model_path,
                    "f0_up_key": config.rvc_pitch_shift,
                },
                timeout=20.0,
            )
            if r.status_code == 200 and r.content:
                return r.content, "wav"
        except Exception as e:
            log.debug("RVC skipped (%s), using base TTS voice.", e)
        finally:
            if in_path is not None:
                in_path.unlink(missing_ok=True)
        return audio_bytes, fmt
