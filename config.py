"""Konfigurasi utama Miyoria-Py (dioptimalkan untuk Windows + AMD RX 9070 16GB)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
PROFILES_DIR = ROOT_DIR / "profiles"


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "y", "on")


def _int(name: str, default: int) -> int:
    val = os.getenv(name, "").strip()
    return int(val) if val else default


def _float(name: str, default: float) -> float:
    val = os.getenv(name, "").strip()
    return float(val) if val else default


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(x.strip() for x in raw.split(",") if x.strip())


@dataclass
class AppConfig:
    # --- Discord ---
    discord_token: str = field(default_factory=lambda: os.getenv("DISCORD_TOKEN", "").strip())
    guild_id: int | None = field(
        default_factory=lambda: int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID", "").strip() else None
    )
    owner_ids: tuple[int, ...] = field(
        default_factory=lambda: tuple(int(x) for x in _csv("OWNER_IDS") if x.isdigit())
    )

    # --- Karakter & Database ---
    profile_id: str = field(default_factory=lambda: os.getenv("PROFILE_ID", "default"))
    db_path: Path = field(default_factory=lambda: Path(os.getenv("DB_PATH", str(DATA_DIR / "miyoria.db"))))

    # --- LLM (Default: Lokal via Ollama di RX 9070 16GB, dengan opsional Gemini/OpenAI) ---
    # Pilihan: "ollama" | "openai_compat" | "gemini"
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b"))
    ollama_keep_alive: str = field(default_factory=lambda: os.getenv("OLLAMA_KEEP_ALIVE", "15m"))

    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "lm-studio"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "qwen3-8b"))

    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    gemini_fallback_models: tuple[str, ...] = field(
        default_factory=lambda: _csv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite")
    )

    # --- STT / ASR (Default: Lokal Faster-Whisper, opsional Groq) ---
    # Pilihan: "local_whisper" | "groq"
    stt_provider: str = field(default_factory=lambda: os.getenv("STT_PROVIDER", "local_whisper").lower())
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "auto"))
    whisper_compute_type: str = field(default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
    stt_language: str = field(default_factory=lambda: os.getenv("STT_LANGUAGE", "id"))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    groq_stt_model: str = field(default_factory=lambda: os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"))

    # --- TTS (Default: "piper" 100% offline ONNX atau "edge" neural Indo gratis, + opsional RVC clone) ---
    # Pilihan: "piper" | "edge" | "kokoro" | "elevenlabs"
    tts_provider: str = field(default_factory=lambda: os.getenv("TTS_PROVIDER", "edge").lower())
    piper_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("PIPER_MODEL_PATH", str(MODELS_DIR / "piper" / "id_ID-news_tts-medium.onnx"))
        )
    )
    edge_tts_voice: str = field(default_factory=lambda: os.getenv("EDGE_TTS_VOICE", "id-ID-GadisNeural"))
    edge_tts_rate: str = field(default_factory=lambda: os.getenv("EDGE_TTS_RATE", "+6%"))
    edge_tts_pitch: str = field(default_factory=lambda: os.getenv("EDGE_TTS_PITCH", "+8Hz"))
    kokoro_voice: str = field(default_factory=lambda: os.getenv("KOKORO_VOICE", "af_heart"))
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", "").strip())
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", "").strip())
    elevenlabs_model: str = field(default_factory=lambda: os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"))

    # Opsional konversi suara karakter lokal (Applio / RVC HTTP atau CLI)
    rvc_enabled: bool = field(default_factory=lambda: _bool("RVC_ENABLED", False))
    rvc_api_url: str = field(default_factory=lambda: os.getenv("RVC_API_URL", "http://127.0.0.1:7897"))
    rvc_model_path: str = field(default_factory=lambda: os.getenv("RVC_MODEL_PATH", ""))
    rvc_pitch_shift: int = field(default_factory=lambda: _int("RVC_PITCH_SHIFT", 0))

    # --- Voice & Smart Turn-Taking ---
    vad_rms_threshold: int = field(default_factory=lambda: _int("VAD_RMS_THRESHOLD", 420))
    turn_fast_silence_ms: int = field(default_factory=lambda: _int("TURN_FAST_SILENCE_MS", 550))
    turn_max_silence_ms: int = field(default_factory=lambda: _int("TURN_MAX_SILENCE_MS", 1350))
    parked_stream_ttl_ms: int = field(default_factory=lambda: _int("PARKED_STREAM_TTL_MS", 1200))
    min_speech_ms: int = field(default_factory=lambda: _int("MIN_SPEECH_MS", 320))
    filler_enabled: bool = field(default_factory=lambda: _bool("FILLER_ENABLED", True))
    filler_delay_ms: int = field(default_factory=lambda: _int("FILLER_DELAY_MS", 1300))
    allow_barge_in: bool = field(default_factory=lambda: _bool("ALLOW_BARGE_IN", True))
    log_voice_to_text: bool = field(default_factory=lambda: _bool("LOG_VOICE_TO_TEXT", True))

    # --- Autonomy & Brain ---
    default_channel_mode: str = field(default_factory=lambda: os.getenv("DEFAULT_CHANNEL_MODE", "quiet"))
    autonomy_enabled: bool = field(default_factory=lambda: _bool("AUTONOMY_ENABLED", True))
    mind_loop_interval_sec: int = field(default_factory=lambda: _int("MIND_LOOP_INTERVAL_SEC", 45))
    max_history_turns: int = field(default_factory=lambda: _int("MAX_HISTORY_TURNS", 24))

    # --- Web Admin Panel ---
    panel_host: str = field(default_factory=lambda: os.getenv("PANEL_HOST", "127.0.0.1"))
    panel_port: int = field(default_factory=lambda: _int("PANEL_PORT", 7878))
    panel_token: str = field(default_factory=lambda: os.getenv("PANEL_TOKEN", "").strip())

    # --- GPU VRAM Budget (RX 9070 = 16 GB) ---
    gpu_vram_gb: int = field(default_factory=lambda: _int("GPU_VRAM_GB", 16))
    vram_offload_for_singing: bool = field(default_factory=lambda: _bool("VRAM_OFFLOAD_FOR_SINGING", True))


config = AppConfig()
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
