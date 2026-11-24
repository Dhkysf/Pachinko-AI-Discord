"""Local readiness checker (Doctor & Pre-Download Offline Piper Voice + Check Ollama RX 9070)."""
from __future__ import annotations

import asyncio
import shutil

import httpx

from config import config
from speech.tts import TTSEngine


async def main() -> None:
    print("=== Pachinko Local Doctor (Windows + AMD RX 9070 16GB) ===\n")

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    print(f"[{'OK' if ffmpeg_ok else '!!'}] FFmpeg on PATH: {'Found' if ffmpeg_ok else 'NOT INSTALLED (winget install Gyan.FFmpeg)'}")

    # 1. Download local offline Indonesian Piper voice
    tts = TTSEngine()
    try:
        p = await tts.ensure_piper_model()
        print(f"[OK] Offline Piper TTS model ready: {p}")
    except Exception as e:
        print(f"[!!] Failed to download Piper model: {e}")
    finally:
        await tts.close()

    # 2. Check local Ollama connection
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{config.ollama_base_url.rstrip('/')}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                print(f"[OK] Local Ollama active ({config.ollama_base_url}). Installed models: {models or 'None yet'}")
                if config.ollama_model not in models and not any(m.startswith(config.ollama_model.split(':')[0]) for m in models):
                    print(f"     -> Run: ollama pull {config.ollama_model}")
            else:
                print(f"[!!] Ollama responded with status {r.status_code}")
    except Exception:
        print(
            f"[INFO] Ollama is not running at {config.ollama_base_url}.\n"
            f"       Install Ollama for Windows (supports RX 9070 ROCm/Vulkan), then run:\n"
            f"       ollama pull {config.ollama_model}"
        )

    print("\nDone! Fill in DISCORD_TOKEN in .env, then double-click run.bat.")


if __name__ == "__main__":
    asyncio.run(main())
