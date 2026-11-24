"""Pemeriksa kesiapan lokal (Doctor & Pre-Download Suara Offline Piper + Cek Ollama RX 9070)."""
from __future__ import annotations

import asyncio
import shutil

import httpx

from config import config
from speech.tts import TTSEngine


async def main() -> None:
    print("=== Pachinko Local Doctor (Windows + AMD RX 9070 16GB) ===\n")

    ffmpeg_ok = shutil.which("ffmpeg") is not None
    print(f"[{'OK' if ffmpeg_ok else '!!'}] FFmpeg pada PATH: {'Ditemukan' if ffmpeg_ok else 'BELUM TERINSTALL (winget install Gyan.FFmpeg)'}")

    # 1. Unduh suara offline lokal Piper Indonesia
    tts = TTSEngine()
    try:
        p = await tts.ensure_piper_model()
        print(f"[OK] Model TTS Offline Piper siap: {p}")
    except Exception as e:
        print(f"[!!] Gagal mengunduh model Piper: {e}")
    finally:
        await tts.close()

    # 2. Cek koneksi Ollama lokal
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{config.ollama_base_url.rstrip('/')}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                print(f"[OK] Ollama lokal aktif ({config.ollama_base_url}). Model terpasang: {models or 'Belum ada'}")
                if config.ollama_model not in models and not any(m.startswith(config.ollama_model.split(':')[0]) for m in models):
                    print(f"     -> Jalankan: ollama pull {config.ollama_model}")
            else:
                print(f"[!!] Ollama merespons status {r.status_code}")
    except Exception:
        print(
            f"[INFO] Ollama belum berjalan di {config.ollama_base_url}.\n"
            f"       Install Ollama Windows (mendukung RX 9070 ROCm/Vulkan), lalu jalankan:\n"
            f"       ollama pull {config.ollama_model}"
        )

    print("\nSelesai! Isi DISCORD_TOKEN di file .env lalu klik dua kali run.bat.")


if __name__ == "__main__":
    asyncio.run(main())
