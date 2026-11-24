"""Koordinator VRAM untuk AMD Radeon RX 9070 (16 GB).

Mengatur berbagi memori GPU antara Ollama (LLM), STT/TTS lokal, dan Singing Cover (pemisahan vokal + RVC).
Saat Singing membutuhkan VRAM besar, koordinator meminta Ollama melepas model (`keep_alive: 0`)
dan melepas cache model audio sementara, lalu mengembalikannya setelah selesai.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

import httpx

from config import config

log = logging.getLogger("vram")


class VRAMCoordinator:
    def __init__(self) -> None:
        self._exclusive_lock = asyncio.Lock()
        self._active_holder: str | None = None
        self._offload_callbacks: list[tuple[str, Callable[[], Awaitable[None]]]] = []

    @property
    def busy_with(self) -> str | None:
        return self._active_holder

    def register_offload(self, name: str, callback: Callable[[], Awaitable[None]]) -> None:
        self._offload_callbacks.append((name, callback))

    async def offload_ollama(self) -> bool:
        """Minta Ollama mengosongkan VRAM untuk sementara (`keep_alive: 0`)."""
        if config.llm_provider != "ollama":
            return False
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.post(
                    f"{config.ollama_base_url.rstrip('/')}/api/generate",
                    json={"model": config.ollama_model, "prompt": "", "keep_alive": 0},
                )
                if r.status_code == 200:
                    log.info("VRAM RX 9070: Model Ollama '%s' berhasil di-offload sementara.", config.ollama_model)
                    return True
        except Exception as e:
            log.debug("Offload Ollama dilewati: %s", e)
        return False

    @asynccontextmanager
    async def lease_heavy_gpu(self, holder: str):
        """Pinjam VRAM eksklusif (mis. untuk render cover lagu)."""
        async with self._exclusive_lock:
            self._active_holder = holder
            if config.vram_offload_for_singing:
                await self.offload_ollama()
                for name, cb in self._offload_callbacks:
                    try:
                        await cb()
                        log.info("VRAM: '%s' di-offload untuk '%s'", name, holder)
                    except Exception:
                        log.warning("Gagal offload '%s'", name, exc_info=True)
            try:
                yield
            finally:
                self._active_holder = None
                log.info("VRAM: lease '%s' selesai.", holder)


vram_guard = VRAMCoordinator()
