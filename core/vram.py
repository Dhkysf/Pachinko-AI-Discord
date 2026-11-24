"""VRAM Coordinator for AMD Radeon RX 9070 (16 GB).

Manages GPU memory sharing between Ollama (LLM), local STT/TTS, and Singing Cover (vocal separation + RVC).
When Singing needs heavy VRAM, the coordinator asks Ollama to unload its model (`keep_alive: 0`)
and temporarily releases audio model caches, then restores them afterward.
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
        """Ask Ollama to temporarily free VRAM (`keep_alive: 0`)."""
        if config.llm_provider != "ollama":
            return False
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.post(
                    f"{config.ollama_base_url.rstrip('/')}/api/generate",
                    json={"model": config.ollama_model, "prompt": "", "keep_alive": 0},
                )
                if r.status_code == 200:
                    log.info("VRAM RX 9070: Ollama model '%s' temporarily offloaded.", config.ollama_model)
                    return True
        except Exception as e:
            log.debug("Ollama offload skipped: %s", e)
        return False

    @asynccontextmanager
    async def lease_heavy_gpu(self, holder: str):
        """Acquire exclusive VRAM lease (e.g., for rendering a song cover)."""
        async with self._exclusive_lock:
            self._active_holder = holder
            if config.vram_offload_for_singing:
                await self.offload_ollama()
                for name, cb in self._offload_callbacks:
                    try:
                        await cb()
                        log.info("VRAM: '%s' offloaded for '%s'", name, holder)
                    except Exception:
                        log.warning("Failed to offload '%s'", name, exc_info=True)
            try:
                yield
            finally:
                self._active_holder = None
                log.info("VRAM: lease '%s' released.", holder)


vram_guard = VRAMCoordinator()
