"""AI Singing Engine (integrated with RX 9070 16GB VRAM Coordinator).

How it works:
1. User requests a song via `/sing <title/YouTube URL>` or voice chat (`queue_song`).
2. Downloads audio via `yt-dlp` into the `data/songs/` cache.
3. Acquires an exclusive VRAM lease via `vram_guard.lease_heavy_gpu("singing")` (offloading Ollama from RX 9070 VRAM):
   - If `RVC_ENABLED=true` and `audio-separator` is available, separates vocals (DirectML GPU) -> converts voice via RVC -> remixes with instrumental at -16 LUFS!
   - If RVC is not enabled, plays the song directly in the Discord voice channel!
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from config import DATA_DIR, config
from core.bus import EventBus
from core.vram import vram_guard
from speech.tts import audio_bytes_to_pcm48k

log = logging.getLogger("sing")

SONGS_DIR = DATA_DIR / "songs"
SONGS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SongItem:
    id: str
    room_id: str
    requester_id: str
    requester_name: str
    query: str
    title: str = "Memuat judul..."
    status: str = "queued"      # 'queued' | 'preparing' | 'playing' | 'done' | 'failed'
    cover_mode: bool = False


class SingingEngine:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.queue: list[SongItem] = []
        self.current: SongItem | None = None
        self._session_getter: Callable[[str], Any] | None = None
        self._worker_task: asyncio.Task | None = None
        self._http = httpx.AsyncClient(timeout=20.0)

    def bind_voice_lookup(self, getter: Callable[[str], Any]) -> None:
        self._session_getter = getter

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._queue_loop())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
        await self._http.aclose()

    def skip_current(self) -> bool:
        if self.current:
            session = self._session_getter(self.current.room_id) if self._session_getter else None
            if session and session._current_source:
                session._current_source.stop_now()
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "current": asdict(self.current) if self.current else None,
            "queue": [asdict(s) for s in self.queue],
        }

    async def enqueue(self, room_id: str, user_id: str, user_name: str, query: str) -> str:
        item = SongItem(
            id=uuid.uuid4().hex[:8],
            room_id=room_id,
            requester_id=user_id,
            requester_name=user_name,
            query=query.strip(),
            cover_mode=config.rvc_enabled,
        )
        self.queue.append(item)
        await self.bus.emit("sing.queued", **asdict(item))
        return f"Lagu '{query}' masuk antrean #{len(self.queue)} (ID: {item.id})."

    async def _queue_loop(self) -> None:
        while True:
            if not self.queue:
                await asyncio.sleep(1.0)
                continue
            item = self.queue.pop(0)
            self.current = item
            try:
                await self._perform_song(item)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                item.status = "failed"
                log.exception("Failed to play song '%s'", item.query)
                await self.bus.emit("sing.failed", id=item.id, error=str(e))
            finally:
                self.current = None
                await self.bus.emit("sing.idle")

    async def _perform_song(self, item: SongItem) -> None:
        session = self._session_getter(item.room_id) if self._session_getter else None
        if not session or not session.vc.is_connected():
            log.warning("No active voice session in %s to sing '%s'", item.room_id, item.query)
            return

        item.status = "preparing"
        await self.bus.emit("sing.preparing", **asdict(item))

        title, audio_file = await self._download_song(item.query)
        item.title = title

        if config.rvc_enabled:
            async with vram_guard.lease_heavy_gpu(f"sing:{title[:24]}"):
                audio_file = await self._try_rvc_cover_pipeline(audio_file)

        pcm = await asyncio.to_thread(audio_bytes_to_pcm48k, audio_file.read_bytes(), audio_file.suffix.lstrip("."))
        item.status = "playing"
        session.singing_active = True
        await self.bus.emit("sing.started", id=item.id, title=title, requester=item.requester_name)

        try:
            await session.play_raw_pcm(pcm, f"♪ {title}", is_filler=False)
            item.status = "done"
        finally:
            session.singing_active = False

    async def _download_song(self, query: str) -> tuple[str, Path]:
        slug = hashlib.md5(query.lower().encode()).hexdigest()[:12]
        cached = list(SONGS_DIR.glob(f"{slug}.*"))
        meta_file = SONGS_DIR / f"{slug}.title"
        if cached and meta_file.exists():
            audio_candidates = [p for p in cached if p.suffix in (".mp3", ".m4a", ".webm", ".wav", ".opus")]
            if audio_candidates:
                return meta_file.read_text(encoding="utf-8").strip(), audio_candidates[0]

        def _yt_sync() -> tuple[str, Path]:
            import yt_dlp

            target = query if query.startswith(("http://", "https://")) else f"ytsearch1:{query}"
            out_tmpl = str(SONGS_DIR / f"{slug}.%(ext)s")
            opts = {
                "format": "bestaudio[filesize<25M]/bestaudio/best",
                "outtmpl": out_tmpl,
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(target, download=True)
                if "entries" in info:
                    info = info["entries"][0]
                title_str = info.get("title", query)
                downloaded = Path(ydl.prepare_filename(info))
            meta_file.write_text(title_str, encoding="utf-8")
            return title_str, downloaded

        return await asyncio.to_thread(_yt_sync)

    async def _try_rvc_cover_pipeline(self, source_audio: Path) -> Path:
        """If `audio-separator` + RVC server are available, separate vocals & render an AI cover."""
        cover_out = source_audio.with_name(source_audio.stem + "_cover.wav")
        if cover_out.exists():
            return cover_out
        try:
            from audio_separator.separator import Separator  # type: ignore

            def _sep_sync() -> tuple[Path, Path]:
                sep = Separator(output_dir=str(SONGS_DIR), output_format="WAV")
                sep.load_model()
                outputs = sep.separate(str(source_audio))
                inst = SONGS_DIR / outputs[0]
                voc = SONGS_DIR / outputs[1]
                return voc, inst

            vocals_path, inst_path = await asyncio.to_thread(_sep_sync)
            r = await self._http.post(
                f"{config.rvc_api_url.rstrip('/')}/convert",
                json={
                    "input_path": str(vocals_path),
                    "model_path": config.rvc_model_path,
                    "f0_up_key": config.rvc_pitch_shift,
                },
                timeout=180.0,
            )
            if r.status_code == 200 and r.content:
                conv_voc = source_audio.with_name(source_audio.stem + "_rvc_voc.wav")
                conv_voc.write_bytes(r.content)
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(conv_voc), "-i", str(inst_path),
                        "-filter_complex", "[0:a]volume=1.25[v];[1:a]volume=0.85[i];[v][i]amix=inputs=2:duration=longest,loudnorm=I=-16",
                        str(cover_out),
                    ],
                    check=True,
                )
                return cover_out
        except Exception as e:
            log.info("Full RVC pipeline not active (%s), playing original track.", e)
        return source_audio
