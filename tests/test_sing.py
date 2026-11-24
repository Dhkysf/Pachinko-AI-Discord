"""Offline unit tests for sing/engine.py (AUDIT #6 and #25)."""
from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sing.engine as sing_engine
from core.bus import EventBus
from sing.engine import (
    _ERR_BAD_SOURCE,
    _ERR_TOO_LONG,
    SingingEngine,
    _resolve_song_target,
    audio_file_to_pcm48k,
)
from speech.tts import audio_bytes_to_pcm48k


def _gen_sine(path: Path, *codec_args: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            *codec_args, str(path),
        ],
        check=True,
    )


class TestSingDecode(unittest.TestCase):
    def test_decode_m4a_and_opus_from_file_and_old_pipe_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            m4a = tmp_dir / "tone.m4a"
            webm = tmp_dir / "tone.webm"
            opus = tmp_dir / "tone.opus"
            _gen_sine(m4a, "-c:a", "aac")
            _gen_sine(webm, "-c:a", "libopus")
            _gen_sine(opus, "-f", "ogg", "-c:a", "libopus")

            for path in (m4a, webm, opus):
                pcm = audio_file_to_pcm48k(path)
                self.assertGreater(len(pcm), 300_000, f"expected ~2s 48kHz stereo PCM for {path.name}")
                self.assertEqual(len(pcm) % 4, 0)

            with self.assertRaises(subprocess.CalledProcessError):
                audio_bytes_to_pcm48k(opus.read_bytes(), "opus")


class TestSingLimits(unittest.IsolatedAsyncioTestCase):
    async def test_url_rules(self) -> None:
        accepted = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
        ]
        for url in accepted:
            self.assertEqual(_resolve_song_target(url), url)

        self.assertEqual(_resolve_song_target("Aimer - Brave Shine"), "ytsearch1:Aimer - Brave Shine")
        self.assertEqual(_resolve_song_target("Aimer: Brave Shine"), "ytsearch1:Aimer: Brave Shine")

        refused = [
            "http://127.0.0.1:7878/",
            "http://192.168.1.1/x",
            "https://evil.example/a.mp3",
            "http://localhost:7878/api/status",
            "https://youtube.com.evil.com/watch?v=1",
            "file:///C:/Windows/win.ini",
        ]
        for bad in refused:
            with self.assertRaises(ValueError) as ctx:
                _resolve_song_target(bad)
            self.assertEqual(str(ctx.exception), _ERR_BAD_SOURCE)

        engine = SingingEngine(EventBus())
        try:
            with self.assertRaises(ValueError) as ctx:
                await engine._download_song("http://127.0.0.1:7878/")
            self.assertEqual(str(ctx.exception), _ERR_BAD_SOURCE)
        finally:
            await engine.stop()

    async def test_duration_limits_and_noplaylist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            songs_dir = Path(tmp)
            captured_opts: dict = {}
            process_calls = 0
            next_duration = 601

            class FakeYDL:
                def __init__(self, opts):
                    captured_opts.update(opts)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def extract_info(self, target, download=True):
                    self.last_download = download
                    return {"title": "Test Song", "duration": next_duration, "ext": "m4a"}

                def process_ie_result(self, info, download=True):
                    nonlocal process_calls
                    process_calls += 1
                    out = songs_dir / "test_song.m4a"
                    out.write_bytes(b"fake")
                    return {**info, "requested_downloads": [{"filepath": str(out)}]}

                def prepare_filename(self, info):
                    return str(songs_dir / "test_song.m4a")

            fake_yt_dlp = types.SimpleNamespace(YoutubeDL=FakeYDL)
            engine = SingingEngine(EventBus())
            try:
                with patch.object(sing_engine, "SONGS_DIR", songs_dir), patch.dict(sys.modules, {"yt_dlp": fake_yt_dlp}):
                    next_duration = 601
                    with self.assertRaises(ValueError) as ctx:
                        await engine._download_song("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
                    self.assertEqual(str(ctx.exception), _ERR_TOO_LONG)
                    self.assertEqual(process_calls, 0)

                    next_duration = 600
                    title, path = await engine._download_song("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
                    self.assertEqual(title, "Test Song")
                    self.assertTrue(path.exists())
                    self.assertEqual(process_calls, 1)
                    self.assertTrue(captured_opts.get("noplaylist"))
            finally:
                await engine.stop()


class TestSingSkipPreparing(unittest.IsolatedAsyncioTestCase):
    async def test_skip_while_preparing_skips_without_playing(self) -> None:
        engine = SingingEngine(EventBus())
        session = MagicMock()
        session.vc.is_connected.return_value = True
        session._current_source = None
        session.singing_active = False
        session.play_raw_pcm = AsyncMock(return_value=True)
        engine.bind_voice_lookup(lambda _room_id: session)

        started = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def fake_download(_query: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                started.set()
                await release.wait()
            return f"Song {call_count}", Path("dummy.m4a")

        engine.start()
        try:
            with patch.object(engine, "_download_song", side_effect=fake_download), \
                 patch.object(sing_engine, "audio_file_to_pcm48k", return_value=b"\x00\x00\x00\x00" * 100):
                await engine.enqueue("guild:1", "u1", "User", "first song")
                await asyncio.wait_for(started.wait(), timeout=2.0)
                first_item = engine.current
                self.assertIsNotNone(first_item)
                self.assertEqual(first_item.status, "preparing")
                self.assertTrue(engine.skip_current())
                release.set()

                for _ in range(50):
                    if engine.current is None:
                        break
                    await asyncio.sleep(0.02)

                self.assertEqual(first_item.status, "skipped")
                session.play_raw_pcm.assert_not_called()

                # Next song clears _skip_requested and plays normally
                await engine.enqueue("guild:1", "u1", "User", "second song")
                for _ in range(80):
                    if session.play_raw_pcm.await_count == 1 and engine.current is None:
                        break
                    await asyncio.sleep(0.02)
                self.assertEqual(session.play_raw_pcm.await_count, 1)
        finally:
            await engine.stop()
