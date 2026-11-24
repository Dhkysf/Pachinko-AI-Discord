"""Offline unit tests for AUDIT #13, #14, and #30."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import main
import sing.engine as sing_engine
from core.bus import EventBus
from core.store import Store
from sing.engine import SingingEngine
from speech.tts import TTSEngine


class TestEventBusAndStore(unittest.IsolatedAsyncioTestCase):
    async def test_events_not_persisted_and_recent_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "test.db")
            await store.init()
            try:
                bus = EventBus()
                self.assertFalse(hasattr(bus, "set_persist_hook"))
                self.assertFalse(hasattr(bus, "on"))
                self.assertFalse(hasattr(store, "log_event"))

                await bus.emit("chat.msg", text="hi")
                recent = bus.recent("*", limit=60)
                self.assertEqual(len(recent), 1)
                self.assertEqual(recent[0]["topic"], "chat.msg")
                self.assertEqual(recent[0]["payload"], {"text": "hi"})

                cur = await store.conn.execute("SELECT COUNT(*) FROM events_log")
                row = await cur.fetchone()
                self.assertEqual(row[0], 0)
            finally:
                await store.close()

    async def test_startup_cleanup_deletes_only_rows_older_than_7_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test.db"
            store = Store(db_path)
            await store.init()
            now = time.time()
            await store.write(
                "INSERT INTO events_log (id, topic, ts, payload_json) VALUES (?, ?, ?, ?)",
                ("old1", "chat.msg", now - 8 * 86400, "{}"),
            )
            await store.write(
                "INSERT INTO events_log (id, topic, ts, payload_json) VALUES (?, ?, ?, ?)",
                ("new1", "chat.msg", now - 1 * 86400, "{}"),
            )
            await store.close()

            store2 = Store(db_path)
            await store2.init()
            try:
                cur = await store2.conn.execute("SELECT id FROM events_log ORDER BY id")
                rows = [r[0] for r in await cur.fetchall()]
                self.assertEqual(rows, ["new1"])
            finally:
                await store2.close()


class TestSongCacheAndRvc(unittest.IsolatedAsyncioTestCase):
    def test_trim_song_cache_deletes_oldest_and_respects_protected(self) -> None:
        trim_fn = getattr(sing_engine, "trim_song_cache", None)
        self.assertIsNotNone(trim_fn, "trim_song_cache must exist in sing.engine")

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            old_m4a = d / "old.m4a"
            old_title = d / "old.title"
            mid_m4a = d / "mid.m4a"
            mid_title = d / "mid.title"
            new_m4a = d / "new.m4a"
            new_title = d / "new.title"

            for p, sz in (
                (old_m4a, 100),
                (old_title, 10),
                (mid_m4a, 100),
                (mid_title, 10),
                (new_m4a, 100),
                (new_title, 10),
            ):
                p.write_bytes(b"x" * sz)

            now = time.time()
            os.utime(old_m4a, (now - 300, now - 300))
            os.utime(old_title, (now - 300, now - 300))
            os.utime(mid_m4a, (now - 200, now - 200))
            os.utime(mid_title, (now - 200, now - 200))
            os.utime(new_m4a, (now - 100, now - 100))
            os.utime(new_title, (now - 100, now - 100))

            trim_fn(d, max_bytes=220, protected={old_m4a})
            self.assertTrue(old_m4a.exists())
            self.assertTrue(old_title.exists())
            self.assertFalse(mid_m4a.exists())
            self.assertFalse(mid_title.exists())
            self.assertTrue(new_m4a.exists())
            self.assertTrue(new_title.exists())

            trim_fn(d, max_bytes=110, protected=None)
            self.assertFalse(old_m4a.exists())
            self.assertFalse(old_title.exists())
            self.assertTrue(new_m4a.exists())

    async def test_rvc_cover_pipeline_cleans_stems_and_runs_ffmpeg_in_thread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            songs_dir = Path(tmp)
            src = songs_dir / "abc.m4a"
            src.write_bytes(b"src")
            voc = songs_dir / "abc_voc.wav"
            inst = songs_dir / "abc_inst.wav"

            class FakeSeparator:
                def __init__(self, output_dir: str, output_format: str = "WAV") -> None:
                    pass

                def load_model(self) -> None:
                    pass

                def separate(self, _path: str) -> list[str]:
                    inst.write_bytes(b"inst")
                    voc.write_bytes(b"voc")
                    return [inst.name, voc.name]

            fake_mod = types.ModuleType("audio_separator")
            fake_sub = types.ModuleType("audio_separator.separator")
            fake_sub.Separator = FakeSeparator  # type: ignore[attr-defined]
            fake_mod.separator = fake_sub  # type: ignore[attr-defined]

            engine = SingingEngine(EventBus())
            to_thread_targets: list[object] = []
            orig_to_thread = asyncio.to_thread

            async def spy_to_thread(func, /, *args, **kwargs):
                to_thread_targets.append(func)
                return await orig_to_thread(func, *args, **kwargs)

            try:
                with patch.object(sing_engine, "SONGS_DIR", songs_dir), \
                     patch.dict(sys.modules, {"audio_separator": fake_mod, "audio_separator.separator": fake_sub}), \
                     patch.object(asyncio, "to_thread", side_effect=spy_to_thread), \
                     patch.object(subprocess, "run") as mock_run:
                    engine._http.post = AsyncMock(return_value=types.SimpleNamespace(status_code=200, content=b"rvc"))
                    out = await engine._try_rvc_cover_pipeline(src)
                    self.assertEqual(out, songs_dir / "abc_cover.wav")
                    self.assertIn(subprocess.run, to_thread_targets)
                    mock_run.assert_called_once()
                    self.assertFalse(voc.exists())
                    self.assertFalse(inst.exists())
                    self.assertFalse((songs_dir / "abc_rvc_voc.wav").exists())

                    # Failure case also cleans stems
                    (songs_dir / "abc_cover.wav").unlink(missing_ok=True)
                    engine._http.post = AsyncMock(side_effect=RuntimeError("rvc down"))
                    out_fail = await engine._try_rvc_cover_pipeline(src)
                    self.assertEqual(out_fail, src)
                    self.assertFalse(voc.exists())
                    self.assertFalse(inst.exists())
            finally:
                await engine.stop()


class TestTtsRvcTempCleanup(unittest.IsolatedAsyncioTestCase):
    async def test_tts_rvc_temp_file_deleted_on_error(self) -> None:
        tts = TTSEngine()
        captured: list[Path] = []

        async def fake_post(_url: str, json: dict, timeout: float = 20.0):
            captured.append(Path(json["input_path"]))
            raise RuntimeError("boom")

        try:
            tts._http.post = fake_post  # type: ignore[method-assign]
            audio, fmt = await tts._apply_rvc_if_available(b"fake-audio", "mp3")
            self.assertEqual((audio, fmt), (b"fake-audio", "mp3"))
            self.assertEqual(len(captured), 1)
            self.assertFalse(captured[0].exists())
        finally:
            await tts.close()


class TestShutdownOrder(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_all_order_and_resilience(self) -> None:
        shutdown_fn = getattr(main, "shutdown_all", None)
        self.assertIsNotNone(shutdown_fn, "shutdown_all must exist in main")

        calls: list[str] = []

        def rec(name: str, fail: bool = False) -> None:
            calls.append(name)
            if fail:
                raise RuntimeError(f"{name} failed")

        sess = MagicMock()
        sess.stop = AsyncMock(side_effect=lambda: rec("voice_session.stop"))

        bot = MagicMock()
        bot.voice_sessions = {1: sess}
        bot._sched_task = None
        bot.is_closed.return_value = False
        bot.close = AsyncMock(side_effect=lambda: rec("bot.close"))

        singing = MagicMock()
        singing.stop = AsyncMock(side_effect=lambda: rec("singing.stop"))

        uvi_server = types.SimpleNamespace(should_exit=False)

        async def fake_web():
            while not uvi_server.should_exit:
                await asyncio.sleep(0.01)
            calls.append("web_task.done")

        web_task = asyncio.create_task(fake_web())

        tts = MagicMock()
        tts.close = AsyncMock(side_effect=lambda: rec("tts.close"))
        llm = MagicMock()
        llm.close = AsyncMock(side_effect=lambda: rec("llm.close"))
        store = MagicMock()
        store.close = AsyncMock(side_effect=lambda: rec("store.close"))

        await shutdown_fn(
            bot=bot,
            singing=singing,
            uvi_server=uvi_server,
            web_task=web_task,
            tts=tts,
            llm=llm,
            store=store,
        )
        self.assertEqual(
            calls,
            ["voice_session.stop", "bot.close", "singing.stop", "web_task.done", "tts.close", "llm.close", "store.close"],
        )

        # Even when earlier steps raise, store.close() still runs last.
        calls.clear()
        sess.stop = AsyncMock(side_effect=lambda: rec("voice_session.stop", fail=True))
        bot.voice_sessions = {1: sess}
        bot.close = AsyncMock(side_effect=lambda: rec("bot.close", fail=True))
        uvi_server2 = types.SimpleNamespace(should_exit=False)

        async def fake_web2():
            while not uvi_server2.should_exit:
                await asyncio.sleep(0.01)
            calls.append("web_task.done")

        web_task2 = asyncio.create_task(fake_web2())
        await shutdown_fn(
            bot=bot,
            singing=singing,
            uvi_server=uvi_server2,
            web_task=web_task2,
            tts=tts,
            llm=llm,
            store=store,
        )
        self.assertEqual(calls[-1], "store.close")
        self.assertIn("singing.stop", calls)
        self.assertIn("tts.close", calls)
        self.assertIn("llm.close", calls)


if __name__ == "__main__":
    unittest.main()
