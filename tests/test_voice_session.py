import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from brain.triage import TriageEngine
from config import config
from core.bus import EventBus
from core.store import Store
from memory.manager import MemoryManager
from speech.turn import BYTES_PER_MS, SmartTurnSink
from voice.session import FRAME_BYTES, NaturalVoiceSession


class FakeVC:
    def __init__(self) -> None:
        self.channel = SimpleNamespace(members=[])
        self._playing = False
        self._source = None
        self._after = None
        self._play_task: asyncio.Task | None = None
        self.played_texts: list[str] = []
        self.stop_calls = 0

    def is_connected(self) -> bool:
        return True

    def is_recording(self) -> bool:
        return False

    def start_recording(self, sink) -> None:
        pass

    def stop_recording(self) -> None:
        pass

    async def disconnect(self, force: bool = True) -> None:
        pass

    def is_playing(self) -> bool:
        return self._playing

    def play(self, src, after=None) -> None:
        self._source = src
        self._after = after
        self._playing = True
        self.played_texts.append(src.text)
        self._play_task = asyncio.create_task(self._pump(src, after))

    async def _pump(self, src, after) -> None:
        try:
            while self._playing:
                chunk = src.read()
                if not chunk:
                    break
                await asyncio.sleep(0.005)
        finally:
            self._playing = False
            if after:
                after(None)

    def stop(self) -> None:
        self.stop_calls += 1
        self._playing = False
        if self._source is not None:
            self._source.stop_now()


class FakeBrain:
    def __init__(self, turns: dict[str, list[tuple[str, str, list[str]]]] | None = None, delay_s: float = 0.03) -> None:
        self.turns = turns or {}
        self.delay_s = delay_s
        self.calls: list[str] = []

    async def stream_voice_turn(
        self,
        *,
        room_id: str,
        user_id: str,
        user_name: str,
        user_text: str,
        wake_reason: str = "voice_utterance",
    ):
        self.calls.append(user_text)
        items = self.turns.get(
            user_text,
            [
                ("Kalimat pertama nih.", "happy", []),
                ("Kalimat kedua lanjut.", "happy", []),
                ("Kalimat ketiga selesai.", "neutral", []),
            ],
        )
        for idx, item in enumerate(items):
            if idx > 0 and self.delay_s > 0:
                await asyncio.sleep(self.delay_s)
            yield item


class FakeTTS:
    def __init__(self, frames: int = 8, block_after_first: bool = False) -> None:
        self.frames = frames
        self.block_after_first = block_after_first
        self.gate = asyncio.Event()
        self.synth_tasks: list[asyncio.Task] = []
        self._count = 0

    async def get_filler_pcm(self, phrase: str) -> bytes:
        return b""

    async def synthesize_pcm(self, sentence: str, emotion: str = "neutral") -> bytes:
        task = asyncio.current_task()
        if task is not None:
            self.synth_tasks.append(task)
        self._count += 1
        if self.block_after_first and self._count > 1:
            await self.gate.wait()
        return bytes(FRAME_BYTES * self.frames)


class FakeMemory:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.marked: list[tuple[int, str]] = []

    def seed_row(self, content: str = "Balasan lama") -> int:
        row_id = len(self.rows) + 1
        self.rows.append(
            {
                "id": row_id,
                "role": "assistant",
                "modality": "voice",
                "content": content,
                "heard_content": None,
                "interrupted": 0,
                "emotion": "neutral",
            }
        )
        return row_id

    async def add_transcript(
        self,
        *,
        room_id: str,
        speaker_id: str,
        speaker_name: str,
        role: str,
        modality: str,
        content: str,
        emotion: str = "neutral",
        interrupted: int | bool = 0,
        heard_content: str | None = None,
    ) -> int:
        row_id = len(self.rows) + 1
        self.rows.append(
            {
                "id": row_id,
                "room_id": room_id,
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "role": role,
                "modality": modality,
                "content": content,
                "heard_content": heard_content,
                "interrupted": 1 if interrupted else 0,
                "emotion": emotion,
            }
        )
        return row_id

    async def mark_interrupted(self, entry_id: int, heard_content: str) -> None:
        self.marked.append((entry_id, heard_content))
        for r in self.rows:
            if r["id"] == entry_id:
                r["interrupted"] = 1
                r["heard_content"] = heard_content


class FakeSTT:
    def __init__(self) -> None:
        self.echo_guard = SimpleNamespace(record_bot_speech=lambda _s: None)
        self.heard_texts: list[str] = []

    async def transcribe(self, pcm: bytes, hotwords: str = "") -> str:
        txt = pcm.decode("utf-8", errors="ignore")
        self.heard_texts.append(txt)
        return txt


def _make_session(*, brain=None, tts=None, memory=None, stt=None):
    vc = FakeVC()
    bus = EventBus()
    prof = SimpleNamespace(
        display_name="Pachinko",
        nicknames=["pachinko"],
        wake_aliases={},
        backchannel_words=["hmm", "oke", "iya"],
        filler_phrases=[],
    )
    profiles = SimpleNamespace(current=prof)
    triage = TriageEngine()
    brain = brain or FakeBrain()
    memory = memory or FakeMemory()
    stt = stt or FakeSTT()
    tts = tts or FakeTTS()
    sess = NaturalVoiceSession(
        guild_id=1,
        vc=vc,
        text_channel=None,
        bus=bus,
        profiles=profiles,
        triage=triage,
        brain=brain,
        memory=memory,
        stt=stt,
        tts=tts,
    )
    return sess, vc, brain, memory, stt, tts


class TestVoiceSession(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._orig_barge = config.allow_barge_in
        self._orig_filler = config.filler_enabled
        self._orig_log = config.log_voice_to_text
        config.allow_barge_in = True
        config.filler_enabled = False
        config.log_voice_to_text = False

    def tearDown(self) -> None:
        config.allow_barge_in = self._orig_barge
        config.filler_enabled = self._orig_filler
        config.log_voice_to_text = self._orig_log

    async def _wait_until(self, pred, timeout: float = 1.5) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not pred():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("Timed out waiting for condition")
            await asyncio.sleep(0.005)

    async def test_worker_processes_second_utterance_while_speaking(self) -> None:
        brain = FakeBrain(delay_s=0.12)
        sess, vc, _, _, stt, _ = _make_session(brain=brain, tts=FakeTTS(frames=10))
        user = SimpleNamespace(id=1, name="Budi", display_name="Budi")
        try:
            await sess._on_utterance(user, b"pertanyaan pertama")
            await self._wait_until(lambda: vc.is_playing())

            # While turn 1 is still playing, enqueue a second utterance (backchannel)
            await sess._on_utterance(user, b"hmm")
            await self._wait_until(lambda: len(stt.heard_texts) >= 2, timeout=0.35)
            self.assertEqual(stt.heard_texts, ["pertanyaan pertama", "hmm"])
        finally:
            await sess.stop()

    async def test_backchannel_resumes_and_real_interruption_cuts_and_starts_new_turn(self) -> None:
        turns = {
            "cerita dong": [
                ("Satu dua tiga.", "happy", []),
                ("Empat lima enam.", "happy", []),
                ("Tujuh delapan sembilan.", "neutral", []),
            ],
            "eh ganti topik": [
                ("Oke ganti topik ya!", "surprised", []),
            ],
        }
        brain = FakeBrain(turns=turns, delay_s=0.04)
        sess, vc, _, _, _, _ = _make_session(brain=brain, tts=FakeTTS(frames=8))
        user = SimpleNamespace(id=1, name="Budi", display_name="Budi")
        try:
            # Part 1: Backchannel resumes without cutting
            await sess._on_utterance(user, b"cerita dong")
            await self._wait_until(lambda: vc.is_playing() and sess._current_source is not None)
            src = sess._current_source
            src.pause()
            await sess._on_utterance(user, b"hmm")
            await self._wait_until(lambda: not src.paused, timeout=0.35)
            await self._wait_until(lambda: len(vc.played_texts) == 3 and not vc.is_playing(), timeout=1.0)

            # Part 2: Real interruption cuts turn 1 and starts turn 2
            await sess._on_utterance(user, b"cerita dong")
            await self._wait_until(lambda: vc.is_playing())
            await asyncio.sleep(0.015)
            await sess._on_utterance(user, b"eh ganti topik")
            await self._wait_until(lambda: "Oke ganti topik ya!" in vc.played_texts, timeout=1.0)
            await self._wait_until(lambda: not vc.is_playing(), timeout=1.0)
            self.assertGreaterEqual(vc.stop_calls, 1)
        finally:
            await sess.stop()

    async def test_interrupted_turn_writes_one_row_and_never_touches_older_row(self) -> None:
        turns = {
            "cerita panjang": [
                ("Kalimat satu panjang.", "happy", []),
                ("Kalimat dua panjang.", "happy", []),
                ("Kalimat tiga panjang.", "neutral", []),
            ],
            "stop dulu": [
                ("Oke berhenti.", "neutral", []),
            ],
        }
        brain = FakeBrain(turns=turns, delay_s=0.05)
        memory = FakeMemory()
        older_id = memory.seed_row("Balasan lama")
        brain._last_voice_transcript_id = older_id  # simulate stale runner attribute if any

        sess, vc, _, _, _, _ = _make_session(brain=brain, tts=FakeTTS(frames=10), memory=memory)
        user = SimpleNamespace(id=1, name="Budi", display_name="Budi")
        try:
            await sess._on_utterance(user, b"cerita panjang")
            await self._wait_until(lambda: vc.is_playing())
            await asyncio.sleep(0.02)
            await sess._on_utterance(user, b"stop dulu")
            await self._wait_until(lambda: len(memory.rows) >= 3, timeout=1.0)

            self.assertEqual(memory.marked, [])
            self.assertEqual(memory.rows[0]["content"], "Balasan lama")
            self.assertEqual(memory.rows[0]["interrupted"], 0)
            self.assertIsNone(memory.rows[0]["heard_content"])

            interrupted_rows = [r for r in memory.rows[1:] if r["interrupted"] == 1]
            self.assertEqual(len(interrupted_rows), 1)
            row = interrupted_rows[0]
            self.assertTrue(row["heard_content"])
            self.assertIn("Kalimat", row["content"])
            self.assertNotEqual(row["heard_content"], "Kalimat satu panjang. Kalimat dua panjang. Kalimat tiga panjang.")
        finally:
            await sess.stop()

    async def test_normal_turn_writes_one_full_row(self) -> None:
        memory = FakeMemory()
        sess, vc, _, _, _, _ = _make_session(brain=FakeBrain(delay_s=0.01), tts=FakeTTS(frames=4), memory=memory)
        user = SimpleNamespace(id=1, name="Budi", display_name="Budi")
        try:
            await sess._on_utterance(user, b"halo")
            await self._wait_until(lambda: len(memory.rows) == 1, timeout=1.0)
            row = memory.rows[0]
            self.assertEqual(row["interrupted"], 0)
            self.assertIsNone(row["heard_content"])
            self.assertEqual(
                row["content"],
                "Kalimat pertama nih. Kalimat kedua lanjut. Kalimat ketiga selesai.",
            )
        finally:
            await sess.stop()

    async def test_speak_direct_text_writes_one_row_and_clears_stale_barge(self) -> None:
        memory = FakeMemory()
        sess, vc, _, _, _, _ = _make_session(brain=FakeBrain(delay_s=0.0), tts=FakeTTS(frames=3), memory=memory)
        sess._barge_triggered.set()  # stale flag from a prior barge-in
        try:
            ans = await sess.speak_direct_text("Budi", "1", "halo")
            self.assertEqual(ans, "Kalimat pertama nih. Kalimat kedua lanjut. Kalimat ketiga selesai.")
            self.assertEqual(len(vc.played_texts), 3)
            self.assertEqual(len(memory.rows), 1)
            self.assertEqual(memory.rows[0]["content"], ans)
            self.assertEqual(memory.rows[0]["interrupted"], 0)
        finally:
            await sess.stop()

    async def test_leftover_synth_tasks_cancelled_after_cut(self) -> None:
        tts = FakeTTS(frames=10, block_after_first=True)
        brain = FakeBrain(delay_s=0.0)
        sess, vc, _, _, _, _ = _make_session(brain=brain, tts=tts)
        user = SimpleNamespace(id=1, name="Budi", display_name="Budi")
        try:
            await sess._on_utterance(user, b"cerita")
            await self._wait_until(lambda: vc.is_playing() and len(tts.synth_tasks) >= 3, timeout=1.0)

            sess._abort_current_speech(reason="test_cut")
            await sess._cancel_active_turn()

            leftover = tts.synth_tasks[1:]
            self.assertTrue(leftover)
            for t in leftover:
                self.assertTrue(t.cancelled(), f"Expected synth task {t} to be cancelled")
        finally:
            await sess.stop()

    def test_smart_turn_sink_speech_activity_every_200ms(self) -> None:
        fired: list[int] = []
        fake_loop = SimpleNamespace(call_soon_threadsafe=lambda fn, *a: fn(*a))

        async def _noop(_u, _pcm):
            pass

        sink = SmartTurnSink(
            fake_loop,  # type: ignore[arg-type]
            _noop,
            rms_threshold=420,
            on_speech_activity=lambda _u, ms: fired.append(ms),
        )
        try:
            user = SimpleNamespace(id=1, bot=False)
            frame = (np.full(BYTES_PER_MS * 20 // 2, 2000, dtype=np.int16)).tobytes()
            for _ in range(50):  # 1000ms of 20ms voiced frames
                sink.write(frame, user)
        finally:
            sink.cleanup()

        self.assertEqual(fired, [200, 400, 600, 800, 1000])

    async def test_background_ducking_auto_unducks(self) -> None:
        sess, vc, _, _, _, _ = _make_session()
        try:
            sess.focus_user_id = 1
            vc._playing = True
            from voice.session import ExpressivePCMSource

            src = ExpressivePCMSource(bytes(FRAME_BYTES * 8), "halo")
            sess._current_source = src
            other = SimpleNamespace(id=2)

            sess._on_speech_activity(other, 400)
            self.assertAlmostEqual(src._target_gain, 0.28)
            first_handle = sess._unduck_handle
            self.assertIsNotNone(first_handle)

            # New activity from background speaker resets the timer
            sess._on_speech_activity(other, 600)
            self.assertTrue(first_handle.cancelled())
            self.assertIsNotNone(sess._unduck_handle)

            # Invoking the scheduled callback restores full volume
            sess._auto_unduck(src)
            self.assertAlmostEqual(src._target_gain, 1.0)
        finally:
            await sess.stop()

    async def test_memory_add_transcript_persists_interrupted_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "test.db")
            await store.init()
            try:
                mem = MemoryManager(store, EventBus())
                await mem.add_transcript(
                    room_id="guild:1",
                    speaker_id="self",
                    speaker_name="Pachinko",
                    role="assistant",
                    modality="voice",
                    content="Halo dunia ini panjang",
                    emotion="happy",
                    interrupted=1,
                    heard_content="Halo dunia",
                )
                rows = await mem.recent_transcript("guild:1", limit=5)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["interrupted"], 1)
                self.assertEqual(rows[0]["heard_content"], "Halo dunia")
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
