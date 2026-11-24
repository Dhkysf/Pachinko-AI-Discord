"""Smart Prosodic Turn-Taking & Parked-Stream Audio Segmenter.

Why it is smarter than plain VAD:
1. **Parked Stream**: If a user pauses for 0.5s while thinking mid-sentence,
   the audio buffer is "parked" and joined seamlessly when they resume speaking (never split in two).
2. **Prosodic Endpointing**: Analyzes energy slope (whether intonation/volume drops naturally
   like the end of a sentence -> fast cut at `turn_fast_silence_ms`, or stays flat mid-thought
   -> waits patiently up to `turn_max_silence_ms`).
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np
from discord.sinks import Sink

SAMPLE_RATE = 48000
CHANNELS = 2
BYTES_PER_MS = (SAMPLE_RATE * CHANNELS * 2) // 1000  # 192 bytes/ms


def analyze_end_of_turn(pcm_48k_stereo: bytes, rms_threshold: float = 0.0) -> float:
    """Return confidence score (0.0 .. 1.0) that the speaker has finished talking.

    Score >= 0.55 means a natural sentence end (tail energy slopes down).
    Score < 0.55 means a likely mid-sentence thinking pause (energy stays high).
    """
    n_stereo = len(pcm_48k_stereo) // 4
    if n_stereo < SAMPLE_RATE // 2:  # < 0.5 seconds
        return 0.65
    samples = np.frombuffer(pcm_48k_stereo, dtype=np.int16, count=n_stereo * 2)
    mono = samples.reshape(-1, 2).mean(axis=1).astype(np.float32)

    # Trim leading/trailing near-silent frames so trailing silence does not skew the tail
    frame_len = SAMPLE_RATE // 50  # 20ms
    n_frames = mono.size // frame_len
    if n_frames > 0:
        frames = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        peak = float(frame_rms.max())
        if peak < 50.0:
            return 0.65
        floor = min(0.5 * peak, max(50.0, 0.10 * peak, float(rms_threshold)))
        voiced_idx = np.flatnonzero(frame_rms >= floor)
        if voiced_idx.size == 0:
            return 0.65
        mono = mono[int(voiced_idx[0]) * frame_len : (int(voiced_idx[-1]) + 1) * frame_len]
        if mono.size < SAMPLE_RATE // 2:
            return 0.65

    # Take the last 600ms of voiced audio, compare first half vs tail half
    tail_len = min(mono.size, int(0.6 * SAMPLE_RATE))
    tail = mono[-tail_len:]
    half = tail.size // 2
    rms_first = float(np.sqrt(np.mean(tail[:half] ** 2)) + 1e-6)
    rms_last = float(np.sqrt(np.mean(tail[half:] ** 2)) + 1e-6)
    drop_ratio = rms_last / rms_first

    # If tail drops (< 0.72x earlier energy), likely end of clause/sentence
    if drop_ratio < 0.72:
        return 0.82
    if drop_ratio < 0.90:
        return 0.60
    return 0.35


@dataclass
class _SpeakerTrack:
    user: object
    buf: bytearray = field(default_factory=bytearray)
    voiced_ms: int = 0
    last_voice_ts: float = 0.0
    last_packet_ts: float = 0.0
    speaking: bool = False
    last_notify_ms: int = 0


UtteranceHandler = Callable[[object, bytes], Awaitable[None]]
SpeechEventCallback = Callable[[object, int], None]  # (user, voiced_ms_so_far)


class SmartTurnSink(Sink):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_utterance: UtteranceHandler,
        *,
        rms_threshold: int = 420,
        fast_silence_ms: int = 550,
        max_silence_ms: int = 1350,
        min_speech_ms: int = 320,
        on_speech_activity: SpeechEventCallback | None = None,
    ) -> None:
        super().__init__()
        self.loop = loop
        self.on_utterance = on_utterance
        self.on_speech_activity = on_speech_activity
        self.rms_threshold = rms_threshold
        self.fast_silence_s = fast_silence_ms / 1000.0
        self.max_silence_s = max_silence_ms / 1000.0
        self.min_speech_ms = min_speech_ms
        self.preroll_bytes = 260 * BYTES_PER_MS
        self.max_utterance_bytes = 28_000 * BYTES_PER_MS

        self._tracks: dict[int, _SpeakerTrack] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._watcher = threading.Thread(target=self._clock_loop, daemon=True)
        self._watcher.start()

    def write(self, data, user) -> None:  # type: ignore[override]
        if user is None or getattr(user, "bot", False):
            return
        pcm = getattr(data, "pcm", data)
        if not pcm:
            return

        arr = np.frombuffer(pcm, dtype=np.int16)
        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) if arr.size else 0.0
        frame_ms = max(1, len(pcm) // BYTES_PER_MS)
        voiced = rms >= self.rms_threshold
        now = time.monotonic()
        notify_ms: int | None = None

        with self._lock:
            tr = self._tracks.get(user.id)
            if tr is None:
                tr = self._tracks[user.id] = _SpeakerTrack(user=user)

            # Insert short synthetic silence only on a real packet gap (Parked Stream)
            if tr.speaking and tr.last_packet_ts > 0:
                gap_s = now - tr.last_packet_ts
                if 0.08 < gap_s < self.max_silence_s:
                    pad_ms = min(300, round(gap_s * 1000))
                    tr.buf.extend(bytes(pad_ms * BYTES_PER_MS))
            tr.last_packet_ts = now

            tr.buf.extend(pcm)
            if voiced:
                tr.last_voice_ts = now
                tr.voiced_ms += frame_ms
                tr.speaking = True
                if tr.voiced_ms - tr.last_notify_ms >= 200:
                    tr.last_notify_ms = tr.voiced_ms
                    notify_ms = tr.voiced_ms
            elif not tr.speaking and len(tr.buf) > self.preroll_bytes:
                del tr.buf[: len(tr.buf) - self.preroll_bytes]

            if tr.speaking and len(tr.buf) >= self.max_utterance_bytes:
                self._emit_locked(tr)

        if notify_ms is not None and self.on_speech_activity:
            self.loop.call_soon_threadsafe(self.on_speech_activity, user, notify_ms)

    def _clock_loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            with self._lock:
                for tr in self._tracks.values():
                    if not tr.speaking:
                        continue
                    silence_s = now - tr.last_voice_ts
                    if silence_s < self.fast_silence_s:
                        continue
                    if silence_s >= self.max_silence_s:
                        self._emit_locked(tr)
                    else:
                        # Check prosody whether sentence is complete or still thinking
                        conf = analyze_end_of_turn(bytes(tr.buf), rms_threshold=self.rms_threshold)
                        if conf >= 0.58:
                            self._emit_locked(tr)
            time.sleep(0.04)

    def _emit_locked(self, tr: _SpeakerTrack) -> None:
        pcm = bytes(tr.buf)
        voiced_ms = tr.voiced_ms
        tr.buf.clear()
        tr.voiced_ms = 0
        tr.last_packet_ts = 0.0
        tr.speaking = False
        tr.last_notify_ms = 0
        if voiced_ms < self.min_speech_ms:
            return
        asyncio.run_coroutine_threadsafe(self.on_utterance(tr.user, pcm), self.loop)

    def cleanup(self) -> None:
        self._stop.set()
        self.finished = True
