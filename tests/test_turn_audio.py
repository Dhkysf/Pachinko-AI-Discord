"""Unit tests for SmartTurnSink packet-gap padding and analyze_end_of_turn."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from speech.turn import BYTES_PER_MS, SAMPLE_RATE, SmartTurnSink, analyze_end_of_turn


def _pcm_frame(ms: int, amp: int) -> bytes:
    """Create 48kHz stereo int16 PCM bytes of `ms` duration at constant `amp`."""
    n_samples = (SAMPLE_RATE * ms) // 1000
    stereo = np.full((n_samples, 2), amp, dtype=np.int16)
    return stereo.tobytes()


def _make_sink():
    fake_loop = SimpleNamespace(call_soon_threadsafe=lambda fn, *a: fn(*a))

    async def _noop(_u, _pcm):
        pass

    return SmartTurnSink(fake_loop, _noop, rms_threshold=420)  # type: ignore[arg-type]


class TestTurnAudio(unittest.TestCase):
    def test_quiet_frames_after_speech_do_not_pad_synthetic_zeros(self) -> None:
        clock = [10.0]
        user = SimpleNamespace(id=1, bot=False)
        voiced = _pcm_frame(20, 2000)
        quiet = _pcm_frame(20, 20)

        with patch("speech.turn.time.monotonic", side_effect=lambda: clock[0]):
            sink = _make_sink()
            try:
                for _ in range(5):
                    clock[0] += 0.02
                    sink.write(voiced, user)
                for _ in range(10):
                    clock[0] += 0.02
                    sink.write(quiet, user)

                buf_len = len(sink._tracks[user.id].buf)
            finally:
                sink.cleanup()

        self.assertEqual(buf_len, 15 * 20 * BYTES_PER_MS)

    def test_real_200ms_packet_gap_is_padded_once(self) -> None:
        clock = [10.0]
        user = SimpleNamespace(id=1, bot=False)
        voiced = _pcm_frame(20, 2000)
        quiet = _pcm_frame(20, 20)

        with patch("speech.turn.time.monotonic", side_effect=lambda: clock[0]):
            sink = _make_sink()
            try:
                sink.write(voiced, user)
                len_after_voice = len(sink._tracks[user.id].buf)

                clock[0] = 10.20  # real 200ms packet gap
                sink.write(quiet, user)
                len_after_gap = len(sink._tracks[user.id].buf)

                clock[0] = 10.22
                sink.write(quiet, user)
                clock[0] = 10.24
                sink.write(quiet, user)
                len_after_more_quiet = len(sink._tracks[user.id].buf)
            finally:
                sink.cleanup()

        self.assertEqual(len_after_voice, 20 * BYTES_PER_MS)
        self.assertEqual(len_after_gap, (20 + 200 + 20) * BYTES_PER_MS)
        self.assertEqual(len_after_more_quiet, len_after_gap + 2 * 20 * BYTES_PER_MS)

    def test_analyze_end_of_turn_falling_vs_flat_with_trailing_silence(self) -> None:
        falling_voiced = _pcm_frame(300, 3000) + _pcm_frame(300, 1000)
        flat_voiced = _pcm_frame(600, 2000)

        for silence_ms in (500, 600):
            silence = _pcm_frame(silence_ms, 0)
            falling_score = analyze_end_of_turn(falling_voiced + silence)
            flat_score = analyze_end_of_turn(flat_voiced + silence)
            self.assertGreaterEqual(falling_score, 0.58, f"silence_ms={silence_ms}")
            self.assertLess(flat_score, falling_score, f"silence_ms={silence_ms}")
            self.assertLess(flat_score, 0.58, f"silence_ms={silence_ms}")


if __name__ == "__main__":
    unittest.main()
