"""Pachinko Natural Voice Session (Sentence Streaming + TTS Prefetch + Filler + Two-Stage Smart Barge-In).

Key features:
1. `ExpressivePCMSource`: Plays raw 48kHz stereo PCM in 20ms frames with:
   - Smooth volume ducking (`0.28x`) when someone else speaks in the background.
   - Instant Pause & Resume when the active speaker talks (for short backchannels like "hmm/oke", resumes seamlessly without losing words!).
   - Tracks the percentage of the sentence already heard (`heard_ratio`) if truly interrupted (barge-in).
2. `LinePrefetch`: Sentence 1 starts speaking as soon as it leaves the LLM while sentences 2 & 3 synthesize in the background.
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time

import discord
import numpy as np

from brain.profile import ProfileManager
from brain.runner import BrainRunner
from brain.triage import TriageEngine
from config import config
from core.bus import EventBus
from memory.manager import MemoryManager
from speech.stt import STTEngine
from speech.tts import TTSEngine
from speech.turn import SmartTurnSink

log = logging.getLogger("voice.session")

FRAME_BYTES = 3840  # 20ms @ 48kHz stereo int16


class ExpressivePCMSource(discord.AudioSource):
    """PCM AudioSource supporting Ducking, seamless Pause/Resume, & heard-word ratio."""

    def __init__(self, pcm_data: bytes, text: str) -> None:
        # Pad to a multiple of 3840 bytes
        rem = len(pcm_data) % FRAME_BYTES
        if rem:
            pcm_data += bytes(FRAME_BYTES - rem)
        self.pcm = pcm_data
        self.text = text
        self.pos = 0
        self.paused = False
        self.stopped = False
        self.gain = 1.0
        self._target_gain = 1.0

    def set_duck(self, ducked: bool) -> None:
        self._target_gain = 0.28 if ducked else 1.0

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def stop_now(self) -> None:
        self.stopped = True

    @property
    def heard_ratio(self) -> float:
        if not self.pcm:
            return 1.0
        return min(1.0, self.pos / len(self.pcm))

    def heard_text_estimate(self) -> str:
        words = self.text.split()
        if not words:
            return ""
        n = max(1, round(len(words) * self.heard_ratio))
        return " ".join(words[:n])

    def read(self) -> bytes:
        if self.stopped or self.pos >= len(self.pcm):
            return b""

        if self.paused:
            return bytes(FRAME_BYTES)

        chunk = self.pcm[self.pos : self.pos + FRAME_BYTES]
        self.pos += FRAME_BYTES

        # Smooth gain ramp
        if abs(self.gain - self._target_gain) > 0.02:
            self.gain += 0.15 * (self._target_gain - self.gain)
        else:
            self.gain = self._target_gain

        if abs(self.gain - 1.0) > 0.01:
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            scaled = np.clip(samples * self.gain, -32768, 32767).astype(np.int16)
            chunk = scaled.tobytes()

        return chunk

    def is_opus(self) -> bool:
        return False


class NaturalVoiceSession:
    def __init__(
        self,
        *,
        guild_id: int,
        vc: discord.VoiceClient,
        text_channel: discord.abc.Messageable | None,
        bus: EventBus,
        profiles: ProfileManager,
        triage: TriageEngine,
        brain: BrainRunner,
        memory: MemoryManager,
        stt: STTEngine,
        tts: TTSEngine,
    ) -> None:
        self.guild_id = guild_id
        self.vc = vc
        self.text_channel = text_channel
        self.bus = bus
        self.profiles = profiles
        self.triage = triage
        self.brain = brain
        self.memory = memory
        self.stt = stt
        self.tts = tts

        self.loop = asyncio.get_running_loop()
        self.room_id = f"guild:{guild_id}"
        self.focus_user_id: int | None = None
        self.singing_active: bool = False

        self._utterance_q: asyncio.Queue[tuple[discord.abc.User, bytes]] = asyncio.Queue()
        self._current_source: ExpressivePCMSource | None = None
        self._active_turn_task: asyncio.Task | None = None
        self._spoken_so_far: list[str] = []
        self._barge_triggered = asyncio.Event()

        self.sink = SmartTurnSink(
            self.loop,
            self._on_utterance,
            rms_threshold=config.vad_rms_threshold,
            fast_silence_ms=config.turn_fast_silence_ms,
            max_silence_ms=config.turn_max_silence_ms,
            min_speech_ms=config.min_speech_ms,
            on_speech_activity=self._on_speech_activity,
        )
        self._worker_task = asyncio.create_task(self._worker_loop())

    def start(self) -> None:
        self.vc.start_recording(self.sink)

    async def stop(self) -> None:
        self._worker_task.cancel()
        if self._active_turn_task and not self._active_turn_task.done():
            self._active_turn_task.cancel()
        try:
            if self.vc.is_recording():
                self.vc.stop_recording()
        except Exception:
            pass
        self.sink.cleanup()
        if self.vc.is_connected():
            await self.vc.disconnect(force=True)

    # ---------------- Stage 1 Smart Barge-In (When VAD detects speech) ----------------
    def _on_speech_activity(self, user, voiced_ms: int) -> None:
        if not config.allow_barge_in or self.singing_active:
            return
        src = self._current_source
        if src is None or not self.vc.is_playing():
            return

        is_focus = self.focus_user_id is None or user.id == self.focus_user_id
        if not is_focus:
            # Someone else talking in the background: lower volume (ducking) without cutting off
            if voiced_ms >= 240:
                src.set_duck(True)
            return

        # Primary speaker is talking: pause briefly at 320ms, hard-cut if long speech >= 2200ms
        if voiced_ms >= 2200:
            self._abort_current_speech(reason="long_focus_speech")
        elif voiced_ms >= 320 and not src.paused:
            src.pause()
            # If no meaningful transcript arrives within 2.2s (e.g. a cough), auto-resume
            self.loop.call_later(2.2, self._auto_resume_if_still_paused, src)

    def _auto_resume_if_still_paused(self, src: ExpressivePCMSource) -> None:
        if self._current_source is src and src.paused and not src.stopped:
            src.resume()
            src.set_duck(False)

    def _abort_current_speech(self, reason: str = "barge_in") -> None:
        self._barge_triggered.set()
        src = self._current_source
        if src and not src.stopped:
            partial = src.heard_text_estimate()
            if partial:
                self._spoken_so_far.append(partial)
            src.stop_now()
        if self.vc.is_playing():
            self.vc.stop()
        log.info("Bot speech stopped (%s)", reason)

    # ---------------- Stage 2 Smart Barge-In & Utterance Queue ----------------
    async def _on_utterance(self, user, pcm: bytes) -> None:
        await self._utterance_q.put((user, pcm))

    def _is_backchannel(self, text: str) -> bool:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        if not words or len(words) > 2:
            return False
        prof = self.profiles.current
        return all(w in prof.backchannel_words for w in words)

    async def _worker_loop(self) -> None:
        while True:
            user, pcm = await self._utterance_q.get()
            try:
                await self._process_utterance(user, pcm)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Failed to process voice utterance from %s", user)

    async def _process_utterance(self, user, pcm: bytes) -> None:
        prof = self.profiles.current
        members = [m.display_name for m in (self.vc.channel.members if self.vc.channel else []) if not m.bot]
        hotwords = ", ".join([prof.display_name, *prof.nicknames, *members])

        t0 = time.perf_counter()
        raw_text = await self.stt.transcribe(pcm, hotwords=hotwords)
        stt_s = time.perf_counter() - t0

        src = self._current_source
        if not raw_text:
            # Empty noise: if paused/ducked earlier, resume smoothly!
            if src:
                src.resume()
                src.set_duck(False)
            return

        text = self.triage.normalize_wake_words(raw_text, prof)
        speaker_name = getattr(user, "display_name", None) or user.name
        addressed = self.triage.is_addressed(text, prof)

        # While singing (karaoke), ignore chatter unless called by name clearly
        if self.singing_active and not addressed:
            return

        # Stage 2 Smart Barge-In: check if the utterance was just a backchannel ("hmm", "iya", "oke")
        if src and self.vc.is_playing():
            is_focus = self.focus_user_id is None or user.id == self.focus_user_id
            if self._is_backchannel(text) and not addressed:
                log.info("[%s] backchannel '%s' -> resume speaking without cutting off.", speaker_name, text)
                src.resume()
                src.set_duck(False)
                return
            if not is_focus and not addressed:
                src.set_duck(False)
                return
            # Genuine interruption!
            self._abort_current_speech(reason=f"interrupted_by:{speaker_name}")
            if self._active_turn_task and not self._active_turn_task.done():
                self._active_turn_task.cancel()
                try:
                    await self._active_turn_task
                except asyncio.CancelledError:
                    pass

        log.info("🎙️ [%s] (STT %.2fs): %s", speaker_name, stt_s, text)
        await self.bus.emit("voice.heard", speaker=speaker_name, text=text, stt_sec=round(stt_s, 2))

        self.focus_user_id = user.id
        self._barge_triggered.clear()
        self._spoken_so_far.clear()

        self._active_turn_task = asyncio.create_task(
            self._speak_streaming_turn(str(user.id), speaker_name, text)
        )
        try:
            await self._active_turn_task
        except asyncio.CancelledError:
            pass

    async def _speak_streaming_turn(self, user_id: str, speaker_name: str, user_text: str) -> None:
        prof = self.profiles.current
        audio_q: asyncio.Queue[tuple[str, str, asyncio.Task[bytes]] | None] = asyncio.Queue()
        first_sentence_ready = asyncio.Event()

        # Auto-filler ("Hmm?", "Eh, bentar...") if LLM/tool thinks longer than `filler_delay_ms`
        async def _filler_watch() -> None:
            if not config.filler_enabled or not prof.filler_phrases:
                return
            await asyncio.sleep(config.filler_delay_ms / 1000.0)
            if not first_sentence_ready.is_set() and not self._barge_triggered.is_set() and not self.vc.is_playing():
                phrase = random.choice(prof.filler_phrases)
                pcm = await self.tts.get_filler_pcm(phrase)
                if pcm and not first_sentence_ready.is_set() and not self._barge_triggered.is_set():
                    await self.play_raw_pcm(pcm, phrase, is_filler=True)

        filler_task = asyncio.create_task(_filler_watch())
        aside_urls_all: list[str] = []
        full_sentences: list[str] = []

        async def _producer() -> None:
            try:
                async for sentence, emo, urls in self.brain.stream_voice_turn(
                    room_id=self.room_id,
                    user_id=user_id,
                    user_name=speaker_name,
                    user_text=user_text,
                ):
                    if self._barge_triggered.is_set():
                        break
                    aside_urls_all.extend(urls)
                    full_sentences.append(sentence)
                    # Start TTS synthesis in background immediately (LinePrefetch)!
                    synth_task = asyncio.create_task(self.tts.synthesize_pcm(sentence, emotion=emo))
                    await audio_q.put((sentence, emo, synth_task))
            finally:
                await audio_q.put(None)

        prod_task = asyncio.create_task(_producer())

        try:
            while not self._barge_triggered.is_set():
                item = await audio_q.get()
                if item is None:
                    break
                sentence, emo, synth_task = item
                pcm = await synth_task
                first_sentence_ready.set()
                if not pcm or self._barge_triggered.is_set():
                    break

                self.stt.echo_guard.record_bot_speech(sentence)
                completed = await self.play_raw_pcm(pcm, sentence)
                if completed:
                    self._spoken_so_far.append(sentence)
                else:
                    break
        finally:
            filler_task.cancel()
            if not prod_task.done():
                prod_task.cancel()

        # If interrupted mid-way, update transcript with only the words actually heard
        if self._barge_triggered.is_set() and getattr(self.brain, "_last_voice_transcript_id", None):
            heard = " ".join(self._spoken_so_far).strip()
            await self.memory.mark_interrupted(self.brain._last_voice_transcript_id, heard)

        if full_sentences and config.log_voice_to_text and self.text_channel:
            shown = " ".join(self._spoken_so_far) if self._barge_triggered.is_set() else " ".join(full_sentences)
            cut_tag = " *(dipotong)*" if self._barge_triggered.is_set() else ""
            msg = f"🎙️ **{speaker_name}:** {user_text}\n🤖 **{prof.display_name}:** {shown}{cut_tag}"
            if aside_urls_all:
                msg += "\n🔗 " + " ".join(aside_urls_all)
            asyncio.create_task(self._safe_send(msg))

    async def play_raw_pcm(self, pcm: bytes, text: str, *, is_filler: bool = False) -> bool:
        if not self.vc.is_connected() or not pcm:
            return False

        while self.vc.is_playing():
            if self._barge_triggered.is_set():
                return False
            await asyncio.sleep(0.03)

        src = ExpressivePCMSource(pcm, text)
        if not is_filler:
            self._current_source = src
        done = asyncio.Event()
        self.vc.play(src, after=lambda _e: self.loop.call_soon_threadsafe(done.set))
        await done.wait()
        if self._current_source is src:
            self._current_source = None
        return not src.stopped

    async def speak_direct_text(self, speaker_name: str, user_id: str, prompt_text: str) -> str:
        """Used by `/say`: full response + played to the voice channel."""
        collected: list[str] = []
        async for sentence, emo, _ in self.brain.stream_voice_turn(
            room_id=self.room_id,
            user_id=user_id,
            user_name=speaker_name,
            user_text=prompt_text,
            wake_reason="slash_say",
        ):
            collected.append(sentence)
            pcm = await self.tts.synthesize_pcm(sentence, emotion=emo)
            self.stt.echo_guard.record_bot_speech(sentence)
            await self.play_raw_pcm(pcm, sentence)
        return " ".join(collected)

    async def _safe_send(self, content: str) -> None:
        try:
            if self.text_channel:
                await self.text_channel.send(content[:1990])
        except Exception:
            pass
