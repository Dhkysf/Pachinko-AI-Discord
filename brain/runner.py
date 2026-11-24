"""SelfRunner: Unified Brain for Text & Voice (with Real-Time Sentence Streaming & Tool Loop)."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator

from brain.llm import LLMClient
from brain.profile import ProfileManager
from brain.prompt import assemble_messages
from brain.tools import TOOL_SCHEMAS, ToolBox
from core.bus import EventBus
from core.safety import filter_outgoing
from core.store import Store
from heart.engine import HeartEngine, extract_tags
from memory.manager import MemoryManager

log = logging.getLogger("brain.runner")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?~])\s+|\n+")
_URL_RE = re.compile(r"https?://\S+")
_PREFIX_STRIP_RE = re.compile(r"^\[[^\]]+\]:\s*")
_LLM_ERROR_REPLY = "Otakku lagi error, coba lagi bentar ya"


def _assistant_tool_turn(text: str | None, tool_calls: list) -> str:
    if text and text.strip():
        return text.strip()
    return f"[memanggil tool: {', '.join(tc.name for tc in tool_calls)}]"


def clean_for_speech(sentence: str) -> tuple[str, list[str]]:
    """Separate spoken text from URLs (URLs are sent to text chat as `aside`)."""
    urls = _URL_RE.findall(sentence)
    s = _URL_RE.sub(" tautan di chat ", sentence)
    s = _PREFIX_STRIP_RE.sub("", s)
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"[*_`#>~]", "", s)
    # Strip common kaomoji so TTS does not spell out parentheses/symbols
    s = re.sub(r"\([^a-zA-Z0-9]{2,12}\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, urls


@dataclass
class TurnResult:
    text: str
    emotion: str
    silent: bool
    transcript_id: int | None = None


class BrainRunner:
    def __init__(
        self,
        *,
        store: Store,
        bus: EventBus,
        profiles: ProfileManager,
        heart: HeartEngine,
        memory: MemoryManager,
        tools: ToolBox,
        llm: LLMClient,
    ) -> None:
        self.store = store
        self.bus = bus
        self.profiles = profiles
        self.heart = heart
        self.memory = memory
        self.tools = tools
        self.llm = llm
        self._room_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, room_id: str) -> asyncio.Lock:
        if room_id not in self._room_locks:
            self._room_locks[room_id] = asyncio.Lock()
        return self._room_locks[room_id]

    async def _prepare_turn(
        self,
        *,
        room_id: str,
        user_id: str,
        user_name: str,
        user_text: str,
        modality: str,
        wake_reason: str,
    ) -> list[dict]:
        prof = self.profiles.current
        person = await self.heart.touch_interaction(user_id, user_name, modality=modality)
        await self.memory.auto_capture_user_facts(user_id, user_name, user_text)

        notes = await self.memory.get_notes(f"person:{user_id}", limit=8)
        scopes = ("global", f"person:{user_id}", f"room:{room_id}")
        recalled = await self.memory.recall(user_text, scopes=scopes, limit=4)
        history = await self.memory.recent_transcript(room_id, limit=20)
        directives = await self.store.get_kv("owner_directives", "") or ""
        heart_state = self.heart.state_summary(prof.temperament)

        await self.memory.add_transcript(
            room_id=room_id,
            speaker_id=user_id,
            speaker_name=user_name,
            role="user",
            modality=modality,
            content=user_text,
        )

        return assemble_messages(
            profile=prof,
            directives=directives,
            heart_state=heart_state,
            person=person,
            person_notes=notes,
            recalled_memories=recalled,
            history=history,
            speaker_name=user_name,
            user_text=user_text,
            modality=modality,
            wake_reason=wake_reason,
        )

    async def run_text_turn(
        self,
        *,
        room_id: str,
        user_id: str,
        user_name: str,
        user_text: str,
        wake_reason: str = "direct_mention",
    ) -> TurnResult:
        async with self._lock_for(room_id):
            messages = await self._prepare_turn(
                room_id=room_id,
                user_id=user_id,
                user_name=user_name,
                user_text=user_text,
                modality="text",
                wake_reason=wake_reason,
            )

            # At most 1 tool-call round, then one final answer without tools
            try:
                resp = await self.llm.complete_with_tools(messages, tools=TOOL_SCHEMAS, fast_voice=False)
                if resp.tool_calls:
                    messages.append({"role": "assistant", "content": _assistant_tool_turn(resp.text, resp.tool_calls)})
                    for tc in resp.tool_calls:
                        await self.bus.emit("brain.tool_call", tool=tc.name, args=tc.arguments, user=user_name)
                        out = await self.tools.execute(
                            tc.name, tc.arguments, room_id=room_id, user_id=user_id, user_name=user_name
                        )
                        messages.append({"role": "user", "content": f"[Tool Result `{tc.name}`]:\n{out}"})
                    resp = await self.llm.complete_with_tools(messages, tools=None, fast_voice=False)
            except Exception:
                log.exception("LLM call failed in text turn")
                return TurnResult(text=_LLM_ERROR_REPLY, emotion="neutral", silent=False)

            raw = resp.text or ""
            if "<silence/>" in raw or not raw.strip():
                await self.bus.emit("brain.decision", room_id=room_id, action="silence", reason=wake_reason)
                return TurnResult(text="", emotion="neutral", silent=True)

            clean, explicit_emo, feel_raw = extract_tags(raw)
            clean = _PREFIX_STRIP_RE.sub("", filter_outgoing(clean)).strip()
            prof = self.profiles.current
            emotion = await self.heart.apply_feel_block(
                feel_raw, explicit_emo, user_id, user_name, prof.temperament
            )

            if not clean:
                return TurnResult(text="", emotion=emotion, silent=True)

            tid = await self.memory.add_transcript(
                room_id=room_id,
                speaker_id="self",
                speaker_name=prof.display_name,
                role="assistant",
                modality="text",
                content=clean,
                emotion=emotion,
            )
            return TurnResult(text=clean, emotion=emotion, silent=False, transcript_id=tid)

    async def stream_voice_turn(
        self,
        *,
        room_id: str,
        user_id: str,
        user_name: str,
        user_text: str,
        wake_reason: str = "voice_utterance",
    ) -> AsyncIterator[tuple[str, str, list[str]]]:
        """Yield `(spoken_sentence, emotion, aside_urls)` as soon as a sentence finishes streaming from the LLM."""
        async with self._lock_for(room_id):
            messages = await self._prepare_turn(
                room_id=room_id,
                user_id=user_id,
                user_name=user_name,
                user_text=user_text,
                modality="voice",
                wake_reason=wake_reason,
            )
            prof = self.profiles.current

            # Quick check if the user asked for an explicit action (reminder / web search / song)
            low = user_text.lower()
            need_tools = any(
                kw in low
                for kw in ("ingetin", "ingatkan", "timer", "cariin", "googling", "berita", "nyanyiin", "putar lagu", "request lagu")
            )
            if need_tools and self.llm.supports_tools():
                try:
                    first = await self.llm.complete_with_tools(messages, tools=TOOL_SCHEMAS, fast_voice=True)
                except Exception:
                    log.exception("LLM tool pre-check failed in voice turn")
                    return
                if first.tool_calls:
                    messages.append({"role": "assistant", "content": _assistant_tool_turn(first.text, first.tool_calls)})
                    for tc in first.tool_calls:
                        await self.bus.emit("brain.tool_call", tool=tc.name, args=tc.arguments, user=user_name)
                        out = await self.tools.execute(
                            tc.name, tc.arguments, room_id=room_id, user_id=user_id, user_name=user_name
                        )
                        messages.append({"role": "user", "content": f"[Tool Result `{tc.name}`]:\n{out}"})

            buffer = ""
            full_raw = ""
            current_emo = self.heart.dominant_emotion(prof.temperament)
            in_feel = False

            try:
                async for tok in self.llm.stream_text(messages, fast_voice=True):
                    full_raw += tok
                    if "<silence/>" in full_raw:
                        return
                    if "<feel" in buffer + tok:
                        in_feel = True
                    if in_feel:
                        buffer += tok
                        continue

                    buffer += tok
                    # Check for leading <emo:...> tag
                    emo_m = re.search(r"<emo:([a-z_]+)>", buffer, re.I)
                    if emo_m:
                        current_emo = emo_m.group(1).lower()
                        buffer = re.sub(r"<emo:[a-z_]+>", "", buffer, flags=re.I)

                    # Split per sentence as soon as a period/question/exclamation mark followed by space appears
                    parts = _SENTENCE_SPLIT_RE.split(buffer)
                    if len(parts) > 1:
                        for ready in parts[:-1]:
                            spoken, urls = clean_for_speech(filter_outgoing(ready))
                            if len(spoken) >= 2:
                                yield spoken, current_emo, urls
                        buffer = parts[-1]
            except Exception:
                log.exception("LLM stream failed in voice turn")
                return

            # Remaining buffer after stream ends
            clean_tail, explicit_emo, feel_raw = extract_tags(full_raw)
            emotion = await self.heart.apply_feel_block(
                feel_raw, explicit_emo or current_emo, user_id, user_name, prof.temperament
            )

            tail_rem, _, _ = extract_tags(buffer)
            spoken_tail, urls_tail = clean_for_speech(filter_outgoing(tail_rem))
            if len(spoken_tail) >= 2:
                yield spoken_tail, emotion, urls_tail
