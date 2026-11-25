"""SelfRunner: Unified Brain for Text & Voice (with Real-Time Sentence Streaming & Tool Loop)."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import AsyncIterator

from brain.llm import LLMClient
from brain.profile import ProfileManager
from brain.prompt import assemble_messages, detect_register, normalize_register, requested_language
from brain.tools import TOOL_SCHEMAS, ToolBox
from config import config
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

_REMINDER_VERB_RE = re.compile(
    r"\b(?:ingetin|ingatkan|ingatin|remind|(?:pasang(?:in|kan)?|set|bikin(?:in)?|buat(?:in|kan)?)\s+(?:alarm|timer|pengingat))\b",
    re.IGNORECASE,
)
_REMINDER_NEG_OR_PAST_RE = re.compile(
    r"\b(?:jangan|(?:nggak|gak|ngga|ga)\s+usah|tadi|kemarin|barusan|s?udah\s+lewat)\b",
    re.IGNORECASE,
)
_NUM_WORDS = {
    "tiga puluh": 30.0,
    "lima belas": 15.0,
    "sepuluh": 10.0,
    "sembilan": 9.0,
    "delapan": 8.0,
    "tujuh": 7.0,
    "enam": 6.0,
    "lima": 5.0,
    "empat": 4.0,
    "tiga": 3.0,
    "dua": 2.0,
    "satu": 1.0,
}
_UNIT_MULT = {"detik": 1.0 / 60.0, "menit": 1.0, "jam": 60.0}
_NUM_WORDS_PAT = r"(?:tiga\s+puluh|lima\s+belas|sepuluh|sembilan|delapan|tujuh|enam|lima|empat|tiga|dua|satu)"
_TIME_ATOM_PAT = (
    rf"(?:\bsejam\b"
    rf"|\bsemenit\b"
    rf"|(?:\b\d+(?:[.,]\d+)?|\b{_NUM_WORDS_PAT}\b)\s*(?:detik|menit|jam)\b)"
)
_TIME_ATOM_RE = re.compile(_TIME_ATOM_PAT, re.IGNORECASE)
_TIME_EXPR_RE = re.compile(
    r"(?:\b(?:dalam|setelah)\s+)?"
    rf"(?P<expr>\bsetengah\s+jam\b|{_TIME_ATOM_PAT}(?:\s+(?:dan\s+)?{_TIME_ATOM_PAT})*(?:\s+setengah\b)?)"
    r"(?:\s+(?:lagi|dari\s+sekarang))?",
    re.IGNORECASE,
)
_REMINDER_FILLER_RE = re.compile(
    r"\b(?:aku|gue|saya|me|buat|untuk|biar|dong|ya|yah|tolong|please|pls|nih|deh)\b",
    re.IGNORECASE,
)


def _parse_time_expr(expr: str) -> float | None:
    norm = re.sub(r"\s+", " ", expr.strip().lower())
    if norm == "setengah jam":
        return 30.0
    total = 0.0
    last_unit: str | None = None
    for m in _TIME_ATOM_RE.finditer(norm):
        atom = m.group(0)
        if atom == "sejam":
            total += 60.0
            last_unit = "jam"
        elif atom == "semenit":
            total += 1.0
            last_unit = "menit"
        else:
            m_num = re.match(r"^(\d+(?:[.,]\d+)?|[a-z ]+?)\s*(detik|menit|jam)$", atom)
            if not m_num:
                return None
            raw_amt, unit = m_num.group(1).strip(), m_num.group(2)
            if raw_amt in _NUM_WORDS:
                amt = _NUM_WORDS[raw_amt]
            else:
                try:
                    amt = float(raw_amt.replace(",", "."))
                except ValueError:
                    return None
            total += amt * _UNIT_MULT[unit]
            last_unit = unit
    if last_unit is None:
        return None
    if norm.endswith(" setengah"):
        total += 0.5 * _UNIT_MULT[last_unit]
    return total if total > 0 else None


def parse_reminder_request(text: str) -> tuple[float, str] | None:
    """Parse clear Indonesian reminder requests into `(minutes, note)` or `None`."""
    if not text or not _REMINDER_VERB_RE.search(text):
        return None
    tm = _TIME_EXPR_RE.search(text)
    if not tm:
        return None
    if _REMINDER_NEG_OR_PAST_RE.search(text[:tm.start()]):
        return None

    minutes = _parse_time_expr(tm.group("expr"))
    if minutes is None:
        return None

    cleaned = _REMINDER_VERB_RE.sub(" ", text[:tm.start()] + " " + text[tm.end():])
    cleaned = _REMINDER_FILLER_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
    note = re.sub(r"\s+", " ", cleaned).strip() or "Pengingat!"
    return round(minutes, 2), note


def _assistant_tool_turn(text: str | None, tool_calls: list) -> str:
    if text and text.strip():
        return text.strip()
    return f"[memanggil tool: {', '.join(tc.name for tc in tool_calls)}]"


def _has_usable_text(raw: str) -> bool:
    if not raw.strip() or "<silence/>" in raw:
        return False
    clean, _, _ = extract_tags(raw)
    return bool(_PREFIX_STRIP_RE.sub("", filter_outgoing(clean)).strip())


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

    async def _maybe_reminder_fallback(
        self,
        *,
        user_text: str,
        called_tools: set[str],
        room_id: str,
        user_id: str,
        user_name: str,
        wake_reason: str = "",
        reply_text: str = "",
    ) -> bool:
        if wake_reason == "ambient_glance" or not _has_usable_text(reply_text):
            return False
        if "set_reminder" in called_tools:
            return False
        parsed = parse_reminder_request(user_text)
        if parsed is None:
            return False
        minutes, note = parsed
        args = {"minutes": minutes, "note": note}
        await self.bus.emit("brain.tool_call", tool="set_reminder", args=args, user=user_name)
        await self.tools.execute("set_reminder", args, room_id=room_id, user_id=user_id, user_name=user_name)
        log.info("reminder created by fallback (minutes=%s, user=%s)", minutes, user_name)
        return True

    async def _prepare_turn(
        self,
        *,
        room_id: str,
        user_id: str,
        user_name: str,
        user_text: str,
        modality: str,
        wake_reason: str,
    ) -> tuple[list[dict], str | None]:
        prof = self.profiles.current
        person = await self.heart.touch_interaction(user_id, user_name, modality=modality)
        await self.memory.auto_capture_user_facts(user_id, user_name, user_text)

        notes = await self.memory.get_notes(f"person:{user_id}", limit=8)
        scopes = ("global", f"person:{user_id}", f"room:{room_id}")
        recalled = await self.memory.recall(user_text, scopes=scopes, limit=4)
        history = await self.memory.recent_transcript(room_id, limit=max(1, config.max_history_turns))
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

        register = (
            None
            if requested_language(user_text)
            else detect_register(user_text, [t["content"] for t in history if t.get("role") == "user"])
        )
        messages = assemble_messages(
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
        return messages, register

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
            messages, register = await self._prepare_turn(
                room_id=room_id,
                user_id=user_id,
                user_name=user_name,
                user_text=user_text,
                modality="text",
                wake_reason=wake_reason,
            )

            # At most 1 tool-call round, then one final answer without tools
            called_tools: set[str] = set()
            is_ambient = wake_reason == "ambient_glance"
            try:
                resp = await self.llm.complete_with_tools(messages, tools=TOOL_SCHEMAS, fast_voice=False)
                if resp.tool_calls:
                    messages.append({"role": "assistant", "content": _assistant_tool_turn(resp.text, resp.tool_calls)})
                    for tc in resp.tool_calls:
                        called_tools.add(tc.name)
                        await self.bus.emit("brain.tool_call", tool=tc.name, args=tc.arguments, user=user_name)
                        out = await self.tools.execute(
                            tc.name, tc.arguments, room_id=room_id, user_id=user_id, user_name=user_name
                        )
                        messages.append({"role": "user", "content": f"[Tool Result `{tc.name}`]:\n{out}"})
                    resp = await self.llm.complete_with_tools(messages, tools=None, fast_voice=False)
            except Exception:
                log.exception("LLM call failed in text turn")
                return TurnResult(text=_LLM_ERROR_REPLY, emotion="neutral", silent=False)

            if not is_ambient and not _has_usable_text(resp.text or ""):
                provider = getattr(resp, "provider", "") or getattr(
                    self.llm, "active_provider", lambda: config.llm_provider
                )()
                model = getattr(resp, "model", "") or config.active_model()
                finish_reason = getattr(resp, "finish_reason", None)
                has_reasoning = bool(getattr(resp, "has_reasoning", False))
                log.warning(
                    "Empty/silent direct reply (wake_reason=%s, provider=%s, model=%s, finish_reason=%s, has_reasoning=%s); retrying with 2x max_tokens",
                    wake_reason,
                    provider,
                    model,
                    finish_reason,
                    has_reasoning,
                )
                try:
                    resp = await self.llm.complete_with_tools(
                        messages, tools=None, fast_voice=False, max_tokens_scale=2
                    )
                except Exception:
                    log.exception("LLM retry failed in text turn")
                    return TurnResult(text=_LLM_ERROR_REPLY, emotion="neutral", silent=False)

            raw = resp.text or ""
            if not _has_usable_text(raw):
                if is_ambient:
                    await self.bus.emit("brain.decision", room_id=room_id, action="silence", reason=wake_reason)
                    return TurnResult(text="", emotion="neutral", silent=True)
                return TurnResult(text=_LLM_ERROR_REPLY, emotion="neutral", silent=False)

            clean, explicit_emo, feel_raw = extract_tags(raw)
            clean = _PREFIX_STRIP_RE.sub("", filter_outgoing(clean)).strip()
            if register is not None:
                clean = normalize_register(clean, register)
            prof = self.profiles.current
            emotion = await self.heart.apply_feel_block(
                feel_raw, explicit_emo, user_id, user_name, prof.temperament
            )

            if not clean:
                return TurnResult(text="", emotion=emotion, silent=True)

            await self._maybe_reminder_fallback(
                user_text=user_text,
                called_tools=called_tools,
                room_id=room_id,
                user_id=user_id,
                user_name=user_name,
                wake_reason=wake_reason,
                reply_text=raw,
            )

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
            messages, register = await self._prepare_turn(
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
                for kw in (
                    "ingetin", "ingatkan", "ingatin", "remind", "timer", "alarm",
                    "cariin", "googling", "berita", "nyanyiin", "putar lagu", "request lagu",
                )
            )
            called_tools: set[str] = set()
            if need_tools and self.llm.supports_tools():
                try:
                    first = await self.llm.complete_with_tools(messages, tools=TOOL_SCHEMAS, fast_voice=True)
                except Exception:
                    log.exception("LLM tool pre-check failed in voice turn")
                    return
                if first.tool_calls:
                    messages.append({"role": "assistant", "content": _assistant_tool_turn(first.text, first.tool_calls)})
                    for tc in first.tool_calls:
                        called_tools.add(tc.name)
                        await self.bus.emit("brain.tool_call", tool=tc.name, args=tc.arguments, user=user_name)
                        out = await self.tools.execute(
                            tc.name, tc.arguments, room_id=room_id, user_id=user_id, user_name=user_name
                        )
                        messages.append({"role": "user", "content": f"[Tool Result `{tc.name}`]:\n{out}"})
                elif first.text and first.text.strip():
                    if "<silence/>" in first.text:
                        return
                    await self._maybe_reminder_fallback(
                        user_text=user_text,
                        called_tools=called_tools,
                        room_id=room_id,
                        user_id=user_id,
                        user_name=user_name,
                        wake_reason=wake_reason,
                        reply_text=first.text,
                    )
                    clean, explicit_emo, feel_raw = extract_tags(first.text)
                    emotion = await self.heart.apply_feel_block(
                        feel_raw,
                        explicit_emo or self.heart.dominant_emotion(prof.temperament),
                        user_id,
                        user_name,
                        prof.temperament,
                    )
                    for part in _SENTENCE_SPLIT_RE.split(clean):
                        filtered = filter_outgoing(part)
                        if register is not None:
                            filtered = normalize_register(filtered, register)
                        spoken, urls = clean_for_speech(filtered)
                        if len(spoken) >= 2:
                            yield spoken, emotion, urls
                    return

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
                            filtered = filter_outgoing(ready)
                            if register is not None:
                                filtered = normalize_register(filtered, register)
                            spoken, urls = clean_for_speech(filtered)
                            if len(spoken) >= 2:
                                yield spoken, current_emo, urls
                        buffer = parts[-1]
            except Exception:
                log.exception("LLM stream failed in voice turn")
                return

            # Remaining buffer after stream ends
            await self._maybe_reminder_fallback(
                user_text=user_text,
                called_tools=called_tools,
                room_id=room_id,
                user_id=user_id,
                user_name=user_name,
                wake_reason=wake_reason,
                reply_text=full_raw,
            )
            clean_tail, explicit_emo, feel_raw = extract_tags(full_raw)
            emotion = await self.heart.apply_feel_block(
                feel_raw, explicit_emo or current_emo, user_id, user_name, prof.temperament
            )

            tail_rem, _, _ = extract_tags(buffer)
            filtered_tail = filter_outgoing(tail_rem)
            if register is not None:
                filtered_tail = normalize_register(filtered_tail, register)
            spoken_tail, urls_tail = clean_for_speech(filtered_tail)
            if len(spoken_tail) >= 2:
                yield spoken_tail, emotion, urls_tail
