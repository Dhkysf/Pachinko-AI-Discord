"""Safe Tool Registry callable by the Character Brain (Web Search, Memory Notes, Reminders, Songs)."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from typing import Any, Callable, Awaitable

from brain.gate import ActionGate
from core.safety import data_block
from memory.manager import MemoryManager

log = logging.getLogger("brain.tools")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search for up-to-date information on the internet (news, facts, games, lyrics, weather, etc.).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search keywords"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_note",
            "description": (
                "Save a lasting fact that the user stated about themselves or explicitly asked to remember "
                "(likes, dislikes, name/nickname, school/work, important plans). Never save that someone asked "
                "a question, said hello, or sent a one-word message. Write the fact in short Indonesian, "
                "third person, e.g. 'Suka kopi susu gula aren'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "Short lasting fact in Indonesian, third person, e.g. 'Suka kopi susu gula aren'",
                    },
                    "target": {"type": "string", "description": "'user' for the speaker, or a topic name"},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a timed reminder / timer for the user in this channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "number", "description": "Minutes from now"},
                    "note": {"type": "string", "description": "Message to remind about"},
                },
                "required": ["minutes", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "queue_song",
            "description": "Add a YouTube song / song title to the singing / karaoke queue in the voice channel.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Song title or YouTube URL"}},
                "required": ["query"],
            },
        },
    },
]

_ALLOWED_TOOLS: frozenset[str] = frozenset(t["function"]["name"] for t in TOOL_SCHEMAS)

_NOTE_EVENT_LOG_RE = re.compile(
    r"\b(?:asked|said|bertanya|nanya|bilang|menyapa|baru\s+kenal)\b",
    re.IGNORECASE,
)
_NOTE_SCORE_RE = re.compile(
    r"\b(?:bond|trust|warmth|score)\s*[:=]?\s*[+-]?\d+(?:[.,]\d+)?(?:/\d+)?",
    re.IGNORECASE,
)


def _reject_note_reason(fact: str) -> str | None:
    if len(fact) < 8:
        return "too short"
    if _NOTE_EVENT_LOG_RE.search(fact):
        return "event log, not a lasting fact"
    if _NOTE_SCORE_RE.search(fact):
        return "contains relationship/score numbers"
    return None


class ToolBox:
    def __init__(
        self,
        memory: MemoryManager,
        gate: ActionGate,
        song_enqueue_cb: Callable[[str, str, str, str], Awaitable[str]] | None = None,
    ) -> None:
        self.memory = memory
        self.gate = gate
        self.song_enqueue_cb = song_enqueue_cb

    async def execute(
        self,
        name: str,
        raw_args: dict[str, Any] | str,
        *,
        room_id: str,
        user_id: str,
        user_name: str,
    ) -> str:
        if name not in _ALLOWED_TOOLS:
            log.warning("Rejected unadvertised tool call: %r (user=%s)", name, user_name)
            return f"Unknown tool '{name}'."

        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:
                args = {"query": raw_args}
        else:
            args = raw_args or {}
        if not isinstance(args, dict):
            args = {}

        allowed, reason = await self.gate.check(
            name, args, requester_id=str(user_id), requester_name=user_name
        )
        if not allowed:
            return f"[Gate denied: {reason}]"

        if name == "web_search":
            return await self._web_search(str(args.get("query", "")))
        if name == "remember_note":
            fact = str(args.get("fact", "")).strip()
            reject = _reject_note_reason(fact)
            if reject:
                log.info("Rejected remember_note (%s) for user=%s: %r", reject, user_name, fact)
                return f"Skipped note: {reject}."
            target = str(args.get("target", "user")).strip().lower()
            doc_key = f"person:{user_id}" if target in ("user", "orang", "") else f"topic:{target[:24]}"
            await self.memory.upsert_note(doc_key, fact, evidence=f"From chat with {user_name}")
            await self.memory.remember(f"{user_name}: {fact}", scope=f"person:{user_id}", subject_id=str(user_id), importance=0.8)
            return f"Saved to notebook ({doc_key}): {fact}"
        if name == "set_reminder":
            raw_mins = args.get("minutes", 5)
            if isinstance(raw_mins, bool):
                return f"Invalid minutes value: {raw_mins!r}. Provide a finite number."
            try:
                mins_val = float(raw_mins)
            except (TypeError, ValueError):
                return f"Invalid minutes value: {raw_mins!r}. Provide a finite number."
            if not math.isfinite(mins_val):
                return f"Invalid minutes value: {raw_mins!r}. Provide a finite number."
            mins = max(0.2, min(1440.0, mins_val))
            note = str(args.get("note", "Pengingat!")).strip()
            due = time.time() + mins * 60.0
            await self.memory.add_schedule(
                kind="reminder", room_id=room_id, user_id=user_id, user_name=user_name, note=note, due_at=due
            )
            return f"Reminder set for {mins:g} minutes from now: '{note}'"
        if name == "queue_song":
            query = str(args.get("query", "")).strip()
            if not self.song_enqueue_cb:
                return "Song module is not connected to a voice session."
            return await self.song_enqueue_cb(room_id, str(user_id), user_name, query)

        return f"Unknown tool '{name}'."

    async def _web_search(self, query: str) -> str:
        if not query.strip():
            return "Empty query."

        def _search_sync() -> list[dict[str, Any]]:
            from ddgs import DDGS
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=4))

        try:
            items = await asyncio.to_thread(_search_sync)
        except Exception as e:
            return f"Search failed: {e}"
        if not items:
            return "No results found."
        lines = [f"- {it.get('title', '')}: {it.get('body', '')} ({it.get('href', '')})" for it in items]
        return data_block("web_search", "\n".join(lines), title=query, max_chars=3500)
