"""Safe Tool Registry callable by the Character Brain (Web Search, Memory Notes, Reminders, Songs)."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from typing import Any, Callable, Awaitable

import httpx

from brain.gate import ActionGate
from config import config
from core.safety import data_block, redact_secrets
from core.store import normalize_person_name
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
                    "target": {
                        "type": "string",
                        "description": "'user' for the speaker, the person's name for someone else, or a topic (game, band, place)",
                    },
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
_SELF_TARGETS: frozenset[str] = frozenset({"", "user", "orang", "aku", "saya", "gue"})

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
        *,
        http_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.memory = memory
        self.gate = gate
        self.song_enqueue_cb = song_enqueue_cb
        self._http_transport = http_transport

    async def _resolve_note_target(
        self,
        raw_target: Any,
        *,
        user_id: str,
        user_name: str,
        username: str = "",
    ) -> tuple[str, str, str]:
        """Resolve remember_note target to (doc_key, memory_uid, memory_label)."""
        cleaned = normalize_person_name("" if raw_target is None else str(raw_target))
        uid_str = str(user_id)
        if cleaned in _SELF_TARGETS:
            return f"person:{uid_str}", uid_str, user_name

        matches: dict[str, str] = {}
        if cleaned in (normalize_person_name(user_name), normalize_person_name(username)):
            matches[uid_str] = user_name

        store = getattr(self.memory, "store", None)
        if store is not None:
            cur = await store.conn.execute("SELECT user_id, display_name, nickname FROM heart_person")
            for row in await cur.fetchall():
                if cleaned in (
                    normalize_person_name(row["display_name"]),
                    normalize_person_name(row["nickname"]),
                ):
                    row_uid = str(row["user_id"])
                    label = user_name if row_uid == uid_str else str(row["display_name"] or user_name)
                    matches.setdefault(row_uid, label)

        if len(matches) == 1:
            resolved_uid, resolved_name = next(iter(matches.items()))
            return f"person:{resolved_uid}", resolved_uid, resolved_name

        return f"topic:{cleaned[:24]}", uid_str, user_name

    async def execute(
        self,
        name: str,
        raw_args: dict[str, Any] | str,
        *,
        room_id: str,
        user_id: str,
        user_name: str,
        username: str = "",
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
            doc_key, mem_uid, mem_label = await self._resolve_note_target(
                args.get("target", "user"),
                user_id=str(user_id),
                user_name=user_name,
                username=username,
            )
            await self.memory.upsert_note(doc_key, fact, evidence=f"From chat with {user_name}")
            await self.memory.remember(f"{mem_label}: {fact}", scope=f"person:{mem_uid}", subject_id=mem_uid, importance=0.8)
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

    async def _ninerouter_search(self, query: str) -> list[str] | None:
        url = config.ninerouter_base_url.rstrip("/") + "/search"
        headers = (
            {"Authorization": f"Bearer {config.ninerouter_api_key}"}
            if config.ninerouter_api_key
            else {}
        )
        payload: dict[str, Any] = {
            "query": query,
            "max_results": 4,
            "country": "id",
            "language": "id",
        }
        if config.web_search_ninerouter_provider:
            payload["provider"] = config.web_search_ninerouter_provider

        try:
            async with httpx.AsyncClient(timeout=10.0, transport=self._http_transport) as client:
                resp = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as e:
            log.warning("9router web_search failed (%s); falling back to ddgs.", redact_secrets(str(e))[:160])
            return None

        if resp.status_code != 200:
            detail = resp.reason_phrase or "error"
            try:
                err_body = resp.json()
                if isinstance(err_body, dict):
                    err = err_body.get("error")
                    if isinstance(err, dict) and err.get("message"):
                        detail = str(err["message"])
                    elif isinstance(err, str) and err:
                        detail = err
            except Exception:
                pass
            log.warning(
                "9router web_search HTTP %s (%s); falling back to ddgs.",
                resp.status_code,
                redact_secrets(detail)[:160],
            )
            return None

        try:
            data = resp.json()
        except Exception:
            log.warning("9router web_search returned invalid JSON; falling back to ddgs.")
            return None

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            log.warning("9router web_search returned no results list; falling back to ddgs.")
            return None

        lines: list[str] = []
        for it in results:
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or "").strip()
            snippet = str(it.get("snippet") or it.get("content") or it.get("body") or "").strip()
            href = str(it.get("url") or it.get("link") or it.get("href") or "").strip()
            if not (title or snippet or href):
                continue
            lines.append(f"- {title}: {snippet} ({href})")
            if len(lines) >= 4:
                break

        if not lines:
            log.warning("9router web_search returned no results; falling back to ddgs.")
            return None
        return lines

    async def _web_search(self, query: str) -> str:
        if not query.strip():
            return "Empty query."

        if config.web_search_provider == "ninerouter":
            lines = await self._ninerouter_search(query)
            if lines:
                return data_block("web_search", "\n".join(lines), title=query, max_chars=3500)

        try:
            items = await asyncio.to_thread(_ddgs_search, query)
        except Exception as e:
            return f"Search failed: {e}"
        if not items:
            return "No results found."
        lines = [f"- {it.get('title', '')}: {it.get('body', '')} ({it.get('href', '')})" for it in items]
        return data_block("web_search", "\n".join(lines), title=query, max_chars=3500)


def _ddgs_search(query: str) -> list[dict[str, Any]]:
    from ddgs import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=4))
