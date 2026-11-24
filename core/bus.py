"""EventBus asinkron untuk menghubungkan semua modul secara longgar (decoupled).

Semua kejadian penting (chat, voice, emosi berubah, bond naik, keputusan otak)
lewat di sini dan otomatis disiarkan ke Web Panel.
"""
from __future__ import annotations

import asyncio
import fnmatch
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable


@dataclass(slots=True)
class Event:
    id: str
    topic: str
    ts: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Subscriber = Callable[[Event], Awaitable[None] | None]


class EventBus:
    def __init__(self, max_recent: int = 300) -> None:
        self._recent: deque[Event] = deque(maxlen=max_recent)
        self._handlers: list[tuple[str, Subscriber]] = []
        self._streams: set[asyncio.Queue[Event]] = set()
        self._persist_hook: Callable[[Event], Awaitable[None]] | None = None

    def set_persist_hook(self, hook: Callable[[Event], Awaitable[None]]) -> None:
        self._persist_hook = hook

    def on(self, pattern: str, handler: Subscriber) -> Callable[[], None]:
        entry = (pattern, handler)
        self._handlers.append(entry)
        return lambda: self._handlers.remove(entry) if entry in self._handlers else None

    def subscribe_stream(self, maxsize: int = 200) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._streams.add(q)
        return q

    def unsubscribe_stream(self, q: asyncio.Queue[Event]) -> None:
        self._streams.discard(q)

    def recent(self, pattern: str = "*", limit: int = 80) -> list[dict[str, Any]]:
        matched = [e.to_dict() for e in self._recent if fnmatch.fnmatch(e.topic, pattern)]
        return matched[-limit:]

    async def emit(self, topic: str, **payload: Any) -> Event:
        ev = Event(id=uuid.uuid4().hex[:16], topic=topic, ts=time.time(), payload=payload)
        self._recent.append(ev)

        if self._persist_hook:
            try:
                await self._persist_hook(ev)
            except Exception:
                pass

        for q in list(self._streams):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass

        for pattern, fn in list(self._handlers):
            if fnmatch.fnmatch(topic, pattern):
                try:
                    res = fn(ev)
                    if asyncio.iscoroutine(res):
                        asyncio.create_task(res)
                except Exception:
                    pass
        return ev
