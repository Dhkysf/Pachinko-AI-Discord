"""Daftar Tool Aman yang bisa dipanggil oleh Otak Karakter (Web Search, Baca Web, Catat Memori, Pengingat, Lagu)."""
from __future__ import annotations

import asyncio
import html
import json
import re
import time
from typing import Any, Callable, Awaitable

import httpx

from brain.gate import ActionGate
from core.safety import data_block
from memory.manager import MemoryManager

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Cari informasi terbaru di internet (berita, fakta, game, lirik, cuaca, dll).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Kata kunci pencarian"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_note",
            "description": "Simpan fakta penting tentang lawan bicara atau topik ke Buku Catatan jangka panjang.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "Fakta ringkas yang perlu diingat"},
                    "target": {"type": "string", "description": "'user' untuk lawan bicara, atau nama topik"},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Pasang pengingat waktu / timer untuk user di channel ini.",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "number", "description": "Berapa menit dari sekarang"},
                    "note": {"type": "string", "description": "Pesan yang harus diingatkan"},
                },
                "required": ["minutes", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "queue_song",
            "description": "Masukkan lagu dari YouTube / judul lagu ke antrean nyanyi / karaoke di voice channel.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Judul lagu atau URL YouTube"}},
                "required": ["query"],
            },
        },
    },
]

VOICE_TOOL_NAMES = {"web_search", "remember_note", "set_reminder", "queue_song"}


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

    def set_song_callback(self, cb: Callable[[str, str, str, str], Awaitable[str]]) -> None:
        self.song_enqueue_cb = cb

    async def execute(
        self,
        name: str,
        raw_args: dict[str, Any] | str,
        *,
        room_id: str,
        user_id: str,
        user_name: str,
    ) -> str:
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception:
                args = {"query": raw_args}
        else:
            args = raw_args or {}

        allowed, reason = await self.gate.check(
            name, args, requester_id=str(user_id), requester_name=user_name
        )
        if not allowed:
            return f"[Gate menolak: {reason}]"

        if name == "web_search":
            return await self._web_search(str(args.get("query", "")))
        if name == "web_read":
            return await self._web_read(str(args.get("url", "")))
        if name == "remember_note":
            fact = str(args.get("fact", "")).strip()
            target = str(args.get("target", "user")).strip().lower()
            doc_key = f"person:{user_id}" if target in ("user", "orang", "") else f"topic:{target[:24]}"
            await self.memory.upsert_note(doc_key, fact, evidence=f"Dari obrolan dengan {user_name}")
            await self.memory.remember(f"{user_name}: {fact}", scope=f"person:{user_id}", subject_id=str(user_id), importance=0.8)
            return f"Tersimpan di catatan ({doc_key}): {fact}"
        if name == "set_reminder":
            mins = max(0.2, min(1440.0, float(args.get("minutes", 5))))
            note = str(args.get("note", "Pengingat!")).strip()
            due = time.time() + mins * 60.0
            await self.memory.add_schedule(
                kind="reminder", room_id=room_id, user_id=user_id, user_name=user_name, note=note, due_at=due
            )
            return f"Pengingat dipasang untuk {mins:g} menit lagi: '{note}'"
        if name == "queue_song":
            query = str(args.get("query", "")).strip()
            if not self.song_enqueue_cb:
                return "Modul lagu belum terhubung ke sesi voice."
            return await self.song_enqueue_cb(room_id, str(user_id), user_name, query)

        return f"Tool '{name}' tidak dikenal."

    async def _web_search(self, query: str) -> str:
        if not query.strip():
            return "Query kosong."

        def _search_sync() -> list[dict[str, Any]]:
            from ddgs import DDGS
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=4))

        try:
            items = await asyncio.to_thread(_search_sync)
        except Exception as e:
            return f"Pencarian gagal: {e}"
        if not items:
            return "Tidak ada hasil ditemukan."
        lines = [f"- {it.get('title', '')}: {it.get('body', '')} ({it.get('href', '')})" for it in items]
        return data_block("web_search", "\n".join(lines), title=query, max_chars=3500)

    async def _web_read(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return "URL tidak valid."
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": "Mozilla/5.0 MiyoriaReader/1.0"})
                raw = re.sub(r"<(script|style).*?>.*?</\1>", " ", r.text, flags=re.S | re.I)
                text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
                text = re.sub(r"\s+", " ", text).strip()
                return data_block("web_page", text, title=url[:60], max_chars=5000)
        except Exception as e:
            return f"Gagal membaca halaman: {e}"
