"""Single SQLite storage (WAL mode) for all memories, emotions, coins, & settings."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

LEGACY_DB_NAME = "miyoria.db"
_log = logging.getLogger("core.store")


def migrate_legacy_db(db_path: Path, default_dir: Path) -> bool:
    """Rename legacy data/miyoria.db (plus -wal/-shm) to data/pachinko.db once if needed."""
    target = default_dir / "pachinko.db"
    legacy = default_dir / LEGACY_DB_NAME
    if db_path.resolve() != target.resolve() or target.exists() or not legacy.exists():
        return False

    moved: list[tuple[Path, Path]] = []
    try:
        for suffix in ("", "-wal", "-shm"):
            src = Path(f"{legacy}{suffix}")
            if src.exists():
                dst = Path(f"{target}{suffix}")
                src.replace(dst)
                moved.append((src, dst))
    except OSError:
        for src, dst in reversed(moved):
            if dst.exists() and not src.exists():
                dst.replace(src)
        raise

    _log.info("Migrated legacy database %s -> %s", legacy, target)
    return True

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- L0: Append-only Event & Decision Log
CREATE TABLE IF NOT EXISTS events_log (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    ts REAL NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events_log(ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_topic ON events_log(topic, ts DESC);

-- L1: Persistent Transcripts (Text & Voice unified per room/channel)
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    speaker_name TEXT NOT NULL,
    role TEXT NOT NULL,            -- 'user' | 'assistant' | 'system_note'
    modality TEXT NOT NULL,        -- 'text' | 'voice'
    content TEXT NOT NULL,
    heard_content TEXT,            -- if interrupted (barge-in), only the words heard so far
    interrupted INTEGER DEFAULT 0,
    emotion TEXT DEFAULT 'neutral',
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transcripts_room ON transcripts(room_id, id DESC);

-- L2: Episodic & Semantic Memory + FTS5
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,           -- 'global' | 'guild:<id>' | 'person:<id>'
    subject_id TEXT DEFAULT '',
    content TEXT NOT NULL,
    importance REAL DEFAULT 0.5,
    vec_json TEXT DEFAULT '[]',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope, created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

-- L3: Living Notebook (live notes per person, topic, and self-reflection)
CREATE TABLE IF NOT EXISTS notebook (
    doc_key TEXT NOT NULL,         -- 'person:<id>' | 'topic:<slug>' | 'self'
    line_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    evidence TEXT DEFAULT '',
    active INTEGER DEFAULT 1,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notebook_doc ON notebook(doc_key, active);

-- Heart: Bot internal emotions
CREATE TABLE IF NOT EXISTS heart_self (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    emotions_json TEXT NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    energy REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Heart: 9-Axis Relationship per User
CREATE TABLE IF NOT EXISTS heart_person (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    nickname TEXT DEFAULT '',
    bond REAL DEFAULT 0.0,         -- -100 .. 1000
    trust REAL DEFAULT 10.0,       -- 0 .. 100
    familiarity REAL DEFAULT 5.0,  -- 0 .. 100
    warmth REAL DEFAULT 0.1,       -- -1 .. 1
    respect REAL DEFAULT 0.1,      -- -1 .. 1
    ease REAL DEFAULT 0.2,         -- 0 .. 1
    grudge REAL DEFAULT 0.0,       -- 0 .. 1
    longing REAL DEFAULT 0.0,      -- 0 .. 1
    rapport REAL DEFAULT 0.1,      -- 0 .. 1
    impression TEXT DEFAULT 'Baru kenal, belum banyak ngobrol.',
    opt_out_approach INTEGER DEFAULT 0,
    daily_date TEXT DEFAULT '',
    daily_counts_json TEXT DEFAULT '{}',
    interactions INTEGER DEFAULT 0,
    last_seen REAL NOT NULL
);

-- Schedules & Reminders
CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,            -- 'reminder' | 'self'
    room_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    note TEXT NOT NULL,
    due_at REAL NOT NULL,
    done INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_schedules_due ON schedules(done, due_at);

-- Economy: Coin Wallets & Double-Entry Ledger + Card Collection
CREATE TABLE IF NOT EXISTS economy_wallets (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    balance INTEGER DEFAULT 100,
    streak INTEGER DEFAULT 0,
    last_daily_date TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS economy_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    delta INTEGER NOT NULL,
    reason TEXT NOT NULL,
    ts REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    rarity TEXT NOT NULL,
    title TEXT NOT NULL,
    mood_tag TEXT NOT NULL,
    quote TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- Per-channel settings & key-value runtime overrides (for Web Panel)
CREATE TABLE IF NOT EXISTS channel_modes (
    channel_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kv_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Store is not initialized (call await store.init() first)")
        return self._conn

    @asynccontextmanager
    async def write_tx(self) -> AsyncIterator[aiosqlite.Connection]:
        async with self._write_lock:
            conn = self.conn
            try:
                yield conn
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def write(self, sql: str, params: Iterable[Any] = ()) -> aiosqlite.Cursor:
        async with self.write_tx() as conn:
            return await conn.execute(sql, params)

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()
        await self.prune_old_events()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def prune_old_events(self, max_age_days: int = 7) -> int:
        cutoff = time.time() - max_age_days * 86400
        cur = await self.write("DELETE FROM events_log WHERE ts < ?", (cutoff,))
        return cur.rowcount

    async def get_kv(self, key: str, default: Any = None) -> Any:
        cur = await self.conn.execute("SELECT value_json FROM kv_settings WHERE key = ?", (key,))
        row = await cur.fetchone()
        return json.loads(row["value_json"]) if row else default

    async def set_kv(self, key: str, value: Any) -> None:
        await self.write(
            "INSERT INTO kv_settings (key, value_json, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at",
            (key, json.dumps(value, ensure_ascii=False), time.time()),
        )

    async def get_channel_mode(self, channel_id: str, default_mode: str) -> str:
        cur = await self.conn.execute("SELECT mode FROM channel_modes WHERE channel_id = ?", (str(channel_id),))
        row = await cur.fetchone()
        return row["mode"] if row else default_mode

    async def set_channel_mode(self, channel_id: str, mode: str) -> None:
        await self.write(
            "INSERT INTO channel_modes (channel_id, mode, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET mode = excluded.mode, updated_at = excluded.updated_at",
            (str(channel_id), mode, time.time()),
        )
