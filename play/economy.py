"""Play & Economy: Coin Ledger (`koin`), Daily Check-in + Fortune, SVG Collectible Card Gacha, & Party Games."""
from __future__ import annotations

import datetime as dt
import html
import random
import time
from typing import Any

from core.bus import EventBus
from core.store import Store
from heart.engine import HeartEngine

FORTUNES = [
    ("Keberuntungan Besar ✨", "Hari ini energi kode dan hoki gacha lo lagi tajam banget."),
    ("Angin Hangat 🌤️", "Cocok buat nyantai sambil dengerin lagu favorit bareng."),
    ("Tenang Fokus ☕", "Seduh kopi dulu, kerjaan berat bakal beres satu per satu."),
    ("Hoki Tipis 🍀", "Jangan begadang kelewatan, simpen tenaga buat malam minggu."),
]

CARD_POOL = [
    ("UR", "Midnight Radeon Overclock", "smug", "16 GB VRAM penuh cuma buat nemenin lo ngobrol jam tiga pagi."),
    ("SSR", "Kopi Gula Aren Dingin", "happy", "Satu gelas buat lo, dua gelas buat gue. Adil kan?"),
    ("SSR", "Hujan di Balik Headset", "shy", "Jangan dengerin detak suaraku terlalu dekat dong..."),
    ("SR", "Sesi Karaoke Dadakan", "happy", "Mic udah nyala, dilarang kabur sebelum nyanyi satu lagu!"),
    ("SR", "Debugger Tengah Malam", "thinking", "Titik koma ketinggalan satu baris, nyarinya dua jam."),
    ("R", "Mode Nyimak Santai", "neutral", "Gue dengerin kok dari pojokan channel."),
    ("R", "Tidur Lima Menit Lagi", "annoyed", "Alarm kelima baru bangun, klasik banget."),
]

TRIVIA_BANK = [
    ("Bahasa pemrograman apa yang namanya terinspirasi dari grup komedi Monty Python?", ["python"], 60),
    ("Arsitektur GPU AMD seri Radeon RX 9070 menggunakan generasi RDNA ke berapa?", ["rdna 4", "rdna4", "4", "empat"], 80),
    ("Protokol enkripsi end-to-end voice milik Discord yang wajib sejak 2026 bernama apa?", ["dave"], 90),
    ("Alat musik tradisional dari Jawa Barat yang terbuat dari bambu dan digoyangkan adalah?", ["angklung"], 50),
    ("Planet apa yang dijuluki Planet Merah di tata surya kita?", ["mars"], 45),
]


def render_card_svg(card: dict[str, Any], owner_name: str, char_name: str) -> str:
    """Render a sharp SVG collectible card with zero external dependencies."""
    colors = {
        "UR": ("#f59e0b", "#78350f"),
        "SSR": ("#14b8a6", "#134e4a"),
        "SR": ("#38bdf8", "#0c4a6e"),
        "R": ("#94a3b8", "#1e293b"),
    }
    accent, bg = colors.get(card["rarity"], ("#94a3b8", "#1e293b"))
    title = html.escape(card["title"])
    quote = html.escape(card["quote"])
    owner = html.escape(owner_name)
    cname = html.escape(char_name)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="520" height="300" viewBox="0 0 520 300">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="{bg}"/>
    </linearGradient>
  </defs>
  <rect width="520" height="300" rx="18" fill="url(#g)" stroke="{accent}" stroke-width="3"/>
  <rect x="20" y="20" width="68" height="32" rx="8" fill="{accent}"/>
  <text x="54" y="42" font-family="Segoe UI, sans-serif" font-size="16" font-weight="bold" fill="#0f172a" text-anchor="middle">{card["rarity"]}</text>
  <text x="102" y="42" font-family="Segoe UI, sans-serif" font-size="14" fill="#cbd5e1">{cname} Collection</text>
  <text x="24" y="110" font-family="Segoe UI, sans-serif" font-size="24" font-weight="bold" fill="#f8fafc">{title}</text>
  <text x="24" y="158" font-family="Segoe UI, sans-serif" font-size="15" font-style="italic" fill="#e2e8f0">"{quote}"</text>
  <line x1="24" y1="235" x2="496" y2="235" stroke="#334155" stroke-width="1"/>
  <text x="24" y="268" font-family="Segoe UI, sans-serif" font-size="13" fill="#94a3b8">Pemilik: {owner}</text>
  <text x="496" y="268" font-family="Segoe UI, sans-serif" font-size="13" fill="{accent}" text-anchor="end">Mood: {card["mood_tag"]}</text>
</svg>"""


class EconomyAndPlay:
    def __init__(self, store: Store, bus: EventBus, heart: HeartEngine) -> None:
        self.store = store
        self.bus = bus
        self.heart = heart
        self.active_trivia: dict[str, tuple[str, list[str], int]] = {}  # room_id -> (question, answers, prize)

    async def get_wallet(self, user_id: str, display_name: str = "User") -> dict[str, Any]:
        uid = str(user_id)
        cur = await self.store.conn.execute("SELECT * FROM economy_wallets WHERE user_id = ?", (uid,))
        row = await cur.fetchone()
        if not row:
            await self.store.write(
                "INSERT OR IGNORE INTO economy_wallets (user_id, display_name, balance) VALUES (?, ?, 100)",
                (uid, display_name),
            )
            cur = await self.store.conn.execute("SELECT * FROM economy_wallets WHERE user_id = ?", (uid,))
            row = await cur.fetchone()
        return dict(row)

    async def add_coins(self, user_id: str, display_name: str, delta: int, reason: str) -> int:
        uid = str(user_id)
        async with self.store.write_tx() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO economy_wallets (user_id, display_name, balance) VALUES (?, ?, 100)",
                (uid, display_name),
            )
            await conn.execute(
                "UPDATE economy_wallets SET display_name = ?, balance = MAX(0, balance + ?) WHERE user_id = ?",
                (display_name, delta, uid),
            )
            await conn.execute(
                "INSERT INTO economy_ledger (user_id, delta, reason, ts) VALUES (?, ?, ?, ?)",
                (uid, delta, reason, time.time()),
            )
            cur = await conn.execute("SELECT balance FROM economy_wallets WHERE user_id = ?", (uid,))
            new_bal = int((await cur.fetchone())["balance"])
        await self.bus.emit("economy.tx", user=display_name, delta=delta, balance=new_bal, reason=reason)
        return new_bal

    async def daily_checkin(self, user_id: str, display_name: str) -> dict[str, Any]:
        uid = str(user_id)
        today = dt.date.today()
        today_s = today.isoformat()
        yesterday_s = (today - dt.timedelta(days=1)).isoformat()

        async with self.store.write_tx() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO economy_wallets (user_id, display_name, balance) VALUES (?, ?, 100)",
                (uid, display_name),
            )
            cur = await conn.execute("SELECT * FROM economy_wallets WHERE user_id = ?", (uid,))
            w = dict(await cur.fetchone())
            streak = (int(w["streak"]) + 1) if w["last_daily_date"] == yesterday_s else 1
            reward = 80 + min(120, streak * 15)
            upd = await conn.execute(
                "UPDATE economy_wallets "
                "SET display_name = ?, balance = MAX(0, balance + ?), streak = ?, last_daily_date = ? "
                "WHERE user_id = ? AND (last_daily_date IS NULL OR last_daily_date != ?)",
                (display_name, reward, streak, today_s, uid, today_s),
            )
            if upd.rowcount == 0:
                return {"ok": False, "message": "Lo udah ambil `/daily` hari ini! Balik lagi besok ya.", "wallet": w}
            await conn.execute(
                "INSERT INTO economy_ledger (user_id, delta, reason, ts) VALUES (?, ?, ?, ?)",
                (uid, reward, f"daily_streak_{streak}", time.time()),
            )
            cur = await conn.execute("SELECT balance FROM economy_wallets WHERE user_id = ?", (uid,))
            new_bal = int((await cur.fetchone())["balance"])

        fortune_title, fortune_desc = random.choice(FORTUNES)
        await self.bus.emit("economy.tx", user=display_name, delta=reward, balance=new_bal, reason=f"daily_streak_{streak}")
        await self.heart.touch_interaction(user_id, display_name, modality="text")
        return {
            "ok": True,
            "reward": reward,
            "streak": streak,
            "balance": new_bal,
            "fortune_title": fortune_title,
            "fortune_desc": fortune_desc,
        }

    async def draw_card(self, user_id: str, display_name: str, cost: int = 60) -> dict[str, Any]:
        uid = str(user_id)
        weights = [4, 14, 14, 24, 24, 35, 35]
        rarity, title, mood_tag, quote = random.choices(CARD_POOL, weights=weights, k=1)[0]

        async with self.store.write_tx() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO economy_wallets (user_id, display_name, balance) VALUES (?, ?, 100)",
                (uid, display_name),
            )
            upd = await conn.execute(
                "UPDATE economy_wallets SET display_name = ?, balance = balance - ? WHERE user_id = ? AND balance >= ?",
                (display_name, cost, uid, cost),
            )
            if upd.rowcount == 0:
                cur = await conn.execute("SELECT balance FROM economy_wallets WHERE user_id = ?", (uid,))
                bal = int((await cur.fetchone())["balance"])
                return {"ok": False, "message": f"Koin kurang! Butuh **{cost} koin**, saldo lo **{bal} koin**."}
            now = time.time()
            await conn.execute(
                "INSERT INTO economy_ledger (user_id, delta, reason, ts) VALUES (?, ?, ?, ?)",
                (uid, -cost, "gacha_card", now),
            )
            await conn.execute(
                "INSERT INTO collection_cards (user_id, rarity, title, mood_tag, quote, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (uid, rarity, title, mood_tag, quote, now),
            )
            cur = await conn.execute("SELECT balance FROM economy_wallets WHERE user_id = ?", (uid,))
            new_bal = int((await cur.fetchone())["balance"])

        await self.bus.emit("economy.tx", user=display_name, delta=-cost, balance=new_bal, reason="gacha_card")
        return {
            "ok": True,
            "card": {"rarity": rarity, "title": title, "mood_tag": mood_tag, "quote": quote},
            "balance": new_bal,
        }

    async def gift_to_character(self, user_id: str, display_name: str, amount: int) -> dict[str, Any]:
        if amount <= 0:
            return {"ok": False, "message": "Jumlah koin harus lebih dari 0."}
        uid = str(user_id)

        async with self.store.write_tx() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO economy_wallets (user_id, display_name, balance) VALUES (?, ?, 100)",
                (uid, display_name),
            )
            upd = await conn.execute(
                "UPDATE economy_wallets SET display_name = ?, balance = balance - ? WHERE user_id = ? AND balance >= ?",
                (display_name, amount, uid, amount),
            )
            if upd.rowcount == 0:
                cur = await conn.execute("SELECT balance FROM economy_wallets WHERE user_id = ?", (uid,))
                bal = int((await cur.fetchone())["balance"])
                return {"ok": False, "message": f"Saldo lo cuma **{bal} koin**."}
            await conn.execute(
                "INSERT INTO economy_ledger (user_id, delta, reason, ts) VALUES (?, ?, ?, ?)",
                (uid, -amount, "gift_character", time.time()),
            )
            cur = await conn.execute("SELECT balance FROM economy_wallets WHERE user_id = ?", (uid,))
            new_bal = int((await cur.fetchone())["balance"])

        await self.bus.emit("economy.tx", user=display_name, delta=-amount, balance=new_bal, reason="gift_character")
        person = await self.heart.add_gift_bonus(user_id, display_name, amount)
        return {"ok": True, "balance": new_bal, "person": person}

    def start_trivia(self, room_id: str) -> tuple[str, int]:
        q, ans, prize = random.choice(TRIVIA_BANK)
        self.active_trivia[str(room_id)] = (q, ans, prize)
        return q, prize

    async def check_trivia_answer(self, room_id: str, user_id: str, display_name: str, text: str) -> dict[str, Any] | None:
        active = self.active_trivia.get(str(room_id))
        if not active:
            return None
        _, valid_answers, prize = active
        low = text.lower().strip()
        if any(ans in low for ans in valid_answers):
            self.active_trivia.pop(str(room_id), None)
            bal = await self.add_coins(user_id, display_name, prize, "trivia_win")
            await self.heart.add_gift_bonus(user_id, display_name, 10)
            return {"winner": display_name, "prize": prize, "balance": bal}
        return None
