"""Entrypoint Utama Pachinko: Menjalankan Core, Web Admin Panel (:7878), Singing, & Discord Bot dalam satu proses."""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

import discord
import uvicorn

from brain.gate import ActionGate
from brain.llm import LLMClient
from brain.profile import ProfileManager
from brain.runner import BrainRunner
from brain.tools import ToolBox
from brain.triage import TriageEngine
from config import DATA_DIR, config
from core.bus import Event, EventBus
from core.store import Store, migrate_legacy_db
from discord_bot.client import build_discord_bot
from heart.engine import HeartEngine
from memory.manager import MemoryManager
from play.economy import EconomyAndPlay
from sing.engine import SingingEngine
from speech.stt import STTEngine
from speech.tts import TTSEngine
from web.server import create_web_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pachinko")


async def run_all() -> None:
    migrate_legacy_db(config.db_path, DATA_DIR)
    store = Store(config.db_path)
    await store.init()

    bus = EventBus()

    async def _persist_ev(ev: Event) -> None:
        await store.log_event(ev.id, ev.topic, ev.ts, ev.payload)

    bus.set_persist_hook(_persist_ev)

    profiles = ProfileManager(store)
    prof = await profiles.reload()

    heart = HeartEngine(store, bus, half_life_mins=prof.half_life_mins)
    await heart.load()

    memory = MemoryManager(store, bus)
    gate = ActionGate(store, bus)
    await gate.load()

    singing = SingingEngine(bus)
    singing.start()

    tools = ToolBox(memory, gate, song_enqueue_cb=singing.enqueue)
    llm = LLMClient()
    triage = TriageEngine()
    brain = BrainRunner(
        store=store,
        bus=bus,
        profiles=profiles,
        heart=heart,
        memory=memory,
        tools=tools,
        llm=llm,
    )

    stt = STTEngine()
    tts = TTSEngine()
    economy = EconomyAndPlay(store, bus, heart)

    web_app = create_web_app(
        store=store,
        bus=bus,
        profiles=profiles,
        heart=heart,
        memory=memory,
        gate=gate,
        brain=brain,
        singing=singing,
        economy=economy,
        tts=tts,
    )

    uvi_cfg = uvicorn.Config(
        web_app,
        host=config.panel_host,
        port=config.panel_port,
        log_level="warning",
    )
    uvi_server = uvicorn.Server(uvi_cfg)
    web_task = asyncio.create_task(uvi_server.serve())
    log.info("🌐 Web Admin Panel aktif di: http://%s:%d", config.panel_host, config.panel_port)

    try:
        if not config.discord_token:
            log.warning("DISCORD_TOKEN belum diisi di .env — menjalankan Mode Web Panel & Playground saja.")
            await web_task
            return

        bot = build_discord_bot(
            store=store,
            bus=bus,
            profiles=profiles,
            heart=heart,
            memory=memory,
            triage=triage,
            gate=gate,
            brain=brain,
            stt=stt,
            tts=tts,
            singing=singing,
            economy=economy,
            enable_message_content=True,
        )
        try:
            await bot.start(config.discord_token)
        except discord.PrivilegedIntentsRequired:
            log.warning(
                "⚠️ Message Content Intent belum diaktifkan di Discord Developer Portal! "
                "Menjalankan ulang dalam mode Voice + Slash Command saja. "
                "(Aktifkan 'Message Content Intent' di tab Bot Developer Portal supaya bot bisa baca obrolan teks biasa)."
            )
            await bot.close()
            bot2 = build_discord_bot(
                store=store,
                bus=bus,
                profiles=profiles,
                heart=heart,
                memory=memory,
                triage=triage,
                gate=gate,
                brain=brain,
                stt=stt,
                tts=tts,
                singing=singing,
                economy=economy,
                enable_message_content=False,
            )
            await bot2.start(config.discord_token)
    finally:
        uvi_server.should_exit = True
        await singing.stop()
        await tts.close()
        await llm.close()
        await store.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\nPachinko dihentikan dengan aman.")
        sys.exit(0)
