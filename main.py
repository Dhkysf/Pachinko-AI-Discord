"""Pachinko Main Entrypoint: Runs Core, Web Admin Panel (:7878), Singing, & Discord Bot in one process."""
from __future__ import annotations

import asyncio
import logging
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
from core.bus import EventBus
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


async def shutdown_all(
    *,
    bot=None,
    singing=None,
    uvi_server=None,
    web_task: asyncio.Task | None = None,
    tts=None,
    stt=None,
    llm=None,
    store=None,
) -> None:
    if bot is not None:
        sessions = getattr(bot, "voice_sessions", None) or {}
        for gid, sess in list(sessions.items()):
            try:
                await sess.stop()
            except Exception:
                log.exception("Failed stopping voice session %s", gid)
            finally:
                sessions.pop(gid, None)

        sched_task = getattr(bot, "_sched_task", None)
        if sched_task is not None and not sched_task.done():
            sched_task.cancel()
            await asyncio.gather(sched_task, return_exceptions=True)

        try:
            if not getattr(bot, "is_closed", lambda: False)():
                await bot.close()
        except Exception:
            log.exception("Failed closing Discord bot")

    if singing is not None:
        try:
            await singing.stop()
        except Exception:
            log.exception("Failed stopping singing engine")

    if uvi_server is not None:
        uvi_server.should_exit = True
    if web_task is not None:
        try:
            await asyncio.wait_for(web_task, timeout=5.0)
        except asyncio.TimeoutError:
            web_task.cancel()
            await asyncio.gather(web_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Failed stopping web server")

    if tts is not None:
        try:
            await tts.close()
        except Exception:
            log.exception("Failed closing TTS client")

    if stt is not None:
        try:
            await stt.close()
        except Exception:
            log.exception("Failed closing STT client")

    if llm is not None:
        try:
            await llm.close()
        except Exception:
            log.exception("Failed closing LLM client")

    if store is not None:
        try:
            await store.close()
        except Exception:
            log.exception("Failed closing store")


async def run_all() -> None:
    migrate_legacy_db(config.db_path, DATA_DIR)
    store = Store(config.db_path)
    await store.init()

    bus = EventBus()

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
    uvi_server.install_signal_handlers = lambda: None
    web_task = asyncio.create_task(uvi_server.serve())
    log.info("🌐 Web Admin Panel active at: http://%s:%d", config.panel_host, config.panel_port)

    active_bot = None
    try:
        if not config.discord_token:
            log.warning("DISCORD_TOKEN is not set in .env — running Web Panel & Playground mode only.")
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
        active_bot = bot
        try:
            await bot.start(config.discord_token)
        except discord.PrivilegedIntentsRequired:
            log.warning(
                "⚠️ Message Content Intent is not enabled in the Discord Developer Portal! "
                "Restarting in Voice + Slash Command mode only. "
                "(Enable 'Message Content Intent' in the Developer Portal Bot tab so the bot can read regular text chat)."
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
            active_bot = bot2
            await bot2.start(config.discord_token)
    finally:
        await shutdown_all(
            bot=active_bot,
            singing=singing,
            uvi_server=uvi_server,
            web_task=web_task,
            tts=tts,
            stt=stt,
            llm=llm,
            store=store,
        )


if __name__ == "__main__":
    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\nPachinko stopped safely.")
        sys.exit(0)
