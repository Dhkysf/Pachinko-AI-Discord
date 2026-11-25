"""Pachinko Main Entrypoint: Runs Core, Web Admin Panel (:7878), Singing, & Discord Bot in one process."""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

import discord
import uvicorn

from brain.gate import ActionGate
from brain.llm import LLMClient
from brain.profile import ProfileManager
from brain.runner import BrainRunner
from brain.tools import ToolBox
from brain.triage import TriageEngine
from config import DATA_DIR, config, pop_startup_warnings
from core.bus import EventBus
from core.runtime import (
    EXIT_ALREADY_RUNNING,
    EXIT_CRASH,
    EXIT_OK,
    acquire_lock,
    release_lock,
    remove_status,
    setup_logging,
    watch_stop_request,
    write_status,
)
from core.store import Store, migrate_legacy_db
from discord_bot.client import build_discord_bot
from heart.engine import HeartEngine
from memory.manager import MemoryManager
from play.economy import EconomyAndPlay
from sing.engine import SingingEngine
from speech.stt import STTEngine
from speech.tts import TTSEngine
from web.server import create_web_app

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


async def run_all(run_dir: Path | None = None, started: int | None = None) -> None:
    run_dir = run_dir or (DATA_DIR / "run")
    started = int(time.time()) if started is None else int(started)
    status_path = run_dir / "status.json"
    stop_path = run_dir / "stop.request"

    store = None
    singing = None
    uvi_server = None
    web_task: asyncio.Task | None = None
    tts = None
    stt = None
    llm = None
    active_bot = None
    fg_task: asyncio.Task | None = None
    stop_task: asyncio.Task | None = None
    status_task: asyncio.Task | None = None

    try:
        write_status(
            status_path,
            state="starting",
            started=started,
            discord_ready=False,
            voice_sessions=0,
            llm_ok=False,
            llm_provider=config.llm_provider,
            llm_model=config.active_model(),
            ollama_model=config.ollama_model,
            llm_last_provider=config.llm_provider,
            llm_last_model=config.active_model(),
            llm_fallback_active=False,
        )

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
        log.info(
            "Brain LLM: provider=%s model=%s fallback=%s",
            config.llm_provider,
            config.active_model() or "(unset)",
            config.llm_fallback,
        )
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

        stop_requested = False

        def _trigger_stop() -> None:
            nonlocal stop_requested
            stop_requested = True
            log.info("Stop request received; shutting down...")
            if fg_task is not None and not fg_task.done():
                fg_task.cancel()

        async def _status_loop() -> None:
            while True:
                discord_ready = bool(active_bot and getattr(active_bot, "is_ready", lambda: False)())
                voice_sessions = len(getattr(active_bot, "voice_sessions", None) or {})
                llm_ok = await llm.check_health(60.0) if llm is not None else False
                last_prov, last_mod, fb_active = (
                    llm.status_snapshot() if llm is not None else (config.llm_provider, config.active_model(), False)
                )
                state = "running" if (discord_ready or not config.discord_token) else "starting"
                write_status(
                    status_path,
                    state=state,
                    started=started,
                    discord_ready=discord_ready,
                    voice_sessions=voice_sessions,
                    llm_ok=llm_ok,
                    llm_provider=config.llm_provider,
                    llm_model=config.active_model(),
                    ollama_model=config.ollama_model,
                    llm_last_provider=last_prov,
                    llm_last_model=last_mod,
                    llm_fallback_active=fb_active,
                )
                await asyncio.sleep(10.0)

        async def _run_foreground() -> None:
            nonlocal active_bot
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

        stop_task = asyncio.create_task(watch_stop_request(stop_path, _trigger_stop, interval=1.0))
        status_task = asyncio.create_task(_status_loop())
        fg_task = asyncio.create_task(_run_foreground())
        try:
            await fg_task
        except asyncio.CancelledError:
            if not stop_requested:
                raise
    finally:
        for task in (stop_task, status_task, fg_task):
            if task is not None and not task.done():
                task.cancel()
        pending = [t for t in (stop_task, status_task, fg_task) if t is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        try:
            last_prov, last_mod, fb_active = (
                llm.status_snapshot() if llm is not None else (config.llm_provider, config.active_model(), False)
            )
            write_status(
                status_path,
                state="stopping",
                started=started,
                discord_ready=bool(active_bot and getattr(active_bot, "is_ready", lambda: False)()),
                voice_sessions=len(getattr(active_bot, "voice_sessions", None) or {}),
                llm_ok=bool(llm and llm.last_ok),
                llm_provider=config.llm_provider,
                llm_model=config.active_model(),
                ollama_model=config.ollama_model,
                llm_last_provider=last_prov,
                llm_last_model=last_mod,
                llm_fallback_active=fb_active,
            )
        except Exception:
            pass

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
        remove_status(status_path)


def main() -> int:
    setup_logging(DATA_DIR / "logs")
    for w in pop_startup_warnings():
        log.warning("%s", w)
    run_dir = DATA_DIR / "run"
    lock_path = run_dir / "bot.lock"
    started = int(time.time())

    acquired, existing_pid = acquire_lock(lock_path, started=started)
    if not acquired:
        log.error("Pachinko is already running (PID %s in %s); refusing to start.", existing_pid, lock_path)
        return EXIT_ALREADY_RUNNING

    try:
        asyncio.run(run_all(run_dir=run_dir, started=started))
        log.info("Pachinko stopped safely.")
        return EXIT_OK
    except KeyboardInterrupt:
        log.info("Pachinko stopped safely.")
        return EXIT_OK
    except Exception:
        log.exception("Unhandled error in Pachinko")
        return EXIT_CRASH
    finally:
        release_lock(lock_path)
        remove_status(run_dir / "status.json")


if __name__ == "__main__":
    sys.exit(main())
