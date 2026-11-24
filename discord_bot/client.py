"""Unified Discord Bot (Voice DAVE E2EE + Text Triage + Mind/Reminder Loop + Economy/Singing Commands).

Important note: DO NOT add `from __future__ import annotations` in this file so
py-cord slash command parameter type introspection always reads the real Python classes.
"""
import asyncio
import io
import logging
import time

import discord

from brain.gate import ActionGate
from brain.profile import ProfileManager
from brain.runner import BrainRunner
from brain.triage import MODE_HOURLY_CAP, TriageEngine
from config import config
from core.bus import EventBus
from core.store import Store
from heart.engine import HeartEngine
from memory.manager import MemoryManager
from play.economy import EconomyAndPlay, render_card_svg
from sing.engine import SingingEngine
from speech.stt import STTEngine
from speech.tts import TTSEngine
from voice.session import NaturalVoiceSession

log = logging.getLogger("discord_bot")
logging.getLogger("discord.voice.receive.reader").setLevel(logging.WARNING)

GUILDS = [config.guild_id] if config.guild_id else None


def build_discord_bot(
    *,
    store: Store,
    bus: EventBus,
    profiles: ProfileManager,
    heart: HeartEngine,
    memory: MemoryManager,
    triage: TriageEngine,
    gate: ActionGate,
    brain: BrainRunner,
    stt: STTEngine,
    tts: TTSEngine,
    singing: SingingEngine,
    economy: EconomyAndPlay,
    enable_message_content: bool = True,
) -> discord.Bot:
    intents = discord.Intents.default()
    intents.voice_states = True
    intents.guilds = True
    intents.message_content = enable_message_content

    bot = discord.Bot(intents=intents)
    sessions: dict[int, NaturalVoiceSession] = {}
    bot.voice_sessions = sessions  # type: ignore[attr-defined]

    def _lookup_voice_by_room(room_id: str):
        if room_id.startswith("guild:"):
            gid = int(room_id.split(":", 1)[1])
            return sessions.get(gid)
        for s in sessions.values():
            if s.text_channel and str(getattr(s.text_channel, "id", "")) == room_id:
                return s
        return next(iter(sessions.values()), None)

    singing.bind_voice_lookup(_lookup_voice_by_room)

    # ---------------- Background Schedule & Mind Loop ----------------
    async def _schedule_loop() -> None:
        await bot.wait_until_ready()
        while not bot.is_closed():
            try:
                due_list = await memory.pop_due_schedules()
                for item in due_list:
                    prof = profiles.current
                    note = item["note"]
                    uname = item["user_name"]
                    uid = item["user_id"]
                    room_id = item["room_id"]
                    msg_txt = f"⏰ <@{uid}> (**{uname}**), pengingat dari {prof.display_name}: **{note}**"

                    # Send to text channel if found
                    if room_id.isdigit():
                        ch = bot.get_channel(int(room_id))
                        if ch and hasattr(ch, "send"):
                            await ch.send(msg_txt)

                    # If in a voice channel, speak it aloud as well!
                    for sess in sessions.values():
                        if sess.vc.is_connected() and not sess.singing_active:
                            spoken = f"Eh {uname}, waktunya pengingat kamu nih: {note}."
                            pcm = await tts.synthesize_pcm(spoken, emotion="happy")
                            await sess.play_raw_pcm(pcm, spoken)
                            break
            except Exception:
                log.exception("Error in schedule loop")
            await asyncio.sleep(5.0)

    @bot.event
    async def on_ready():
        log.info(
            "🟢 %s online as %s (ID: %s) | Panel: http://%s:%d",
            profiles.current.display_name,
            bot.user,
            bot.user.id,
            config.panel_host,
            config.panel_port,
        )
        if not getattr(bot, "_sched_started", False):
            bot._sched_started = True  # type: ignore[attr-defined]
            asyncio.create_task(_schedule_loop())
        await bus.emit("discord.ready", bot_user=str(bot.user), bot_id=str(bot.user.id))

    # ---------------- Text Channel Listener & Triage ----------------
    @bot.event
    async def on_message(message: discord.Message):
        if message.author.bot or not message.content.strip():
            return

        room_id = str(message.channel.id)
        uid = str(message.author.id)
        uname = message.author.display_name

        # 1. Check if answering an active Trivia mini-game
        trivia_win = await economy.check_trivia_answer(room_id, uid, uname, message.content)
        if trivia_win:
            await message.reply(
                f"🎉 **Benar, {trivia_win['winner']}!** Kamu dapat **+{trivia_win['prize']} koin** "
                f"(Saldo: **{trivia_win['balance']} koin**) & Bond naik!",
                mention_author=False,
            )
            return

        # 2. Deterministic triage (mention / called by name / chime in per channel mode)
        prof = profiles.current
        is_dm = isinstance(message.channel, discord.DMChannel)
        mentioned = (bot.user in message.mentions) if bot.user else False
        mode = await store.get_channel_mode(room_id, prof.default_channel_mode)
        person = await heart.get_person(uid, uname)

        decision = triage.decide_text(
            room_id=room_id,
            text=message.content,
            mode=mode,
            profile=prof,
            mentioned=mentioned,
            is_dm=is_dm,
            autonomy_paused=gate.paused,
            user_opted_out=bool(person.get("opt_out_approach")),
        )
        if not decision.should_wake:
            return

        await bus.emit("brain.triage", room_id=room_id, speaker=uname, reason=decision.reason, mode=mode)
        async with message.channel.typing():
            res = await brain.run_text_turn(
                room_id=room_id,
                user_id=uid,
                user_name=uname,
                user_text=message.content,
                wake_reason=decision.reason,
            )
        if not res.silent and res.text and res.text.strip():
            await message.reply(res.text[:1990], mention_author=False)

    @bot.event
    async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        sess = sessions.get(member.guild.id)
        if not sess or not sess.vc.channel:
            return
        humans = [m for m in sess.vc.channel.members if not m.bot]
        if not humans:
            log.info("Voice channel empty, leaving automatically.")
            await sessions.pop(member.guild.id).stop()
            await bus.emit("voice.left", guild_id=member.guild.id, reason="channel_empty")

    # ================ SLASH COMMANDS ================

    @bot.slash_command(name="join", description="Ajak karakter masuk ke voice channel untuk ngobrol verbal", guild_ids=GUILDS)
    async def cmd_join(ctx: discord.ApplicationContext):
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.respond("Masuk ke voice channel dulu ya baru panggil `/join`.", ephemeral=True)
        await ctx.defer()

        if ctx.guild.id in sessions:
            await sessions.pop(ctx.guild.id).stop()

        ch = ctx.author.voice.channel
        vc: discord.VoiceClient = await ch.connect()
        dave_ok = getattr(vc, "is_dave_connection", lambda: False)()

        sess = NaturalVoiceSession(
            guild_id=ctx.guild.id,
            vc=vc,
            text_channel=ctx.channel,
            bus=bus,
            profiles=profiles,
            triage=triage,
            brain=brain,
            memory=memory,
            stt=stt,
            tts=tts,
        )
        sess.start()
        sessions[ctx.guild.id] = sess
        await bus.emit("voice.joined", guild_id=ctx.guild.id, channel=ch.name, dave=dave_ok)

        prof = profiles.current
        await ctx.followup.send(
            f"🎧 **{prof.display_name}** masuk ke **{ch.name}** *(DAVE E2EE: {'Aktif' if dave_ok else 'Standar'})*.\n"
            f"Langsung ngomong aja—aku dengerin dengan *Smart Turn-Taking* & *Streaming Voice*!"
        )

    @bot.slash_command(name="leave", description="Suruh karakter keluar dari voice channel", guild_ids=GUILDS)
    async def cmd_leave(ctx: discord.ApplicationContext):
        sess = sessions.pop(ctx.guild.id, None)
        if not sess:
            return await ctx.respond("Aku lagi nggak di voice channel.", ephemeral=True)
        await sess.stop()
        await bus.emit("voice.left", guild_id=ctx.guild.id, reason="command")
        await ctx.respond("Sampai nanti ya! 👋")

    @bot.slash_command(name="say", description="Kirim pesan ke karakter (dijawab teks & diucapkan di voice bila aktif)", guild_ids=GUILDS)
    @discord.option("pesan", str, description="Apa yang mau kamu sampaikan")
    async def cmd_say(ctx: discord.ApplicationContext, pesan):
        await ctx.defer()
        sess = sessions.get(ctx.guild.id)
        try:
            if sess and sess.vc.is_connected():
                ans = await sess.speak_direct_text(ctx.author.display_name, str(ctx.author.id), pesan)
                return await ctx.followup.send(f"🎙️ **{ctx.author.display_name}:** {pesan}\n🤖 **{profiles.current.display_name}:** {ans or '...'}")
            res = await brain.run_text_turn(
                room_id=str(ctx.channel.id),
                user_id=str(ctx.author.id),
                user_name=ctx.author.display_name,
                user_text=pesan,
                wake_reason="slash_say",
            )
            await ctx.followup.send(res.text.strip() or "*(diam)*")
        except Exception as e:
            log.exception("Error in /say")
            await ctx.followup.send(f"⚠️ Gagal memproses: `{str(e)[:280]}`")

    @bot.slash_command(name="mode", description="Atur seberapa sering karakter nimbrung otomatis di text channel ini", guild_ids=GUILDS)
    @discord.option("level", str, choices=list(MODE_HOURLY_CAP.keys()), description="off | mention-only | quiet | normal | lively")
    async def cmd_mode(ctx: discord.ApplicationContext, level):
        await store.set_channel_mode(str(ctx.channel.id), level)
        cap = MODE_HOURLY_CAP[level]
        info = "hanya saat dipanggil" if cap == 0 else f"bisa ikut nimbrung maks {cap}x/jam"
        await ctx.respond(f"⚙️ Mode channel ini diubah ke **`{level}`** ({info}).")

    @bot.slash_command(name="mood", description="Lihat perasaan, energi, dan emosi aktif karakter saat ini", guild_ids=GUILDS)
    async def cmd_mood(ctx: discord.ApplicationContext):
        prof = profiles.current
        st = heart.state_summary(prof.temperament)
        emos = ", ".join(f"`{k}` ({v:.2f})" for k, v in st["active_emotions"].items()) or "tenang / netral"
        await ctx.respond(
            f"💓 **Kondisi Hati {prof.display_name}**\n"
            f"- Ekspresi Dominan: **{st['dominant']}**\n"
            f"- Valence (Suasana Hati): `{st['valence']:+.2f}` | Arousal: `{st['arousal']:.2f}` | Energi: `{int(st['energy']*100)}%`\n"
            f"- Emosi Aktif: {emos}"
        )

    @bot.slash_command(name="bond", description="Cek tingkat kedekatan (9-Axis Bond) karakter denganmu", guild_ids=GUILDS)
    async def cmd_bond(ctx: discord.ApplicationContext):
        p = await heart.get_person(str(ctx.author.id), ctx.author.display_name)
        notes = await memory.get_notes(f"person:{ctx.author.id}", limit=4)
        note_str = "\n".join(f"  • {n['fact']}" for n in notes) if notes else "  • *(belum ada catatan khusus)*"
        await ctx.respond(
            f"🤝 **Hubungan {profiles.current.display_name} & {ctx.author.display_name}**\n"
            f"- Status: **{p['tier']}** (`{p['bond']:.1f} / 1000`)\n"
            f"- Trust: `{p['trust']:.1f}/100` | Familiarity: `{p['familiarity']:.1f}/100` | Warmth: `{p['warmth']:+.2f}` | Rapport: `{p['rapport']:.2f}`\n"
            f"- Kesan: *{p['impression']}*\n"
            f"- Buku Catatan:\n{note_str}"
        )

    @bot.slash_command(name="sing", description="Request lagu / AI cover untuk dinyanyikan di voice channel", guild_ids=GUILDS)
    @discord.option("lagu", str, description="Judul lagu atau URL YouTube")
    async def cmd_sing(ctx: discord.ApplicationContext, lagu):
        sess = sessions.get(ctx.guild.id)
        if not sess or not sess.vc.is_connected():
            return await ctx.respond("Panggil `/join` ke voice channel dulu sebelum request lagu ya!", ephemeral=True)
        await ctx.defer()
        msg = await singing.enqueue(f"guild:{ctx.guild.id}", str(ctx.author.id), ctx.author.display_name, lagu)
        await ctx.followup.send(f"🎤 {msg}")

    @bot.slash_command(name="skip", description="Lewati lagu yang sedang diputar", guild_ids=GUILDS)
    async def cmd_skip(ctx: discord.ApplicationContext):
        if singing.skip_current():
            await ctx.respond("⏭️ Lagu dilewati!")
        else:
            await ctx.respond("Sedang tidak ada lagu yang diputar.", ephemeral=True)

    @bot.slash_command(name="queue", description="Lihat antrean lagu saat ini", guild_ids=GUILDS)
    async def cmd_queue(ctx: discord.ApplicationContext):
        snap = singing.snapshot()
        cur = snap["current"]
        q = snap["queue"]
        if not cur and not q:
            return await ctx.respond("Antrean lagu kosong. Pakai `/sing` buat request!", ephemeral=True)
        lines = []
        if cur:
            lines.append(f"▶️ **Sekarang:** {cur['title']} *(req: {cur['requester_name']}, status: {cur['status']})*")
        for i, it in enumerate(q[:10], 1):
            lines.append(f"`{i}.` {it['query']} *(req: {it['requester_name']})*")
        await ctx.respond("🎶 **Antrean Lagu**\n" + "\n".join(lines))

    @bot.slash_command(name="daily", description="Ambil koin harian + ramalan keberuntungan hari ini", guild_ids=GUILDS)
    async def cmd_daily(ctx: discord.ApplicationContext):
        res = await economy.daily_checkin(str(ctx.author.id), ctx.author.display_name)
        if not res["ok"]:
            return await ctx.respond(res["message"], ephemeral=True)
        await ctx.respond(
            f"🪙 **Check-in Harian Berhasil!** (Streak: **{res['streak']} hari**)\n"
            f"- Hadiah: **+{res['reward']} koin** (Saldo: **{res['balance']} koin**)\n"
            f"- Ramalan hari ini: **{res['fortune_title']}** — *{res['fortune_desc']}*"
        )

    @bot.slash_command(name="card", description="Tarik kartu koleksi karakter (60 koin)", guild_ids=GUILDS)
    async def cmd_card(ctx: discord.ApplicationContext):
        res = await economy.draw_card(str(ctx.author.id), ctx.author.display_name, cost=60)
        if not res["ok"]:
            return await ctx.respond(res["message"], ephemeral=True)
        card = res["card"]
        svg = render_card_svg(card, ctx.author.display_name, profiles.current.display_name)
        file = discord.File(io.BytesIO(svg.encode("utf-8")), filename=f"card_{card['rarity'].lower()}.svg")
        await ctx.respond(
            f"🎴 **[{card['rarity']}] {card['title']}**\n*\"{card['quote']}\"*\n*(Sisa saldo: {res['balance']} koin)*",
            file=file,
        )

    @bot.slash_command(name="gift", description="Kasih koin/traktiran ke karakter untuk menaikkan Bond & mood", guild_ids=GUILDS)
    @discord.option("jumlah", int, description="Jumlah koin yang diberikan")
    async def cmd_gift(ctx: discord.ApplicationContext, jumlah):
        res = await economy.gift_to_character(str(ctx.author.id), ctx.author.display_name, int(jumlah))
        if not res["ok"]:
            return await ctx.respond(res["message"], ephemeral=True)
        p = res["person"]
        await ctx.respond(
            f"🎁 **{ctx.author.display_name}** mentraktir **{jumlah} koin** ke **{profiles.current.display_name}**!\n"
            f"Bond naik ke **`{p['bond']:.1f}` ({p['tier']})** | Sisa saldo: **{res['balance']} koin**."
        )

    @bot.slash_command(name="trivia", description="Mulai mini-game tebak cepat berhadiah koin di channel ini", guild_ids=GUILDS)
    async def cmd_trivia(ctx: discord.ApplicationContext):
        q, prize = economy.start_trivia(str(ctx.channel.id))
        await ctx.respond(f"🧩 **Mini-Game Cepat (Hadiah {prize} koin)!**\n> {q}\n*Ketik jawabanmu langsung di chat!*")

    @bot.slash_command(name="remind", description="Minta karakter mengingatkan sesuatu beberapa menit lagi", guild_ids=GUILDS)
    @discord.option("menit", int, description="Berapa menit lagi")
    @discord.option("catatan", str, description="Apa yang perlu diingatkan")
    async def cmd_remind(ctx: discord.ApplicationContext, menit, catatan):
        mins = max(1, min(1440, int(menit)))
        await memory.add_schedule(
            kind="reminder",
            room_id=str(ctx.channel.id),
            user_id=str(ctx.author.id),
            user_name=ctx.author.display_name,
            note=catatan,
            due_at=time.time() + mins * 60.0,
        )
        await ctx.respond(f"⏰ Sip! Aku bakal ingetin **{catatan}** dalam **{mins} menit** lagi.")

    @bot.slash_command(name="privacy", description="Atur apakah karakter boleh menyapa/nimbrung ke pesanmu secara otomatis", guild_ids=GUILDS)
    @discord.option("izinkan", bool, description="True = boleh nimbrung, False = hanya merespons kalau kamu panggil")
    async def cmd_privacy(ctx: discord.ApplicationContext, izinkan):
        await heart.get_person(str(ctx.author.id), ctx.author.display_name)
        await store.conn.execute(
            "UPDATE heart_person SET opt_out_approach = ? WHERE user_id = ?",
            (0 if izinkan else 1, str(ctx.author.id)),
        )
        await store.conn.commit()
        status = "diizinkan menyapa otomatis" if izinkan else "hanya membalas saat kamu panggil"
        await ctx.respond(f"🔒 Pengaturan privasi disimpan: **{status}**.", ephemeral=True)

    @bot.slash_command(name="reset", description="Bersihkan riwayat obrolan jangka pendek di ruang ini (Bond & Notebook tetap aman)", guild_ids=GUILDS)
    async def cmd_reset(ctx: discord.ApplicationContext):
        await memory.clear_room_transcript(str(ctx.channel.id))
        await memory.clear_room_transcript(f"guild:{ctx.guild.id}")
        await ctx.respond("🧹 Ingatan obrolan pendek sudah disegarkan! *(Bond & Buku Catatan tetap tersimpan)*", ephemeral=True)

    @bot.slash_command(name="reload", description="Muat ulang file persona & lorebook tanpa restart bot", guild_ids=GUILDS)
    async def cmd_reload(ctx: discord.ApplicationContext):
        prof = await profiles.reload()
        await ctx.respond(f"✨ Persona **{prof.display_name}** berhasil dimuat ulang!", ephemeral=True)

    return bot
