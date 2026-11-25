import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord_bot.client as client_mod
from config import config
from core.bus import EventBus


def _make_bot(
    *,
    store=None,
    bus=None,
    profiles=None,
    memory=None,
    tts=None,
    singing=None,
):
    bus = bus or EventBus()
    prof = SimpleNamespace(display_name="Pachinko", default_channel_mode="quiet")
    profiles = profiles or SimpleNamespace(
        current=prof,
        reload=AsyncMock(return_value=prof),
    )
    store = store or SimpleNamespace(set_channel_mode=AsyncMock())
    memory = memory or SimpleNamespace(clear_room_transcript=AsyncMock())
    tts = tts or SimpleNamespace(synthesize_pcm=AsyncMock(return_value=b"\x00\x01"))
    singing = singing or SimpleNamespace(
        bind_voice_lookup=MagicMock(),
        snapshot=MagicMock(return_value={"current": None, "queue": []}),
        skip_current=MagicMock(return_value=True),
    )
    return client_mod.build_discord_bot(
        store=store,
        bus=bus,
        profiles=profiles,
        heart=SimpleNamespace(),
        memory=memory,
        triage=SimpleNamespace(),
        gate=SimpleNamespace(paused=False),
        brain=SimpleNamespace(),
        stt=SimpleNamespace(),
        tts=tts,
        singing=singing,
        economy=SimpleNamespace(),
    )


def _get_cmd(bot, name: str):
    for cmd in bot.pending_application_commands:
        if cmd.name == name:
            return cmd.callback
    raise KeyError(f"Command {name!r} not found")


def _fake_ctx(user_id: int, *, manage_guild: bool = False, guild_id: int | None = 1):
    responses = []

    async def _respond(msg, **kwargs):
        responses.append((msg, kwargs))

    guild = SimpleNamespace(id=guild_id) if guild_id is not None else None
    author = SimpleNamespace(
        id=user_id,
        display_name=f"User{user_id}",
        guild_permissions=SimpleNamespace(manage_guild=manage_guild),
    )
    return SimpleNamespace(
        author=author,
        guild=guild,
        channel=SimpleNamespace(id=100),
        respond=_respond,
        responses=responses,
    )


class TestDiscordRules(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._orig_owner_ids = config.owner_ids

    def tearDown(self):
        config.owner_ids = self._orig_owner_ids

    def test_allowed_mentions_blocks_mass_pings(self):
        bot = _make_bot()
        am = bot.allowed_mentions
        self.assertIsNotNone(am)
        self.assertIs(am.everyone, False)
        self.assertIs(am.roles, False)
        self.assertIs(am.users, True)
        self.assertIs(am.replied_user, False)

    def test_is_admin_owner_manage_guild_and_member(self):
        config.owner_ids = (111,)
        self.assertTrue(client_mod.is_admin(_fake_ctx(111, manage_guild=False)))
        self.assertTrue(client_mod.is_admin(_fake_ctx(222, manage_guild=True)))
        self.assertFalse(client_mod.is_admin(_fake_ctx(333, manage_guild=False)))
        self.assertFalse(client_mod.is_admin(_fake_ctx(222, manage_guild=True, guild_id=None)))

        config.owner_ids = ()
        self.assertTrue(client_mod.is_admin(_fake_ctx(222, manage_guild=True)))
        self.assertFalse(client_mod.is_admin(_fake_ctx(333, manage_guild=False)))

    async def test_admin_commands_denied_for_member_allowed_for_admin(self):
        config.owner_ids = ()
        store = SimpleNamespace(set_channel_mode=AsyncMock())
        memory = SimpleNamespace(clear_room_transcript=AsyncMock())
        prof = SimpleNamespace(display_name="Pachinko", default_channel_mode="quiet")
        profiles = SimpleNamespace(current=prof, reload=AsyncMock(return_value=prof))
        bot = _make_bot(store=store, memory=memory, profiles=profiles)

        cmd_mode = _get_cmd(bot, "mode")
        cmd_reset = _get_cmd(bot, "reset")
        cmd_reload = _get_cmd(bot, "reload")

        member_ctx = _fake_ctx(333, manage_guild=False)
        await cmd_mode(member_ctx, "quiet")
        await cmd_reset(member_ctx)
        await cmd_reload(member_ctx)
        self.assertEqual(len(member_ctx.responses), 3)
        for msg, kw in member_ctx.responses:
            self.assertTrue(kw.get("ephemeral"))
            self.assertIn("admin", msg.lower())
        store.set_channel_mode.assert_not_awaited()
        memory.clear_room_transcript.assert_not_awaited()
        profiles.reload.assert_not_awaited()

        admin_ctx = _fake_ctx(222, manage_guild=True)
        await cmd_mode(admin_ctx, "quiet")
        await cmd_reset(admin_ctx)
        await cmd_reload(admin_ctx)
        store.set_channel_mode.assert_awaited_once()
        self.assertEqual(memory.clear_room_transcript.await_count, 2)
        profiles.reload.assert_awaited_once()

    async def test_skip_permissions(self):
        config.owner_ids = ()
        singing = SimpleNamespace(
            bind_voice_lookup=MagicMock(),
            snapshot=MagicMock(return_value={"current": {"requester_id": "42", "title": "Song"}, "queue": []}),
            skip_current=MagicMock(return_value=True),
        )
        bot = _make_bot(singing=singing)
        cmd_skip = _get_cmd(bot, "skip")

        other_ctx = _fake_ctx(77, manage_guild=False)
        await cmd_skip(other_ctx)
        self.assertTrue(other_ctx.responses[0][1].get("ephemeral"))
        singing.skip_current.assert_not_called()

        req_ctx = _fake_ctx(42, manage_guild=False)
        await cmd_skip(req_ctx)
        singing.skip_current.assert_called_once()

        singing.skip_current.reset_mock()
        admin_ctx = _fake_ctx(99, manage_guild=True)
        await cmd_skip(admin_ctx)
        singing.skip_current.assert_called_once()

        singing.snapshot.return_value = {"current": None, "queue": []}
        empty_ctx = _fake_ctx(42, manage_guild=False)
        await cmd_skip(empty_ctx)
        self.assertTrue(empty_ctx.responses[0][1].get("ephemeral"))

    async def test_reminder_routing_text_channel(self):
        ch = SimpleNamespace(id=123, guild=SimpleNamespace(id=1), send=AsyncMock())
        bot = SimpleNamespace(get_channel=MagicMock(return_value=ch), get_guild=MagicMock(return_value=None))
        sess1 = SimpleNamespace(
            guild_id=1,
            text_channel=ch,
            singing_active=False,
            vc=SimpleNamespace(is_connected=lambda: True),
            play_raw_pcm=AsyncMock(),
        )
        sess2 = SimpleNamespace(
            guild_id=2,
            text_channel=None,
            singing_active=False,
            vc=SimpleNamespace(is_connected=lambda: True),
            play_raw_pcm=AsyncMock(),
        )
        profiles = SimpleNamespace(current=SimpleNamespace(display_name="Pachinko"))
        tts = SimpleNamespace(synthesize_pcm=AsyncMock(return_value=b"pcm"))
        item = {"id": 10, "room_id": "123", "user_id": "42", "user_name": "Budi", "note": "minum air"}

        ok = await client_mod.deliver_reminder(bot, {1: sess1, 2: sess2}, item, profiles, tts)
        self.assertTrue(ok)
        ch.send.assert_awaited_once()
        sess1.play_raw_pcm.assert_awaited_once()
        sess2.play_raw_pcm.assert_not_awaited()

    async def test_reminder_routing_guild_with_session(self):
        ch1 = SimpleNamespace(id=555, send=AsyncMock())
        bot = SimpleNamespace(get_channel=MagicMock(return_value=None), get_guild=MagicMock(return_value=None))
        sess1 = SimpleNamespace(
            guild_id=1,
            text_channel=ch1,
            singing_active=False,
            vc=SimpleNamespace(is_connected=lambda: True),
            play_raw_pcm=AsyncMock(),
        )
        sess2 = SimpleNamespace(
            guild_id=2,
            text_channel=None,
            singing_active=False,
            vc=SimpleNamespace(is_connected=lambda: True),
            play_raw_pcm=AsyncMock(),
        )
        profiles = SimpleNamespace(current=SimpleNamespace(display_name="Pachinko"))
        tts = SimpleNamespace(synthesize_pcm=AsyncMock(return_value=b"pcm"))
        item = {"id": 11, "room_id": "guild:1", "user_id": "42", "user_name": "Budi", "note": "rapat"}

        ok = await client_mod.deliver_reminder(bot, {1: sess1, 2: sess2}, item, profiles, tts)
        self.assertTrue(ok)
        ch1.send.assert_awaited_once()
        sess1.play_raw_pcm.assert_awaited_once()
        sess2.play_raw_pcm.assert_not_awaited()

        # When singing_active=True, still posts to text_channel but does not speak
        ch1.send.reset_mock()
        sess1.play_raw_pcm.reset_mock()
        sess1.singing_active = True
        ok2 = await client_mod.deliver_reminder(bot, {1: sess1, 2: sess2}, item, profiles, tts)
        self.assertTrue(ok2)
        ch1.send.assert_awaited_once()
        sess1.play_raw_pcm.assert_not_awaited()

    async def test_reminder_routing_guild_without_session_fallback(self):
        me = SimpleNamespace(id=999)
        no_send = SimpleNamespace(send=AsyncMock(), permissions_for=lambda _m: SimpleNamespace(send_messages=False))
        yes_send = SimpleNamespace(send=AsyncMock(), permissions_for=lambda _m: SimpleNamespace(send_messages=True))
        guild1 = SimpleNamespace(id=1, me=me, system_channel=None, text_channels=[no_send, yes_send])
        bot = SimpleNamespace(get_channel=MagicMock(return_value=None), get_guild=MagicMock(return_value=guild1))
        sess2 = SimpleNamespace(
            guild_id=2,
            text_channel=None,
            singing_active=False,
            vc=SimpleNamespace(is_connected=lambda: True),
            play_raw_pcm=AsyncMock(),
        )
        profiles = SimpleNamespace(current=SimpleNamespace(display_name="Pachinko"))
        tts = SimpleNamespace(synthesize_pcm=AsyncMock(return_value=b"pcm"))
        item = {"id": 12, "room_id": "guild:1", "user_id": "42", "user_name": "Budi", "note": "tidur"}

        ok = await client_mod.deliver_reminder(bot, {2: sess2}, item, profiles, tts)
        self.assertTrue(ok)
        no_send.send.assert_not_awaited()
        yes_send.send.assert_awaited_once()
        sess2.play_raw_pcm.assert_not_awaited()

    async def test_reminder_undeliverable_logs_warning(self):
        bot = SimpleNamespace(get_channel=MagicMock(return_value=None), get_guild=MagicMock(return_value=None))
        profiles = SimpleNamespace(current=SimpleNamespace(display_name="Pachinko"))
        tts = SimpleNamespace(synthesize_pcm=AsyncMock())
        item = {"id": 777, "room_id": "guild:999", "user_id": "42", "user_name": "Budi", "note": "hilang"}

        with self.assertLogs("discord_bot", level="WARNING") as cm:
            ok = await client_mod.deliver_reminder(bot, {}, item, profiles, tts)
        self.assertFalse(ok)
        self.assertTrue(any("777" in line for line in cm.output))

    async def test_bot_kicked_from_voice_cleans_up_session(self):
        bus = EventBus()
        bot = _make_bot(bus=bus)
        bot._connection.user = SimpleNamespace(id=999)

        sess = SimpleNamespace(
            vc=SimpleNamespace(channel=None),
            stop=AsyncMock(),
        )
        bot.voice_sessions[1] = sess

        member = SimpleNamespace(id=999, guild=SimpleNamespace(id=1))
        await bot.on_voice_state_update(
            member,
            SimpleNamespace(channel=SimpleNamespace(id=50)),
            SimpleNamespace(channel=None),
        )
        self.assertNotIn(1, bot.voice_sessions)
        sess.stop.assert_awaited_once()
        self.assertTrue(
            any(e["topic"] == "voice.left" and e["payload"].get("reason") == "disconnected" for e in bus.recent())
        )

    async def test_channel_empty_still_leaves_voice(self):
        bus = EventBus()
        bot = _make_bot(bus=bus)
        bot._connection.user = SimpleNamespace(id=999)

        bot_member = SimpleNamespace(id=999, bot=True)
        vc_ch = SimpleNamespace(members=[bot_member])
        sess = SimpleNamespace(
            vc=SimpleNamespace(channel=vc_ch),
            stop=AsyncMock(),
        )
        bot.voice_sessions[1] = sess

        human = SimpleNamespace(id=42, guild=SimpleNamespace(id=1))
        await bot.on_voice_state_update(
            human,
            SimpleNamespace(channel=vc_ch),
            SimpleNamespace(channel=None),
        )
        self.assertNotIn(1, bot.voice_sessions)
        sess.stop.assert_awaited_once()
        self.assertTrue(
            any(e["topic"] == "voice.left" and e["payload"].get("reason") == "channel_empty" for e in bus.recent())
        )

    def test_format_message_text_media_markers(self):
        tenor_msg = SimpleNamespace(
            content="https://tenor.com/view/cat-123",
            attachments=[],
            stickers=[],
            embeds=[SimpleNamespace(type="gifv")],
        )
        self.assertEqual(client_mod.format_message_text(tenor_msg), "[kirim GIF]")

        giphy_msg = SimpleNamespace(
            content="https://giphy.com/gifs/funny-abc",
            attachments=[],
            stickers=[],
            embeds=[],
        )
        self.assertEqual(client_mod.format_message_text(giphy_msg), "[kirim GIF]")

        img_msg = SimpleNamespace(
            content="lihat ini",
            attachments=[SimpleNamespace(filename="x.png", content_type="image/png")],
            stickers=[],
            embeds=[],
        )
        self.assertEqual(client_mod.format_message_text(img_msg), "lihat ini [kirim gambar: x.png]")

        sticker_msg = SimpleNamespace(
            content="",
            attachments=[],
            stickers=[SimpleNamespace(name="kaget")],
            embeds=[],
        )
        self.assertEqual(client_mod.format_message_text(sticker_msg), "[kirim stiker: kaget]")

    async def test_on_message_attachment_only_reply_and_non_mention(self):
        from brain.triage import TriageEngine
        from brain.profile import CharacterProfile

        prof = CharacterProfile(
            profile_id="test",
            display_name="Pachinko",
            nicknames=["pachinko"],
            wake_aliases={},
            language="id",
            interests=[],
            temperament="tsundere",
            half_life_mins=25.0,
            default_channel_mode="normal",
            filler_phrases=[],
            backchannel_words=set(),
            identity_md="",
            style_md="",
            examples_md="",
            lore_entries=[],
            kaomoji={},
        )
        brain = SimpleNamespace(
            run_text_turn=AsyncMock(return_value=SimpleNamespace(silent=False, text="Lucu!"))
        )
        store = SimpleNamespace(get_channel_mode=AsyncMock(return_value="normal"), set_channel_mode=AsyncMock())
        heart = SimpleNamespace(get_person=AsyncMock(return_value={"opt_out_approach": 0}))
        economy = SimpleNamespace(check_trivia_answer=AsyncMock(return_value=None))
        bus = EventBus()

        bot = client_mod.build_discord_bot(
            store=store,
            bus=bus,
            profiles=SimpleNamespace(current=prof, reload=AsyncMock()),
            heart=heart,
            memory=SimpleNamespace(clear_room_transcript=AsyncMock()),
            triage=TriageEngine(),
            gate=SimpleNamespace(paused=False),
            brain=brain,
            stt=SimpleNamespace(),
            tts=SimpleNamespace(synthesize_pcm=AsyncMock()),
            singing=SimpleNamespace(bind_voice_lookup=MagicMock(), snapshot=MagicMock(), skip_current=MagicMock()),
            economy=economy,
        )
        bot._connection.user = SimpleNamespace(id=999)

        class _Typing:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        ch = SimpleNamespace(id=100, typing=lambda: _Typing())
        reply_msg = SimpleNamespace(
            author=SimpleNamespace(id=42, bot=False, display_name="Budi"),
            channel=ch,
            content="",
            mentions=[],
            reference=SimpleNamespace(resolved=SimpleNamespace(author=SimpleNamespace(id=999)), message_id=1),
            attachments=[SimpleNamespace(filename="x.png", content_type="image/png")],
            stickers=[],
            embeds=[],
            reply=AsyncMock(),
        )
        await bot.on_message(reply_msg)
        brain.run_text_turn.assert_awaited_once()
        self.assertEqual(brain.run_text_turn.call_args.kwargs["user_text"], "[kirim gambar: x.png]")
        reply_msg.reply.assert_awaited_once()

        # Attachment-only message NOT mentioning/replying to Pachinko should not wake
        brain.run_text_turn.reset_mock()
        ambient_msg = SimpleNamespace(
            author=SimpleNamespace(id=42, bot=False, display_name="Budi"),
            channel=ch,
            content="",
            mentions=[],
            reference=None,
            attachments=[SimpleNamespace(filename="x.png", content_type="image/png")],
            stickers=[],
            embeds=[],
            reply=AsyncMock(),
        )
        await bot.on_message(ambient_msg)
        brain.run_text_turn.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
