import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
from fastapi.testclient import TestClient

import discord_bot.client as client_mod
from brain.profile import CharacterProfile
from brain.tools import ToolBox
from brain.triage import TriageEngine
from config import config
from core.bus import EventBus
from web.server import create_web_app


def _make_bot(*, store=None, brain=None, economy=None, singing=None, triage=None):
    bus = EventBus()
    prof = SimpleNamespace(
        display_name="Pachinko",
        nicknames=["pachinko"],
        wake_aliases={},
        default_channel_mode="mention-only",
    )
    return client_mod.build_discord_bot(
        store=store or SimpleNamespace(set_channel_mode=AsyncMock(), get_channel_mode=AsyncMock(return_value="mention-only")),
        bus=bus,
        profiles=SimpleNamespace(current=prof, reload=AsyncMock(return_value=prof)),
        heart=SimpleNamespace(get_person=AsyncMock(return_value={"opt_out_approach": 0})),
        memory=SimpleNamespace(clear_room_transcript=AsyncMock()),
        triage=triage or TriageEngine(),
        gate=SimpleNamespace(paused=False),
        brain=brain or SimpleNamespace(run_text_turn=AsyncMock(return_value=SimpleNamespace(silent=False, text="halo balik"))),
        stt=SimpleNamespace(),
        tts=SimpleNamespace(synthesize_pcm=AsyncMock(return_value=b"\x00\x01")),
        singing=singing or SimpleNamespace(
            bind_voice_lookup=MagicMock(),
            snapshot=MagicMock(return_value={"current": None, "queue": []}),
            skip_current=MagicMock(return_value=True),
        ),
        economy=economy or SimpleNamespace(
            check_trivia_answer=AsyncMock(return_value=None),
            start_trivia=MagicMock(return_value=("Q?", 10)),
        ),
    )


def _get_slash(bot, name: str):
    for cmd in bot.pending_application_commands:
        if cmd.name == name:
            return cmd
    raise KeyError(f"Command {name!r} not found")


def _dm_ctx(user_id: int = 42):
    responses = []

    async def _respond(msg, **kwargs):
        responses.append((msg, kwargs))

    return SimpleNamespace(
        author=SimpleNamespace(id=user_id, display_name="User42"),
        guild=None,
        channel=SimpleNamespace(id=100),
        respond=_respond,
        defer=AsyncMock(),
        followup=SimpleNamespace(send=AsyncMock()),
        responses=responses,
    )


class TestGuildOnlyCommands(unittest.IsolatedAsyncioTestCase):
    async def test_guild_only_commands_refuse_dm_cleanly(self):
        bot = _make_bot()
        guild_only_cmds = {
            "join": (),
            "leave": (),
            "say": ("hi",),
            "mode": ("quiet",),
            "sing": ("song",),
            "skip": (),
            "queue": (),
            "reset": (),
            "trivia": (),
        }
        for name, args in guild_only_cmds.items():
            cmd = _get_slash(bot, name)
            if client_mod.GUILDS is None:
                self.assertEqual(cmd.contexts, {discord.InteractionContextType.guild}, msg=f"/{name} missing guild context")
            ctx = _dm_ctx()
            await cmd.callback(ctx, *args)
            self.assertEqual(len(ctx.responses), 1, msg=f"/{name} did not respond")
            msg, kw = ctx.responses[0]
            self.assertEqual(msg, "Command ini cuma bisa dipakai di server.")
            self.assertTrue(kw.get("ephemeral"), msg=f"/{name} response not ephemeral")


class TestFitDiscord(unittest.TestCase):
    def test_fit_discord_boundaries(self):
        from core.safety import fit_discord

        self.assertEqual(fit_discord("halo dunia"), "halo dunia")

        words = "kata " * 1000
        out_words = fit_discord(words)
        self.assertLessEqual(len(out_words), 2000)
        self.assertTrue(out_words.endswith("…"))
        self.assertTrue(out_words[:-1].endswith("kata"))

        lines = ("baris satu\n" * 300).strip()
        out_lines = fit_discord(lines)
        self.assertLessEqual(len(out_lines), 2000)
        self.assertTrue(out_lines.endswith("…"))
        self.assertTrue(out_lines[:-1].endswith(("satu", "baris")))

        unbroken = "a" * 5000
        out_unbroken = fit_discord(unbroken)
        self.assertEqual(len(out_unbroken), 2000)
        self.assertTrue(out_unbroken.endswith("…"))


class TestWebInputValidation(unittest.TestCase):
    def setUp(self):
        self._orig_token = config.panel_token
        self._orig_host = config.panel_host
        self._orig_port = config.panel_port
        config.panel_token = ""
        config.panel_host = "127.0.0.1"
        config.panel_port = 7878

    def tearDown(self):
        config.panel_token = self._orig_token
        config.panel_host = self._orig_host
        config.panel_port = self._orig_port

    def _make_client(self):
        kv = {"persona_overrides": {"display_name": "Pachinko"}, "owner_directives": "orig"}
        prof = SimpleNamespace(
            profile_id="default",
            display_name="Pachinko",
            temperament="tsundere",
            default_channel_mode="quiet",
            identity_md="",
            style_md="",
            examples_md="",
            lore_entries=[],
        )

        async def _get_kv(k, default=None):
            return kv.get(k, default)

        async def _set_kv(k, v):
            kv[k] = v

        async def _save_overrides(data):
            cur = dict(kv["persona_overrides"])
            cur.update({k: v for k, v in data.items() if v is not None})
            kv["persona_overrides"] = cur
            if "display_name" in cur:
                prof.display_name = cur["display_name"]
            return prof

        heart = SimpleNamespace(
            state_summary=lambda _t: {"dominant": "neutral", "valence": 0.0, "energy": 0.8, "active_emotions": {}},
            get_person=AsyncMock(return_value={"user_id": "123"}),
        )
        store = SimpleNamespace(
            get_kv=_get_kv,
            set_kv=_set_kv,
            write_tx=MagicMock(),
        )
        memory = SimpleNamespace(upsert_note=AsyncMock(), remember=AsyncMock())
        gate = SimpleNamespace(paused=False, set_paused=AsyncMock())
        app = create_web_app(
            store=store,
            bus=EventBus(),
            profiles=SimpleNamespace(current=prof, save_overrides=_save_overrides),
            heart=heart,
            memory=memory,
            gate=gate,
            brain=SimpleNamespace(),
            singing=SimpleNamespace(snapshot=lambda: {"current": None, "queue": []}),
        )
        return TestClient(app, base_url="http://127.0.0.1:7878"), kv, heart, store, memory, gate

    def test_persona_validation_rejects_bad_payloads_without_saving(self):
        client, kv, _, _, _, _ = self._make_client()
        bad_payloads = [
            {"directives": "should not save", "lore_entries": {}},
            {"directives": "should not save", "lore_entries": "x"},
            {"directives": "should not save", "lore_entries": [{"content": 5}]},
            {"directives": "should not save", "lore_entries": [{"keywords": "not-list", "content": "ok"}]},
            {"directives": "should not save", "display_name": "x" * 33},
            {"directives": "should not save", "identity_md": "x" * 20001},
        ]
        for payload in bad_payloads:
            res = client.post("/api/persona", json=payload)
            self.assertEqual(res.status_code, 400, msg=f"Expected 400 for {payload!r}, got {res.status_code}")
            self.assertEqual(kv["persona_overrides"], {"display_name": "Pachinko"})
            self.assertEqual(kv["owner_directives"], "orig")

        ok = client.post(
            "/api/persona",
            json={"display_name": "Pachi", "lore_entries": [{"keywords": ["kopi"], "content": "Suka kopi"}]},
        )
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(kv["persona_overrides"]["display_name"], "Pachi")

    def test_people_validation(self):
        client, _, heart, store, _, _ = self._make_client()
        for uid in ("abc", "123a"):
            res = client.post(f"/api/people/{uid}", json={"bond": 10})
            self.assertEqual(res.status_code, 400)
        for bad_bond in ("abc", "NaN", "inf", True):
            res = client.post("/api/people/12345", json={"bond": bad_bond})
            self.assertEqual(res.status_code, 400)
        heart.get_person.assert_not_awaited()
        store.write_tx.assert_not_called()

    def test_memory_and_runtime_validation(self):
        client, _, _, _, memory, gate = self._make_client()
        for bad_mem in ({"fact": ""}, {"fact": 123}, {"fact": "ok", "doc_key": 123}):
            self.assertEqual(client.post("/api/memory", json=bad_mem).status_code, 400)
        memory.upsert_note.assert_not_awaited()

        for bad_rt in ({"autonomy_paused": "yes"}, {"rvc_enabled": 1}, {"llm_provider": ""}):
            self.assertEqual(client.post("/api/runtime", json=bad_rt).status_code, 400)
        gate.set_paused.assert_not_awaited()


class TestProfileAndToolsGuards(unittest.IsolatedAsyncioTestCase):
    def test_matches_lore_skips_malformed_entries(self):
        prof = CharacterProfile(
            profile_id="default",
            display_name="Pachinko",
            nicknames=["pachinko"],
            wake_aliases={},
            language="id",
            interests=[],
            temperament="tsundere",
            half_life_mins=25.0,
            default_channel_mode="quiet",
            filler_phrases=[],
            backchannel_words=set(),
            identity_md="",
            style_md="",
            examples_md="",
            lore_entries=[
                "not-a-dict",
                {"keywords": "str", "content": "bad"},
                {"keywords": [123, "kopi"], "content": 5},
                {"keywords": ["kopi"], "content": "Suka kopi hitam", "priority": "bad"},
            ],
        )
        self.assertEqual(prof.matches_lore("aku suka kopi"), ["Suka kopi hitam"])

    async def test_set_reminder_rejects_non_finite_minutes(self):
        memory = SimpleNamespace(add_schedule=AsyncMock())
        gate = SimpleNamespace(check=AsyncMock(return_value=(True, "ok")))
        toolbox = ToolBox(memory=memory, gate=gate)
        for bad in ("abc", float("nan"), "inf", None, True):
            out = await toolbox.execute(
                "set_reminder",
                {"minutes": bad, "note": "x"},
                room_id="100",
                user_id="42",
                user_name="Budi",
            )
            self.assertIsInstance(out, str)
            self.assertIn("invalid", out.lower())
        memory.add_schedule.assert_not_awaited()


class TestQuietReplies(unittest.IsolatedAsyncioTestCase):
    async def test_reply_to_bot_without_ping_wakes_bot(self):
        brain = SimpleNamespace(run_text_turn=AsyncMock(return_value=SimpleNamespace(silent=False, text="halo balik")))
        bot = _make_bot(brain=brain)
        bot._connection.user = SimpleNamespace(id=999)

        # Case A: cached resolved reference
        msg_a = SimpleNamespace(
            author=SimpleNamespace(id=42, display_name="Budi", bot=False),
            channel=SimpleNamespace(id=100, typing=MagicMock(return_value=AsyncMock())),
            content="iya bener",
            mentions=[],
            reference=SimpleNamespace(resolved=SimpleNamespace(author=SimpleNamespace(id=999)), message_id=55),
            reply=AsyncMock(),
        )
        await bot.on_message(msg_a)
        brain.run_text_turn.assert_awaited_once()
        msg_a.reply.assert_awaited_once()

        # Case B: uncached reference fetched via channel.fetch_message
        brain.run_text_turn.reset_mock()
        ch_b = SimpleNamespace(
            id=100,
            typing=MagicMock(return_value=AsyncMock()),
            fetch_message=AsyncMock(return_value=SimpleNamespace(author=SimpleNamespace(id=999))),
        )
        msg_b = SimpleNamespace(
            author=SimpleNamespace(id=42, display_name="Budi", bot=False),
            channel=ch_b,
            content="iya bener",
            mentions=[],
            reference=SimpleNamespace(resolved=None, message_id=55),
            reply=AsyncMock(),
        )
        await bot.on_message(msg_b)
        brain.run_text_turn.assert_awaited_once()
        msg_b.reply.assert_awaited_once()

        # Case C: fetch_message fails -> ignored cleanly, no crash
        brain.run_text_turn.reset_mock()
        ch_c = SimpleNamespace(
            id=100,
            typing=MagicMock(return_value=AsyncMock()),
            fetch_message=AsyncMock(side_effect=RuntimeError("boom")),
        )
        msg_c = SimpleNamespace(
            author=SimpleNamespace(id=42, display_name="Budi", bot=False),
            channel=ch_c,
            content="iya bener",
            mentions=[],
            reference=SimpleNamespace(resolved=None, message_id=55),
            reply=AsyncMock(),
        )
        await bot.on_message(msg_c)
        brain.run_text_turn.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
