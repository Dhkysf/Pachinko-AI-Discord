import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from config import config
from core.bus import EventBus
from web.server import create_web_app


def _make_app():
    bus = EventBus()
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
    async def _upsert_note(*_args, **_kwargs):
        return 1

    async def _remember(*_args, **_kwargs):
        return 1

    async def _delete_ids(ids):
        return len(ids)

    return create_web_app(
        store=SimpleNamespace(),
        bus=bus,
        profiles=SimpleNamespace(current=prof),
        heart=SimpleNamespace(state_summary=lambda _t: {"dominant": "neutral", "valence": 0.0, "energy": 0.8, "active_emotions": {}}),
        memory=SimpleNamespace(upsert_note=_upsert_note, remember=_remember, delete_notes=_delete_ids, delete_memories=_delete_ids),
        gate=SimpleNamespace(paused=False),
        brain=SimpleNamespace(),
        singing=SimpleNamespace(snapshot=lambda: {"current": None, "queue": []}, skip_current=lambda: True),
    )


class TestPanelSecurity(unittest.TestCase):
    def setUp(self):
        self._orig_token = config.panel_token
        self._orig_host = config.panel_host
        self._orig_port = config.panel_port
        config.panel_host = "127.0.0.1"
        config.panel_port = 7878

    def tearDown(self):
        config.panel_token = self._orig_token
        config.panel_host = self._orig_host
        config.panel_port = self._orig_port

    def _client(self, app):
        return TestClient(app, base_url=f"http://127.0.0.1:{config.panel_port}")

    def test_ws_live_requires_token_when_set(self):
        config.panel_token = "secret-tok"
        client = self._client(_make_app())
        host_hdr = {"Host": "127.0.0.1:7878"}

        with self.assertRaises(WebSocketDisconnect) as ctx1:
            with client.websocket_connect("/ws/live", headers=host_hdr):
                pass
        self.assertEqual(ctx1.exception.code, 1008)

        with self.assertRaises(WebSocketDisconnect) as ctx2:
            with client.websocket_connect("/ws/live?token=wrong", headers=host_hdr):
                pass
        self.assertEqual(ctx2.exception.code, 1008)

        with client.websocket_connect("/ws/live?token=secret-tok", headers=host_hdr):
            pass

    def test_ws_live_rejects_foreign_origin_even_without_token(self):
        config.panel_token = ""
        client = self._client(_make_app())

        with self.assertRaises(WebSocketDisconnect) as ctx:
            with client.websocket_connect(
                "/ws/live",
                headers={"Host": "127.0.0.1:7878", "Origin": "http://evil.example"},
            ):
                pass
        self.assertEqual(ctx.exception.code, 1008)

    def test_ws_live_allows_same_origin_and_no_origin(self):
        config.panel_token = ""
        client = self._client(_make_app())

        with client.websocket_connect(
            "/ws/live",
            headers={"Host": "127.0.0.1:7878", "Origin": "http://127.0.0.1:7878"},
        ):
            pass
        with client.websocket_connect(
            "/ws/live",
            headers={"Host": "localhost:7878", "Origin": "http://localhost:7878"},
        ):
            pass
        with client.websocket_connect("/ws/live", headers={"Host": "127.0.0.1:7878"}):
            pass

    def test_wrong_host_header_returns_400(self):
        config.panel_token = ""
        client = self._client(_make_app())
        res = client.get("/api/status", headers={"Host": "evil.example:7878"})
        self.assertEqual(res.status_code, 400)

    def test_api_status_auth(self):
        config.panel_token = "secret-tok"
        client = self._client(_make_app())

        self.assertEqual(client.get("/api/status").status_code, 401)
        self.assertEqual(
            client.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code,
            401,
        )
        self.assertEqual(
            client.get("/api/status", headers={"Authorization": "Bearer secret-tok"}).status_code,
            200,
        )

    def test_docs_disabled_when_token_set(self):
        config.panel_token = "secret-tok"
        client = self._client(_make_app())
        self.assertEqual(client.get("/api/docs").status_code, 404)
        self.assertEqual(client.get("/openapi.json").status_code, 404)

        config.panel_token = ""
        client_open = self._client(_make_app())
        self.assertEqual(client_open.get("/api/docs").status_code, 200)

    def test_api_status_active_llm_model(self):
        config.panel_token = ""
        orig_provider = config.llm_provider
        try:
            client = self._client(_make_app())
            for provider, expected in (
                ("ollama", config.ollama_model),
                ("ninerouter", config.ninerouter_model),
                ("openai_compat", config.openai_model),
            ):
                config.llm_provider = provider
                res = client.get("/api/status")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["runtime"]["llm_model"], expected)
                self.assertNotIn("pending_approvals", data["autonomy"])
        finally:
            config.llm_provider = orig_provider

    def test_mutating_api_rejects_foreign_origin_even_without_token(self):
        config.panel_token = ""
        client = self._client(_make_app())
        payload = {"fact": "Suka kopi", "doc_key": "self"}

        self.assertEqual(
            client.post("/api/memory", json=payload, headers={"Origin": "http://evil.example"}).status_code,
            403,
        )
        self.assertEqual(
            client.post("/api/memory", json=payload, headers={"Origin": "null"}).status_code,
            403,
        )
        self.assertEqual(
            client.post("/api/memory", json=payload, headers={"Origin": "http://127.0.0.1:7878"}).status_code,
            200,
        )
        self.assertEqual(
            client.post(
                "/api/memory",
                json=payload,
                headers={"Host": "localhost:7878", "Origin": "http://localhost:7878"},
            ).status_code,
            200,
        )
        self.assertEqual(client.post("/api/memory", json=payload).status_code, 200)
        self.assertEqual(
            client.post("/api/sing/skip", headers={"Origin": "http://evil.example"}).status_code,
            403,
        )
        self.assertEqual(client.post("/api/sing/skip").status_code, 200)

    def test_delete_memory_auth_validation_and_origin(self):
        config.panel_token = "secret-tok"
        client = self._client(_make_app())
        valid = {"notebook_ids": [1, 2], "memory_ids": [3]}

        self.assertEqual(client.post("/api/memory/delete", json=valid).status_code, 401)
        self.assertEqual(
            client.post("/api/memory/delete", json=valid, headers={"Authorization": "Bearer wrong"}).status_code,
            401,
        )
        res = client.post("/api/memory/delete", json=valid, headers={"Authorization": "Bearer secret-tok"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"ok": True, "notebook_deleted": 2, "memories_deleted": 1})

        config.panel_token = ""
        client = self._client(_make_app())
        for bad in (
            {},
            {"notebook_ids": [], "memory_ids": []},
            {"notebook_ids": ["1"]},
            {"notebook_ids": [True]},
            {"memory_ids": [1.5]},
            {"notebook_ids": None, "memory_ids": [1]},
            {"notebook_ids": list(range(1, 202))},
            {"notebook_ids": list(range(1, 102)), "memory_ids": list(range(1, 101))},
        ):
            self.assertEqual(client.post("/api/memory/delete", json=bad).status_code, 400, f"Expected 400 for {bad!r}")

        self.assertEqual(client.post("/api/memory/delete", json={"notebook_ids": list(range(1, 201))}).status_code, 200)
        self.assertEqual(
            client.post("/api/memory/delete", json=valid, headers={"Origin": "http://evil.example"}).status_code,
            403,
        )
        self.assertEqual(
            client.post("/api/memory/delete", json=valid, headers={"Origin": "http://127.0.0.1:7878"}).status_code,
            200,
        )
        self.assertEqual(client.post("/api/memory/delete", json=valid).status_code, 200)


if __name__ == "__main__":
    unittest.main()
