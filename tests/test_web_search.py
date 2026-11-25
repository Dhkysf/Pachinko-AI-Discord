"""Offline unit tests for web_search (9router /v1/search + ddgs fallback)."""
from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from brain.tools import ToolBox
from config import AppConfig


def _make_cfg(**overrides: object) -> AppConfig:
    with patch.dict(os.environ, {}, clear=True):
        cfg = AppConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_toolbox(transport: httpx.AsyncBaseTransport | None = None) -> ToolBox:
    return ToolBox(
        memory=SimpleNamespace(),
        gate=SimpleNamespace(),
        http_transport=transport,
    )


class TestWebSearch(unittest.IsolatedAsyncioTestCase):
    async def test_ninerouter_happy_path(self) -> None:
        seen: list[tuple[str, str, str, dict[str, object]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            seen.append((
                request.method,
                str(request.url),
                request.headers.get("authorization", ""),
                body,
            ))
            return httpx.Response(
                200,
                json={
                    "provider": "serper",
                    "query": "Honkai Star Rail Firefly",
                    "results": [
                        {"title": f"Title {i}", "snippet": f"Snippet {i}", "url": f"https://example.com/{i}"}
                        for i in range(1, 6)
                    ],
                },
            )

        cfg = _make_cfg(
            web_search_provider="ninerouter",
            web_search_ninerouter_provider="serper",
            ninerouter_base_url="http://127.0.0.1:20128/v1/",
            ninerouter_api_key="nr-secret-key-123",
        )
        tb = _make_toolbox(httpx.MockTransport(handler))
        with patch("brain.tools.config", cfg), patch("brain.tools._ddgs_search") as ddgs_mock:
            out = await tb._web_search("Honkai Star Rail Firefly")

        ddgs_mock.assert_not_called()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "POST")
        self.assertEqual(seen[0][1], "http://127.0.0.1:20128/v1/search")
        self.assertEqual(seen[0][2], "Bearer nr-secret-key-123")
        self.assertEqual(
            seen[0][3],
            {
                "query": "Honkai Star Rail Firefly",
                "max_results": 4,
                "country": "id",
                "language": "id",
                "provider": "serper",
            },
        )
        self.assertTrue(out.startswith('<data source="web_search" title="Honkai Star Rail Firefly">'))
        self.assertIn("- Title 1: Snippet 1 (https://example.com/1)", out)
        self.assertIn("- Title 4: Snippet 4 (https://example.com/4)", out)
        self.assertNotIn("Title 5", out)

    async def test_ninerouter_omits_provider_when_empty(self) -> None:
        seen_bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_bodies.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"results": [{"title": "T", "snippet": "S", "url": "https://x.id"}]})

        cfg = _make_cfg(
            web_search_provider="ninerouter",
            web_search_ninerouter_provider="",
            ninerouter_base_url="http://127.0.0.1:20128/v1",
            ninerouter_api_key="",
        )
        tb = _make_toolbox(httpx.MockTransport(handler))
        with patch("brain.tools.config", cfg), patch("brain.tools._ddgs_search") as ddgs_mock:
            out = await tb._web_search("berita")

        ddgs_mock.assert_not_called()
        self.assertNotIn("provider", seen_bodies[0])
        self.assertIn("- T: S (https://x.id)", out)

    async def test_ninerouter_401_falls_back_to_ddgs_once_without_leaking_key(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "Invalid key sk-secret-token-12345678901234"}})

        cfg = _make_cfg(
            web_search_provider="ninerouter",
            web_search_ninerouter_provider="serper",
            ninerouter_api_key="sk-secret-token-12345678901234",
        )
        tb = _make_toolbox(httpx.MockTransport(handler))
        ddgs_items = [{"title": "DDG", "body": "Fallback", "href": "https://ddg.gg"}]
        with (
            patch("brain.tools.config", cfg),
            patch("brain.tools._ddgs_search", return_value=ddgs_items) as ddgs_mock,
            self.assertLogs("brain.tools", level="WARNING") as logs,
        ):
            out = await tb._web_search("firefly")

        ddgs_mock.assert_called_once_with("firefly")
        self.assertIn("- DDG: Fallback (https://ddg.gg)", out)
        self.assertEqual(len(logs.output), 1)
        self.assertNotIn("sk-secret-token-12345678901234", logs.output[0])

    async def test_ninerouter_429_and_400_fall_back_to_ddgs_once(self) -> None:
        for status in (429, 400):
            def handler(request: httpx.Request, st: int = status) -> httpx.Response:
                return httpx.Response(st, json={"error": {"message": f"err {st}"}})

            cfg = _make_cfg(web_search_provider="ninerouter", web_search_ninerouter_provider="serper")
            tb = _make_toolbox(httpx.MockTransport(handler))
            ddgs_items = [{"title": "DDG", "body": "Body", "href": "https://ddg.gg"}]
            with (
                patch("brain.tools.config", cfg),
                patch("brain.tools._ddgs_search", return_value=ddgs_items) as ddgs_mock,
                self.assertLogs("brain.tools", level="WARNING") as logs,
            ):
                out = await tb._web_search("query")

            ddgs_mock.assert_called_once_with("query")
            self.assertIn("- DDG: Body (https://ddg.gg)", out)
            self.assertEqual(len(logs.output), 1)

    async def test_ninerouter_connect_error_falls_back_to_ddgs_once(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        cfg = _make_cfg(web_search_provider="ninerouter", web_search_ninerouter_provider="serper")
        tb = _make_toolbox(httpx.MockTransport(handler))
        ddgs_items = [{"title": "DDG", "body": "Offline", "href": "https://ddg.gg"}]
        with (
            patch("brain.tools.config", cfg),
            patch("brain.tools._ddgs_search", return_value=ddgs_items) as ddgs_mock,
            self.assertLogs("brain.tools", level="WARNING") as logs,
        ):
            out = await tb._web_search("firefly")

        ddgs_mock.assert_called_once_with("firefly")
        self.assertIn("- DDG: Offline (https://ddg.gg)", out)
        self.assertEqual(len(logs.output), 1)

    async def test_ninerouter_empty_results_falls_back_to_ddgs_once(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": []})

        cfg = _make_cfg(web_search_provider="ninerouter", web_search_ninerouter_provider="serper")
        tb = _make_toolbox(httpx.MockTransport(handler))
        ddgs_items = [{"title": "DDG", "body": "Found", "href": "https://ddg.gg"}]
        with (
            patch("brain.tools.config", cfg),
            patch("brain.tools._ddgs_search", return_value=ddgs_items) as ddgs_mock,
            self.assertLogs("brain.tools", level="WARNING") as logs,
        ):
            out = await tb._web_search("firefly")

        ddgs_mock.assert_called_once_with("firefly")
        self.assertIn("- DDG: Found (https://ddg.gg)", out)
        self.assertEqual(len(logs.output), 1)

    async def test_default_ddgs_path_unchanged(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("HTTP transport must not be called when provider is ddgs")

        cfg = _make_cfg(web_search_provider="ddgs")
        tb = _make_toolbox(httpx.MockTransport(handler))
        ddgs_items = [{"title": "T1", "body": "B1", "href": "https://example.com"}]
        with patch("brain.tools.config", cfg), patch("brain.tools._ddgs_search", return_value=ddgs_items) as ddgs_mock:
            out = await tb._web_search("test query")

        ddgs_mock.assert_called_once_with("test query")
        self.assertEqual(
            out,
            '<data source="web_search" title="test query">\n- T1: B1 (https://example.com)\n</data>',
        )

    async def test_empty_query_returns_empty_query(self) -> None:
        tb = _make_toolbox()
        self.assertEqual(await tb._web_search("   "), "Empty query.")
