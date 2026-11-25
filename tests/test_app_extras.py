"""Offline unit tests for app/extras.py, OLLAMA_MODEL override, backup rotation, and diagnostics redaction."""
from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

from app.extras import (
    backup_database,
    build_diagnostics,
    fetch_gemini_models,
    fetch_ninerouter_models,
    fetch_ollama_models,
    model_is_installed,
    ninerouter_dashboard_url,
    run_preflight,
    warm_up_ollama,
)
from app.process import (
    STATE_RUNNING,
    STATE_STOPPED,
    BotProcessManager,
    BotStatus,
    load_settings,
    save_settings,
)
from app.tray import format_tooltip


class _FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class TestAppExtras(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.repo_root = Path(__file__).resolve().parent.parent

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_preflight_missing_env_empty_token_ollama_down_and_missing_model(self) -> None:
        # 1. Missing .env + missing ffmpeg + Ollama down + busy port
        res, models = run_preflight(
            self.tmp_dir,
            which=lambda _name: None,
            fetch_models=lambda _url: (False, [], "connection refused"),
            port_free=lambda _host, _port: False,
        )
        by_name = {r.name: r for r in res}
        self.assertEqual(models, [])
        self.assertFalse(by_name[".env & DISCORD_TOKEN"].ok)
        self.assertIn("missing", by_name[".env & DISCORD_TOKEN"].detail.lower())
        self.assertFalse(by_name["ffmpeg"].ok)
        self.assertFalse(by_name["Ollama"].ok)
        self.assertEqual(by_name["Ollama"].code, "ollama_down")
        self.assertFalse(by_name["Panel port 7878"].ok)

        # 2. Empty DISCORD_TOKEN + model missing
        (self.tmp_dir / ".env").write_text(
            "DISCORD_TOKEN=\nOLLAMA_MODEL=gemma3:12b\n",
            encoding="utf-8",
        )
        res2, models2 = run_preflight(
            self.tmp_dir,
            which=lambda _name: "C:\\ffmpeg\\ffmpeg.exe",
            fetch_models=lambda _url: (True, ["qwen3:8b"], ""),
            port_free=lambda _host, _port: True,
        )
        by_name2 = {r.name: r for r in res2}
        self.assertEqual(models2, ["qwen3:8b"])
        self.assertFalse(by_name2[".env & DISCORD_TOKEN"].ok)
        self.assertIn("empty", by_name2[".env & DISCORD_TOKEN"].detail.lower())
        self.assertTrue(by_name2["ffmpeg"].ok)
        self.assertFalse(by_name2["Ollama"].ok)
        self.assertIn("gemma3:12b", by_name2["Ollama"].detail)
        self.assertTrue(by_name2["Panel port 7878"].ok)

        # 3. Valid .env: secret must never appear in preflight output, implicit :latest matches
        secret = "M" + ("A" * 24) + "." + ("B" * 6) + "." + ("C" * 28)
        (self.tmp_dir / ".env").write_text(
            f"DISCORD_TOKEN={secret}\nOLLAMA_MODEL=qwen3\n",
            encoding="utf-8",
        )
        res3, _ = run_preflight(
            self.tmp_dir,
            bot_running=True,
            which=lambda _name: "ffmpeg",
            fetch_models=lambda _url: (True, ["qwen3:latest"], ""),
            port_free=lambda _host, _port: False,
        )
        self.assertTrue(all(r.ok for r in res3))
        joined = "\n".join(f"{r.name} {r.detail} {r.hint}" for r in res3)
        self.assertNotIn(secret, joined)

    def test_preflight_skips_ollama_when_non_ollama_provider(self) -> None:
        (self.tmp_dir / ".env").write_text(
            "DISCORD_TOKEN=abc123\nLLM_PROVIDER=gemini\n",
            encoding="utf-8",
        )
        res, _ = run_preflight(
            self.tmp_dir,
            which=lambda _name: "ffmpeg",
            fetch_models=lambda _url: (_ for _ in ()).throw(AssertionError("should not probe")),
            port_free=lambda _host, _port: True,
        )
        by_name = {r.name: r for r in res}
        self.assertNotIn("Ollama", by_name)
        self.assertIn("Gemini", by_name)
        self.assertFalse(by_name["Gemini"].ok)

    def test_model_is_installed(self) -> None:
        self.assertTrue(model_is_installed("qwen3:8b", ["qwen3:8b"]))
        self.assertTrue(model_is_installed("qwen3", ["qwen3:latest"]))
        self.assertFalse(model_is_installed("gemma3:12b", ["qwen3:8b"]))

    def test_ollama_model_env_var_overrides_dotenv_in_subprocess(self) -> None:
        (self.tmp_dir / ".env").write_text("OLLAMA_MODEL=from-dotenv\n", encoding="utf-8")
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(self.repo_root)!r}); "
            "from config import config; "
            "print(config.ollama_model)"
        )
        base_env = os.environ.copy()
        base_env.pop("OLLAMA_MODEL", None)

        r_dotenv = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(self.tmp_dir),
            env=base_env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(r_dotenv.stdout.strip(), "from-dotenv")

        override_env = dict(base_env, OLLAMA_MODEL="from-env-override")
        r_override = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(self.tmp_dir),
            env=override_env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(r_override.stdout.strip(), "from-env-override")

    def test_process_manager_passes_llm_env_per_source_and_use_env(self) -> None:
        out_file = self.tmp_dir / "env_out.json"
        script = self.tmp_dir / "print_env.py"
        script.write_text(
            "import json, os\n"
            "from pathlib import Path\n"
            "data = {k: os.environ.get(k) for k in ('LLM_PROVIDER', 'OLLAMA_MODEL', 'NINEROUTER_MODEL', 'GEMINI_MODEL')}\n"
            f"Path({str(out_file)!r}).write_text(json.dumps(data), encoding='utf-8')\n",
            encoding="utf-8",
        )
        cases = [
            ("ollama", "gemma3:12b", {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "gemma3:12b", "NINEROUTER_MODEL": None, "GEMINI_MODEL": None}),
            ("ninerouter", "openai/gpt-4o", {"LLM_PROVIDER": "ninerouter", "OLLAMA_MODEL": None, "NINEROUTER_MODEL": "openai/gpt-4o", "GEMINI_MODEL": None}),
            ("gemini", "gemini-2.5-flash", {"LLM_PROVIDER": "gemini", "OLLAMA_MODEL": None, "NINEROUTER_MODEL": None, "GEMINI_MODEL": "gemini-2.5-flash"}),
            ("", "", {"LLM_PROVIDER": None, "OLLAMA_MODEL": None, "NINEROUTER_MODEL": None, "GEMINI_MODEL": None}),
        ]
        for src, mdl, expected in cases:
            out_file.unlink(missing_ok=True)
            mgr = BotProcessManager(
                self.repo_root,
                data_dir=self.tmp_dir,
                bot_cmd=[sys.executable, str(script)],
                llm_source=src,
                llm_model=mdl,
            )
            self.assertTrue(mgr.start())
            rc = mgr._proc.wait(timeout=5.0)  # type: ignore[union-attr]
            self.assertEqual(rc, 0)
            got = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(got, expected, f"failed for {(src, mdl)}")

    def test_backup_database_with_open_wal_writer_and_rotation(self) -> None:
        db_path = self.tmp_dir / "pachinko.db"
        backups_dir = self.tmp_dir / "backups"
        backups_dir.mkdir()

        # Create a WAL-mode DB and keep an active writer connection open.
        writer = sqlite3.connect(str(db_path))
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        writer.execute("INSERT INTO notes (body) VALUES ('committed')")
        writer.commit()
        writer.execute("INSERT INTO notes (body) VALUES ('uncommitted')")

        try:
            # Plant 16 matching backups and 1 unrelated file.
            base = datetime(2026, 11, 25, 6, 0, 0)
            for i in range(16):
                ts = (base + timedelta(minutes=i)).strftime("%Y%m%d-%H%M%S")
                (backups_dir / f"pachinko-{ts}.db").write_bytes(b"old")
            unrelated = backups_dir / "manual-keep.db"
            unrelated.write_bytes(b"keep-me")

            created = backup_database(
                db_path,
                backups_dir,
                keep=14,
                now=datetime(2026, 11, 25, 7, 0, 0),
            )
            self.assertEqual(created.name, "pachinko-20261125-070000.db")
            self.assertTrue(created.exists())
            self.assertTrue(unrelated.exists())

            matching = sorted(p.name for p in backups_dir.glob("pachinko-*.db"))
            self.assertEqual(len(matching), 14)
            self.assertIn("pachinko-20261125-070000.db", matching)

            # Verify backup integrity and committed row.
            ro = sqlite3.connect(str(created))
            try:
                rows = ro.execute("SELECT body FROM notes").fetchall()
                self.assertEqual(rows, [("committed",)])
            finally:
                ro.close()
        finally:
            writer.rollback()
            writer.close()

    def test_diagnostics_redacts_planted_secrets(self) -> None:
        log_path = self.tmp_dir / "pachinko.log"
        fake_discord = "M" + ("A" * 24) + "." + ("B" * 6) + "." + ("C" * 28)
        fake_sk = "sk-" + ("Z" * 24)
        fake_env = "DISCORD_TOKEN=super_secret_value_12345"
        log_path.write_text(
            f"2026-11-25 [INFO] test: tok={fake_discord}\n"
            f"2026-11-25 [INFO] test: key={fake_sk}\n"
            f"2026-11-25 [INFO] test: {fake_env}\n",
            encoding="utf-8",
        )
        diag = build_diagnostics(
            status=BotStatus(
                state=STATE_RUNNING,
                pid=1234,
                uptime=42,
                discord_ready=True,
                llm_ok=True,
                llm_provider="ninerouter",
                llm_model="openai/gpt-4o",
            ),
            selected_source="ninerouter",
            selected_model="openai/gpt-4o",
            env_source="ollama",
            env_model="qwen3:8b",
            preflight=[],
            log_path=log_path,
        )
        self.assertNotIn(fake_discord, diag)
        self.assertNotIn(fake_sk, diag)
        self.assertNotIn("super_secret_value_12345", diag)
        self.assertIn("[REDACTED]", diag)
        self.assertIn("ninerouter", diag)
        self.assertIn("openai/gpt-4o", diag)

    def test_warm_up_ollama_posts_generate_payload(self) -> None:
        captured: dict[str, object] = {}

        def fake_opener(req, timeout=0):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse(b'{"done": true}', status=200)

        ok = warm_up_ollama("http://127.0.0.1:11434/", "qwen3:8b", "15m", opener=fake_opener)
        self.assertTrue(ok)
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/generate")
        self.assertEqual(
            captured["body"],
            {"model": "qwen3:8b", "prompt": "", "stream": False, "keep_alive": "15m"},
        )

    def test_settings_migration_and_per_source_models(self) -> None:
        cfg_path = self.tmp_dir / "app_settings.json"
        # Legacy ollama_model migrates into model_ollama
        cfg_path.write_text(json.dumps({"ollama_model": "gemma3:12b", "start_bot_on_open": True}), encoding="utf-8")
        migrated = load_settings(cfg_path)
        self.assertEqual(migrated["model_ollama"], "gemma3:12b")
        self.assertEqual(migrated["llm_source"], "")
        self.assertTrue(migrated["start_bot_on_open"])

        # Explicit empty model_ollama is not overwritten by legacy ollama_model
        cfg_path.write_text(json.dumps({"ollama_model": "gemma3:12b", "model_ollama": ""}), encoding="utf-8")
        self.assertEqual(load_settings(cfg_path)["model_ollama"], "")

        save_settings(
            cfg_path,
            {
                "llm_source": "ninerouter",
                "model_ollama": "qwen3:8b",
                "model_ninerouter": "openai/gpt-4o",
                "model_gemini": "gemini-2.5-flash",
            },
        )
        loaded = load_settings(cfg_path)
        self.assertEqual(loaded["llm_source"], "ninerouter")
        self.assertEqual(loaded["model_ollama"], "qwen3:8b")
        self.assertEqual(loaded["model_ninerouter"], "openai/gpt-4o")
        self.assertEqual(loaded["model_gemini"], "gemini-2.5-flash")

    def test_fetch_models_all_sources_empty_and_401(self) -> None:
        # Ollama
        ok, models, err = fetch_ollama_models(
            "http://127.0.0.1:11434",
            opener=lambda _req, timeout=0: _FakeHTTPResponse(
                json.dumps({"models": [{"name": "qwen3:8b"}, {"name": "gemma3:12b"}]}).encode("utf-8")
            ),
        )
        self.assertTrue(ok)
        self.assertEqual(models, ["qwen3:8b", "gemma3:12b"])
        self.assertEqual(err, "")

        # 9router: sorted ids, Bearer header, empty list, 401, connection refused
        captured_headers: dict[str, str] = {}

        def nr_ok(req, timeout=0):
            captured_headers.update(dict(req.header_items()))
            body = json.dumps({"data": [{"id": "z-model"}, {"id": "a-model"}, {"id": "a-model"}]}).encode("utf-8")
            return _FakeHTTPResponse(body)

        ok_nr, models_nr, err_nr = fetch_ninerouter_models("http://127.0.0.1:20128/v1", "nr-key-123", opener=nr_ok)
        self.assertTrue(ok_nr)
        self.assertEqual(models_nr, ["a-model", "z-model"])
        self.assertEqual(err_nr, "")
        self.assertEqual(captured_headers.get("Authorization"), "Bearer nr-key-123")

        ok_empty, models_empty, err_empty = fetch_ninerouter_models(
            "http://127.0.0.1:20128/v1",
            "nr-key-123",
            opener=lambda _req, timeout=0: _FakeHTTPResponse(b'{"data": []}'),
        )
        self.assertTrue(ok_empty)
        self.assertEqual(models_empty, [])
        self.assertIn("No models found - connect a provider in the 9router dashboard", err_empty)

        def nr_401(req, timeout=0):
            raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b""))

        ok_401, _, err_401 = fetch_ninerouter_models("http://127.0.0.1:20128/v1", "bad-key", opener=nr_401)
        self.assertFalse(ok_401)
        self.assertEqual(err_401, "API key missing or wrong")

        def nr_down(_req, timeout=0):
            raise urllib.error.URLError("connection refused")

        ok_down, _, err_down = fetch_ninerouter_models("http://127.0.0.1:20128/v1", "nr-key-123", opener=nr_down)
        self.assertFalse(ok_down)
        self.assertIn("9router not running at", err_down)

        # Gemini: generateContent filter, short names, x-goog-api-key header (not in URL), empty, 401
        gem_req: dict[str, object] = {}

        def gem_ok(req, timeout=0):
            gem_req["url"] = req.full_url
            gem_req["headers"] = {k.lower(): v for k, v in req.header_items()}
            payload = {
                "models": [
                    {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                    {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]},
                    {"name": "models/gemini-2.5-pro", "supportedGenerationMethods": ["generateContent", "countTokens"]},
                ]
            }
            return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))

        ok_g, models_g, err_g = fetch_gemini_models("AIzaFakeKey123", opener=gem_ok)
        self.assertTrue(ok_g)
        self.assertEqual(models_g, ["gemini-2.5-flash", "gemini-2.5-pro"])
        self.assertEqual(err_g, "")
        self.assertNotIn("AIzaFakeKey123", str(gem_req["url"]))
        self.assertEqual(gem_req["headers"].get("x-goog-api-key"), "AIzaFakeKey123")  # type: ignore[union-attr]

        ok_ge, models_ge, _ = fetch_gemini_models(
            "AIzaFakeKey123",
            opener=lambda _req, timeout=0: _FakeHTTPResponse(b'{"models": []}'),
        )
        self.assertTrue(ok_ge)
        self.assertEqual(models_ge, [])

        ok_g401, _, err_g401 = fetch_gemini_models("AIzaFakeKey123", opener=nr_401)
        self.assertFalse(ok_g401)
        self.assertEqual(err_g401, "API key missing or wrong")

    def test_preflight_per_selected_source_and_dashboard_url(self) -> None:
        self.assertEqual(ninerouter_dashboard_url("http://127.0.0.1:20128/v1"), "http://127.0.0.1:20128/dashboard")
        self.assertEqual(ninerouter_dashboard_url("not-a-url"), "http://127.0.0.1:20128/dashboard")

        # 9router selected: missing key -> fails without probing Ollama
        (self.tmp_dir / ".env").write_text("DISCORD_TOKEN=tok\nLLM_PROVIDER=ollama\n", encoding="utf-8")
        res_nr_nokey, _ = run_preflight(
            self.tmp_dir,
            selected_source="ninerouter",
            selected_model="openai/gpt-4o",
            which=lambda _n: "ffmpeg",
            fetch_models=lambda _u: (_ for _ in ()).throw(AssertionError("should not probe")),
            port_free=lambda _h, _p: True,
        )
        by_name = {r.name: r for r in res_nr_nokey}
        self.assertFalse(by_name["9router"].ok)
        self.assertEqual(by_name["9router"].code, "ninerouter_issue")

        # 9router selected with key and model in list -> ok
        (self.tmp_dir / ".env").write_text(
            "DISCORD_TOKEN=tok\nNINEROUTER_API_KEY=nr-secret-123\n",
            encoding="utf-8",
        )
        res_nr_ok, _ = run_preflight(
            self.tmp_dir,
            selected_source="ninerouter",
            selected_model="openai/gpt-4o",
            which=lambda _n: "ffmpeg",
            fetch_models=lambda _u: (True, ["openai/gpt-4o"], ""),
            port_free=lambda _h, _p: True,
        )
        self.assertTrue(all(r.ok for r in res_nr_ok))

        # Gemini selected with key and model in list -> ok
        (self.tmp_dir / ".env").write_text(
            "DISCORD_TOKEN=tok\nGEMINI_API_KEY=AIzaSecret123\n",
            encoding="utf-8",
        )
        res_gem_ok, _ = run_preflight(
            self.tmp_dir,
            selected_source="gemini",
            selected_model="gemini-2.5-flash",
            which=lambda _n: "ffmpeg",
            fetch_models=lambda _u: (True, ["gemini-2.5-flash"], ""),
            port_free=lambda _h, _p: True,
        )
        self.assertTrue(all(r.ok for r in res_gem_ok))

    def test_tray_tooltip_includes_source_and_model(self) -> None:
        tip = format_tooltip(
            BotStatus(
                state=STATE_RUNNING,
                uptime=125,
                llm_provider="ninerouter",
                llm_model="openai/gpt-4o",
            )
        )
        self.assertEqual(tip, "Pachinko - Running - 9router: openai/gpt-4o (2m)")
        self.assertEqual(format_tooltip(BotStatus(state=STATE_STOPPED)), "Pachinko - Stopped")


if __name__ == "__main__":
    unittest.main()
