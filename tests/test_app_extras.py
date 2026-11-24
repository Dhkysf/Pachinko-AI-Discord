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
from datetime import datetime, timedelta
from pathlib import Path

from app.extras import (
    backup_database,
    build_diagnostics,
    model_is_installed,
    run_preflight,
    warm_up_ollama,
)
from app.process import (
    STATE_RUNNING,
    BotProcessManager,
    BotStatus,
    load_settings,
    save_settings,
)


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
        self.assertTrue(by_name["Ollama"].ok)
        self.assertIn("Skipped", by_name["Ollama"].detail)

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

    def test_process_manager_passes_ollama_model_env(self) -> None:
        out_file = self.tmp_dir / "env_model.txt"
        script = self.tmp_dir / "print_env.py"
        script.write_text(
            "import os, sys; from pathlib import Path; "
            f"Path({str(out_file)!r}).write_text(os.environ.get('OLLAMA_MODEL', '<UNSET>'), encoding='utf-8')",
            encoding="utf-8",
        )
        mgr = BotProcessManager(
            self.repo_root,
            data_dir=self.tmp_dir,
            bot_cmd=[sys.executable, str(script)],
            ollama_model="gemma3:12b",
        )
        self.assertTrue(mgr.start())
        mgr._proc.wait(timeout=5.0)  # type: ignore[union-attr]
        self.assertEqual(out_file.read_text(encoding="utf-8"), "gemma3:12b")

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
            status=BotStatus(state=STATE_RUNNING, pid=1234, uptime=42, discord_ready=True, llm_ok=True),
            selected_model="",
            env_model="qwen3:8b",
            preflight=[],
            log_path=log_path,
        )
        self.assertNotIn(fake_discord, diag)
        self.assertNotIn(fake_sk, diag)
        self.assertNotIn("super_secret_value_12345", diag)
        self.assertIn("[REDACTED]", diag)

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

    def test_settings_persist_ollama_model(self) -> None:
        cfg_path = self.tmp_dir / "app_settings.json"
        save_settings(cfg_path, {"ollama_model": "gemma3:12b", "start_bot_on_open": True})
        loaded = load_settings(cfg_path)
        self.assertEqual(loaded["ollama_model"], "gemma3:12b")
        self.assertTrue(loaded["start_bot_on_open"])


if __name__ == "__main__":
    unittest.main()
