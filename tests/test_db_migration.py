"""Tests for one-time legacy SQLite database migration."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.store import LEGACY_DB_NAME, migrate_legacy_db


class TestLegacyDbMigration(unittest.TestCase):
    def test_migrates_db_wal_and_shm_when_target_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            legacy = data_dir / LEGACY_DB_NAME
            target = data_dir / "pachinko.db"
            for suffix in ("", "-wal", "-shm"):
                Path(f"{legacy}{suffix}").write_bytes(b"dbdata")

            migrated = migrate_legacy_db(target, data_dir)

            self.assertTrue(migrated)
            for suffix in ("", "-wal", "-shm"):
                self.assertFalse(Path(f"{legacy}{suffix}").exists())
                self.assertTrue(Path(f"{target}{suffix}").exists())

    def test_skips_when_target_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            legacy = data_dir / LEGACY_DB_NAME
            target = data_dir / "pachinko.db"
            legacy.write_bytes(b"old")
            target.write_bytes(b"new")

            self.assertFalse(migrate_legacy_db(target, data_dir))
            self.assertEqual(legacy.read_bytes(), b"old")
            self.assertEqual(target.read_bytes(), b"new")

    def test_skips_when_custom_db_path_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            legacy = data_dir / LEGACY_DB_NAME
            custom = data_dir / "custom.db"
            legacy.write_bytes(b"old")

            self.assertFalse(migrate_legacy_db(custom, data_dir))
            self.assertTrue(legacy.exists())


if __name__ == "__main__":
    unittest.main()
