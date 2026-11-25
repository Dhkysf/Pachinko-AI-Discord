"""Offline unit tests for SingleInstance IPC socket in app/__main__.py."""
from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from app.__main__ import SingleInstance


class TestSingleInstance(unittest.TestCase):
    def test_idle_client_and_wrong_token_do_not_block_valid_show(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inst = SingleInstance(root, root / "app.lock")
            self.assertTrue(inst.acquire())
            shown = threading.Event()
            inst.serve(shown.set)
            port = int(json.loads((root / "app.lock").read_text(encoding="utf-8"))["port"])
            idle: socket.socket | None = None
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1.0) as bad:
                    bad.sendall(b"SHOW wrong\n")
                self.assertFalse(shown.wait(0.5))

                idle = socket.create_connection(("127.0.0.1", port), timeout=1.0)
                self.assertTrue(inst.notify_existing(timeout=3.0))
                self.assertTrue(shown.wait(3.0))
            finally:
                if idle is not None:
                    idle.close()
                inst.close()


if __name__ == "__main__":
    unittest.main()
