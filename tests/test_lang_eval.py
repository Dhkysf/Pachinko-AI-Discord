"""Offline unit tests for tools/lang_eval.py leak detectors."""
from __future__ import annotations

import unittest

from tools.lang_eval import detect_leaks


class TestLangEvalDetectors(unittest.TestCase):
    def test_detects_tag_json_and_english_leaks(self) -> None:
        self.assertTrue(detect_leaks("Halo! <feel>self: joy weak</feel>")["tag_leak"])
        self.assertTrue(detect_leaks("Halo! <emo:hap")["tag_leak"])
        self.assertTrue(detect_leaks('Oke {"name": "set_reminder", "arguments": {"minutes": 1}}')["json_leak"])
        self.assertTrue(detect_leaks("I'm sorry, I don't know what you mean by that.")["english_leak"])

    def test_allows_clean_indonesian_with_tech_loanwords(self) -> None:
        clean_samples = (
            "RAM itu memori sementara di komputer buat nyimpen data yang lagi dipakai.",
            "Aku bukan VTuber kok, nggak punya avatar atau stream.",
            "Sorry ya, aku kurang paham maksud slang itu. Bisa jelasin dikit?",
        )
        for sample in clean_samples:
            leaks = detect_leaks(sample)
            self.assertFalse(any(leaks.values()), msg=f"False positive on {sample!r}: {leaks}")


if __name__ == "__main__":
    unittest.main()
