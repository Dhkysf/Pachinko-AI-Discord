"""Pertahanan Prompt-Injection berbasis Quoted-Data & filter output aman.

Prinsip:
- Teks eksternal (pesan user, nama profil, hasil web search, catatan) TIDAK PERNAH
  masuk sebagai instruksi telanjang. Semuanya dinetralisir dari control token LLM
  dan tag struktural, lalu dibungkus `<data source="...">...</data>`.
- Output bot dibersihkan dari mention massal (@everyone/@here) dan kebocoran token/key.
"""
from __future__ import annotations

import re

_CONTROL_TOKENS = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|system\|>|<\|user\|>|<\|assistant\|>"
    r"|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|<｜User｜>|<｜Assistant｜>)",
    re.IGNORECASE,
)

_STRUCTURAL_TAGS = re.compile(
    r"<(/?)(data|system|turn|now|style_note|feel|emo|silence|examples|directives)\b",
    re.IGNORECASE,
)

_ZERO_WIDTH_BIDI = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\x00-\x08\x0b\x0c\x0e-\x1f]")

_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgsk_[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\b[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27,}\b"),  # Discord token
]


def neutralize(text: str, max_chars: int = 8000) -> str:
    """Bersihkan control token & escape tag struktural agar tidak bisa memalsukan blok prompt."""
    if not text:
        return ""
    cleaned = _ZERO_WIDTH_BIDI.sub("", text)
    cleaned = _CONTROL_TOKENS.sub("", cleaned)
    cleaned = _STRUCTURAL_TAGS.sub(r"＜\1\2", cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "… [dipotong]"
    return cleaned.strip()


def safe_name(name: str) -> str:
    """Nama user satu baris aman tanpa karakter pemalsu giliran dialog."""
    cleaned = neutralize(name or "User", max_chars=32)
    cleaned = re.sub(r"[\r\n:\[\]<>]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "User"


def data_block(source: str, content: str, *, title: str = "", max_chars: int = 6000) -> str:
    """Bungkus data tak terpercaya sebagai kutipan murni."""
    src_attr = safe_name(source)
    title_attr = f' title="{safe_name(title)}"' if title else ""
    body = neutralize(content, max_chars=max_chars)
    return f'<data source="{src_attr}"{title_attr}>\n{body}\n</data>'


def filter_outgoing(text: str) -> str:
    """Cegah @everyone/@here dan sensor rahasia API apabila tidak sengaja bocor."""
    if not text:
        return ""
    out = text.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
    out = re.sub(r"<@&\d+>", "[role]", out)
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[RAHASIA_DISENSOR]", out)
    return out.strip()
