"""Loader Profil Karakter (`profile.toml`, `identity.md`, `style.md`, `examples.md`, `lorebook.toml`) + Override Panel."""
from __future__ import annotations

import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import PROFILES_DIR, config
from core.store import Store


@dataclass
class CharacterProfile:
    profile_id: str
    display_name: str
    nicknames: list[str]
    wake_aliases: dict[str, str]
    language: str
    interests: list[str]
    temperament: str
    half_life_mins: float
    default_channel_mode: str
    filler_phrases: list[str]
    backchannel_words: set[str]
    voice_cfg: dict[str, Any]
    identity_md: str
    style_md: str
    examples_md: str
    lore_entries: list[dict[str, Any]] = field(default_factory=list)
    kaomoji: dict[str, list[str]] = field(default_factory=dict)

    def matches_lore(self, recent_text: str, modality: str = "both", limit: int = 3) -> list[str]:
        low = recent_text.lower()
        hits: list[tuple[int, str]] = []
        for entry in self.lore_entries:
            scope = entry.get("scope", "both")
            if scope not in ("both", modality):
                continue
            kws = entry.get("keywords", [])
            if any(kw.lower() in low for kw in kws):
                hits.append((int(entry.get("priority", 1)), str(entry.get("content", ""))))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in hits[:limit] if content]

    def sample_kaomoji(self, emotion: str, k: int = 3) -> list[str]:
        pool = self.kaomoji.get(emotion) or self.kaomoji.get("happy") or []
        if not pool:
            return []
        return random.sample(pool, min(k, len(pool)))


class ProfileManager:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.current: CharacterProfile = self._load_from_disk(config.profile_id)

    def _load_from_disk(self, profile_id: str) -> CharacterProfile:
        pdir: Path = PROFILES_DIR / profile_id
        if not pdir.exists():
            pdir = PROFILES_DIR / "default"

        meta = tomllib.loads((pdir / "profile.toml").read_text(encoding="utf-8"))
        lore_raw = tomllib.loads((pdir / "lorebook.toml").read_text(encoding="utf-8")) if (pdir / "lorebook.toml").exists() else {}

        temp = meta.get("temperament", {})
        beh = meta.get("behavior", {})
        return CharacterProfile(
            profile_id=meta.get("id", profile_id),
            display_name=meta.get("display_name", "Aira"),
            nicknames=[n.lower() for n in meta.get("nicknames", ["aira"])],
            wake_aliases={k.lower(): v for k, v in meta.get("wake_aliases", {}).items()},
            language=meta.get("language", "id"),
            interests=list(meta.get("interests", [])),
            temperament=temp.get("display", "tsundere"),
            half_life_mins=float(temp.get("emotion_half_life_mins", 25.0)),
            default_channel_mode=beh.get("default_channel_mode", config.default_channel_mode),
            filler_phrases=list(beh.get("filler_phrases", ["Hmm?", "Bentar ya..."])),
            backchannel_words={w.lower() for w in beh.get("backchannel_words", ["hmm", "iya", "oke", "ya", "oh"])},
            voice_cfg=dict(meta.get("voice", {})),
            identity_md=(pdir / "identity.md").read_text(encoding="utf-8").strip() if (pdir / "identity.md").exists() else "",
            style_md=(pdir / "style.md").read_text(encoding="utf-8").strip() if (pdir / "style.md").exists() else "",
            examples_md=(pdir / "examples.md").read_text(encoding="utf-8").strip() if (pdir / "examples.md").exists() else "",
            lore_entries=list(lore_raw.get("entries", [])),
            kaomoji={k: list(v) for k, v in lore_raw.get("kaomoji", {}).items()},
        )

    async def reload(self) -> CharacterProfile:
        prof = self._load_from_disk(config.profile_id)
        overrides = await self.store.get_kv("persona_overrides", {})
        if overrides.get("display_name"):
            prof.display_name = str(overrides["display_name"]).strip()
        if overrides.get("identity_md"):
            prof.identity_md = str(overrides["identity_md"]).strip()
        if overrides.get("style_md"):
            prof.style_md = str(overrides["style_md"]).strip()
        if overrides.get("examples_md"):
            prof.examples_md = str(overrides["examples_md"]).strip()
        if overrides.get("lore_entries") is not None:
            prof.lore_entries = list(overrides["lore_entries"])
        self.current = prof
        return prof

    async def save_overrides(self, data: dict[str, Any]) -> CharacterProfile:
        existing = await self.store.get_kv("persona_overrides", {})
        existing.update({k: v for k, v in data.items() if v is not None})
        await self.store.set_kv("persona_overrides", existing)
        return await self.reload()
