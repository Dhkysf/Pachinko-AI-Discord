"""FastAPI Server untuk Web Admin Panel (`http://127.0.0.1:7878`)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from brain.gate import ActionGate
from brain.profile import ProfileManager
from brain.runner import BrainRunner
from config import config
from core.bus import EventBus
from core.store import Store
from core.vram import vram_guard
from heart.engine import HeartEngine
from memory.manager import MemoryManager
from play.economy import EconomyAndPlay
from sing.engine import SingingEngine
from speech.tts import TTSEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_web_app(
    *,
    store: Store,
    bus: EventBus,
    profiles: ProfileManager,
    heart: HeartEngine,
    memory: MemoryManager,
    gate: ActionGate,
    brain: BrainRunner,
    singing: SingingEngine,
    economy: EconomyAndPlay,
    tts: TTSEngine,
) -> FastAPI:
    app = FastAPI(title="Pachinko Studio", docs_url="/api/docs")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _check_auth(req: Request) -> None:
        if not config.panel_token:
            return
        tok = req.headers.get("Authorization", "").removeprefix("Bearer ").strip() or req.query_params.get("token", "")
        if tok != config.panel_token:
            raise HTTPException(status_code=401, detail="Token panel tidak cocok")

    @app.get("/")
    async def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/api/status")
    async def api_status(req: Request):
        _check_auth(req)
        prof = profiles.current
        return {
            "character": {
                "id": prof.profile_id,
                "display_name": prof.display_name,
                "temperament": prof.temperament,
                "default_channel_mode": prof.default_channel_mode,
            },
            "heart": heart.state_summary(prof.temperament),
            "autonomy": {
                "paused": gate.paused,
                "pending_approvals": [vars(a) for a in gate.approvals.values() if a.status == "pending"],
            },
            "runtime": {
                "llm_provider": config.llm_provider,
                "ollama_model": config.ollama_model,
                "stt_provider": config.stt_provider,
                "whisper_model": config.whisper_model,
                "tts_provider": config.tts_provider,
                "edge_tts_voice": config.edge_tts_voice,
                "rvc_enabled": config.rvc_enabled,
                "gpu_vram_gb": config.gpu_vram_gb,
                "vram_busy_with": vram_guard.busy_with,
            },
            "singing": singing.snapshot(),
        }

    @app.get("/api/events")
    async def api_events(req: Request, limit: int = 60):
        _check_auth(req)
        return {"events": bus.recent("*", limit=min(200, limit))}

    @app.get("/api/persona")
    async def api_get_persona(req: Request):
        _check_auth(req)
        prof = profiles.current
        directives = await store.get_kv("owner_directives", "") or ""
        return {
            "display_name": prof.display_name,
            "identity_md": prof.identity_md,
            "style_md": prof.style_md,
            "examples_md": prof.examples_md,
            "lore_entries": prof.lore_entries,
            "directives": directives,
        }

    @app.post("/api/persona")
    async def api_save_persona(req: Request):
        _check_auth(req)
        body: dict[str, Any] = await req.json()
        if "directives" in body:
            await store.set_kv("owner_directives", str(body.pop("directives") or ""))
        prof = await profiles.save_overrides(body)
        await bus.emit("persona.updated", display_name=prof.display_name)
        return {"ok": True, "display_name": prof.display_name}

    @app.post("/api/playground")
    async def api_playground(req: Request):
        _check_auth(req)
        body = await req.json()
        msg = str(body.get("message", "")).strip()
        speaker = str(body.get("speaker", "Owner")).strip() or "Owner"
        if not msg:
            raise HTTPException(status_code=400, detail="Pesan kosong")
        res = await brain.run_text_turn(
            room_id="panel:playground",
            user_id="owner_local",
            user_name=speaker,
            user_text=msg,
            wake_reason="panel_playground",
        )
        return {"reply": res.text, "emotion": res.emotion, "silent": res.silent}

    @app.get("/api/people")
    async def api_people(req: Request):
        _check_auth(req)
        return {"people": await heart.list_people(limit=60)}

    @app.post("/api/people/{user_id}")
    async def api_update_person(user_id: str, req: Request):
        _check_auth(req)
        body = await req.json()
        await heart.get_person(user_id)
        if "nickname" in body:
            await store.conn.execute("UPDATE heart_person SET nickname=? WHERE user_id=?", (str(body["nickname"]), user_id))
        if "impression" in body:
            await store.conn.execute("UPDATE heart_person SET impression=? WHERE user_id=?", (str(body["impression"]), user_id))
        if "bond" in body:
            b = max(-100.0, min(1000.0, float(body["bond"])))
            await store.conn.execute("UPDATE heart_person SET bond=? WHERE user_id=?", (b, user_id))
        await store.conn.commit()
        return {"ok": True}

    @app.get("/api/memory")
    async def api_memory(req: Request):
        _check_auth(req)
        notes = await memory.list_all_notes(limit=60)
        cur = await store.conn.execute("SELECT id, scope, content, importance, created_at FROM memories ORDER BY id DESC LIMIT 40")
        mems = [dict(r) for r in await cur.fetchall()]
        return {"notebook": notes, "memories": mems}

    @app.post("/api/memory")
    async def api_add_memory(req: Request):
        _check_auth(req)
        body = await req.json()
        fact = str(body.get("fact", "")).strip()
        doc_key = str(body.get("doc_key", "self")).strip() or "self"
        if not fact:
            raise HTTPException(status_code=400, detail="Isi catatan kosong")
        await memory.upsert_note(doc_key, fact, evidence="Ditambahkan lewat Admin Panel")
        await memory.remember(fact, scope="global", importance=0.85)
        return {"ok": True}

    @app.post("/api/runtime")
    async def api_update_runtime(req: Request):
        _check_auth(req)
        body = await req.json()
        if "autonomy_paused" in body:
            await gate.set_paused(bool(body["autonomy_paused"]))
        if "llm_provider" in body:
            config.llm_provider = str(body["llm_provider"]).lower()
        if "ollama_model" in body:
            config.ollama_model = str(body["ollama_model"]).strip()
        if "tts_provider" in body:
            config.tts_provider = str(body["tts_provider"]).lower()
        if "edge_tts_voice" in body:
            config.edge_tts_voice = str(body["edge_tts_voice"]).strip()
        if "rvc_enabled" in body:
            config.rvc_enabled = bool(body["rvc_enabled"])
        await bus.emit("runtime.updated")
        return {"ok": True}

    @app.post("/api/sing/enqueue")
    async def api_sing_enqueue(req: Request):
        _check_auth(req)
        body = await req.json()
        q = str(body.get("query", "")).strip()
        if not q:
            raise HTTPException(status_code=400, detail="Judul lagu kosong")
        msg = await singing.enqueue("panel", "owner_local", "Admin Panel", q)
        return {"ok": True, "message": msg}

    @app.post("/api/sing/skip")
    async def api_sing_skip(req: Request):
        _check_auth(req)
        return {"ok": singing.skip_current()}

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        q = bus.subscribe_stream(maxsize=250)
        try:
            while True:
                ev = await q.get()
                await ws.send_text(json.dumps(ev.to_dict(), ensure_ascii=False))
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            bus.unsubscribe_stream(q)

    return app
