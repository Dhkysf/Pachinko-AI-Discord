"""FastAPI Server for Web Admin Panel (`http://127.0.0.1:7878`)."""
from __future__ import annotations

import asyncio
import hmac
import json
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from brain.gate import ActionGate
from brain.profile import ProfileManager
from brain.runner import BrainRunner
from config import VALID_LLM_PROVIDERS, config
from core.bus import EventBus
from core.store import Store
from core.vram import vram_guard
from heart.engine import HeartEngine
from memory.manager import MemoryManager
from sing.engine import SingingEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"


async def _json_dict(req: Request) -> dict[str, Any]:
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return body


def _finite_float(val: Any, field: str) -> float:
    if isinstance(val, bool):
        raise HTTPException(status_code=400, detail=f"Invalid {field}")
    try:
        num = float(val)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"Invalid {field}")
    if not math.isfinite(num):
        raise HTTPException(status_code=400, detail=f"Invalid {field}")
    return num


def _validate_lore_entries(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        raise HTTPException(status_code=400, detail="lore_entries must be a list")
    for item in entries:
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail="Each lore entry must be an object")
        kws = item.get("keywords")
        if not isinstance(kws, list) or not all(isinstance(k, str) for k in kws):
            raise HTTPException(status_code=400, detail="lore_entries keywords must be a list of strings")
        if not isinstance(item.get("content"), str):
            raise HTTPException(status_code=400, detail="lore_entries content must be a string")
        if "priority" in item and item["priority"] is not None:
            _finite_float(item["priority"], "lore_entries priority")
        if "scope" in item and item["scope"] is not None and not isinstance(item["scope"], str):
            raise HTTPException(status_code=400, detail="lore_entries scope must be a string")
    return entries


def _parse_id_list(val: Any, field: str) -> list[int]:
    if not isinstance(val, list):
        raise HTTPException(status_code=400, detail=f"{field} must be a list of integers")
    for item in val:
        if isinstance(item, bool) or not isinstance(item, int):
            raise HTTPException(status_code=400, detail=f"{field} items must be integers")
    return val


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
) -> FastAPI:
    docs_enabled = not bool(config.panel_token)
    app = FastAPI(
        title="Pachinko Studio",
        docs_url="/api/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )
    allowed_hosts = ["127.0.0.1", "localhost"]
    if config.panel_host and config.panel_host not in ("0.0.0.0", "::"):
        host_entry = f"[{config.panel_host}]" if ":" in config.panel_host and not config.panel_host.startswith("[") else config.panel_host
        if host_entry.lower() not in allowed_hosts:
            allowed_hosts.append(host_entry)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    allowed_origins = {f"http://{h.lower()}:{config.panel_port}" for h in allowed_hosts}
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _token_ok(candidate: str) -> bool:
        expected = config.panel_token
        if not expected:
            return True
        return bool(candidate) and hmac.compare_digest(candidate, expected)

    def _check_auth(req: Request) -> None:
        tok = req.headers.get("Authorization", "").removeprefix("Bearer ").strip() or req.query_params.get("token", "")
        if not _token_ok(tok):
            raise HTTPException(status_code=401, detail="Invalid panel token")

    def _check_origin(req: Request) -> None:
        origin = req.headers.get("origin")
        if origin is not None and origin.lower() not in allowed_origins:
            raise HTTPException(status_code=403, detail="Origin not allowed")

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
            },
            "runtime": {
                "llm_provider": config.llm_provider,
                "llm_model": config.active_model(),
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
        _check_origin(req)
        _check_auth(req)
        body = await _json_dict(req)
        if "display_name" in body and body["display_name"] is not None:
            dn = body["display_name"]
            if not isinstance(dn, str) or len(dn.strip()) > 32:
                raise HTTPException(status_code=400, detail="display_name must be a string <= 32 chars")
        for md_key in ("identity_md", "style_md", "examples_md", "directives"):
            if md_key in body and body[md_key] is not None:
                val = body[md_key]
                if not isinstance(val, str) or len(val) > 20_000:
                    raise HTTPException(status_code=400, detail=f"{md_key} must be a string <= 20000 chars")
        if "lore_entries" in body and body["lore_entries"] is not None:
            _validate_lore_entries(body["lore_entries"])

        overrides = {k: v for k, v in body.items() if k != "directives"}
        if "directives" in body:
            await store.set_kv("owner_directives", body["directives"] or "")
        prof = await profiles.save_overrides(overrides)
        await bus.emit("persona.updated", display_name=prof.display_name)
        return {"ok": True, "display_name": prof.display_name}

    @app.post("/api/playground")
    async def api_playground(req: Request):
        _check_origin(req)
        _check_auth(req)
        body = await _json_dict(req)
        raw_msg = body.get("message", "")
        if not isinstance(raw_msg, str) or not raw_msg.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        raw_speaker = body.get("speaker", "Owner")
        speaker = raw_speaker.strip() if isinstance(raw_speaker, str) and raw_speaker.strip() else "Owner"
        res = await brain.run_text_turn(
            room_id="panel:playground",
            user_id="owner_local",
            user_name=speaker,
            user_text=raw_msg.strip(),
            wake_reason="panel_playground",
        )
        return {"reply": res.text, "emotion": res.emotion, "silent": res.silent}

    @app.get("/api/people")
    async def api_people(req: Request):
        _check_auth(req)
        return {"people": await heart.list_people(limit=60)}

    @app.post("/api/people/{user_id}")
    async def api_update_person(user_id: str, req: Request):
        _check_origin(req)
        _check_auth(req)
        if not (user_id.isascii() and user_id.isdigit()):
            raise HTTPException(status_code=400, detail="Invalid user_id")
        body = await _json_dict(req)
        if "nickname" in body and not isinstance(body["nickname"], str):
            raise HTTPException(status_code=400, detail="Invalid nickname")
        if "impression" in body and not isinstance(body["impression"], str):
            raise HTTPException(status_code=400, detail="Invalid impression")
        bond_val = max(-100.0, min(1000.0, _finite_float(body["bond"], "bond"))) if "bond" in body else None

        await heart.get_person(user_id)
        async with store.write_tx() as conn:
            if "nickname" in body:
                await conn.execute("UPDATE heart_person SET nickname=? WHERE user_id=?", (body["nickname"], user_id))
            if "impression" in body:
                await conn.execute("UPDATE heart_person SET impression=? WHERE user_id=?", (body["impression"], user_id))
            if bond_val is not None:
                await conn.execute("UPDATE heart_person SET bond=? WHERE user_id=?", (bond_val, user_id))
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
        _check_origin(req)
        _check_auth(req)
        body = await _json_dict(req)
        raw_fact = body.get("fact", "")
        if not isinstance(raw_fact, str) or not raw_fact.strip():
            raise HTTPException(status_code=400, detail="Empty note")
        raw_key = body.get("doc_key", "self")
        if raw_key is not None and not isinstance(raw_key, str):
            raise HTTPException(status_code=400, detail="Invalid doc_key")
        fact = raw_fact.strip()
        doc_key = (raw_key or "self").strip() or "self"
        await memory.upsert_note(doc_key, fact, evidence="Added via Admin Panel")
        await memory.remember(fact, scope="global", importance=0.85)
        return {"ok": True}

    @app.post("/api/memory/delete")
    async def api_delete_memory(req: Request):
        _check_origin(req)
        _check_auth(req)
        body = await _json_dict(req)
        notebook_ids = _parse_id_list(body.get("notebook_ids", []), "notebook_ids")
        memory_ids = _parse_id_list(body.get("memory_ids", []), "memory_ids")
        total = len(notebook_ids) + len(memory_ids)
        if total == 0:
            raise HTTPException(status_code=400, detail="Provide at least one id in notebook_ids or memory_ids")
        if total > 200:
            raise HTTPException(status_code=400, detail="Maximum 200 ids per request")
        notebook_deleted = await memory.delete_notes(notebook_ids) if notebook_ids else 0
        memories_deleted = await memory.delete_memories(memory_ids) if memory_ids else 0
        return {"ok": True, "notebook_deleted": notebook_deleted, "memories_deleted": memories_deleted}

    @app.post("/api/runtime")
    async def api_update_runtime(req: Request):
        _check_origin(req)
        _check_auth(req)
        body = await _json_dict(req)
        for bkey in ("autonomy_paused", "rvc_enabled"):
            if bkey in body and not isinstance(body[bkey], bool):
                raise HTTPException(status_code=400, detail=f"{bkey} must be a boolean")
        for skey in ("llm_provider", "ollama_model", "tts_provider", "edge_tts_voice"):
            if skey in body and (not isinstance(body[skey], str) or not body[skey].strip()):
                raise HTTPException(status_code=400, detail=f"{skey} must be a non-empty string")
        if "tts_provider" in body and body["tts_provider"].strip().lower() not in ("edge", "piper", "elevenlabs"):
            raise HTTPException(status_code=400, detail=f"Invalid tts_provider: {body['tts_provider'].strip()}")

        if "autonomy_paused" in body:
            await gate.set_paused(body["autonomy_paused"])
        if "llm_provider" in body:
            prov = body["llm_provider"].strip().lower()
            if prov not in VALID_LLM_PROVIDERS:
                raise HTTPException(status_code=400, detail=f"Invalid llm_provider: {prov}")
            config.llm_provider = prov
        if "ollama_model" in body:
            config.ollama_model = body["ollama_model"].strip()
        if "tts_provider" in body:
            config.tts_provider = body["tts_provider"].strip().lower()
        if "edge_tts_voice" in body:
            config.edge_tts_voice = body["edge_tts_voice"].strip()
        if "rvc_enabled" in body:
            config.rvc_enabled = body["rvc_enabled"]
        await bus.emit("runtime.updated")
        return {"ok": True}

    @app.post("/api/sing/enqueue")
    async def api_sing_enqueue(req: Request):
        _check_origin(req)
        _check_auth(req)
        body = await _json_dict(req)
        raw_q = body.get("query", "")
        if not isinstance(raw_q, str) or not raw_q.strip():
            raise HTTPException(status_code=400, detail="Empty song query")
        msg = await singing.enqueue("panel", "owner_local", "Admin Panel", raw_q.strip())
        return {"ok": True, "message": msg}

    @app.post("/api/sing/skip")
    async def api_sing_skip(req: Request):
        _check_origin(req)
        _check_auth(req)
        return {"ok": singing.skip_current()}

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        origin = ws.headers.get("origin")
        if origin is not None and origin.lower() not in allowed_origins:
            await ws.close(code=1008)
            return
        if not _token_ok(ws.query_params.get("token", "")):
            await ws.close(code=1008)
            return
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
