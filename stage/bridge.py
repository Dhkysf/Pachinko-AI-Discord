"""VTuber Stage Bridge: VTube Studio Live2D (Lip-Sync + Ekspresi + Idle Sway) & OBS WebSocket v5."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import time
from typing import Any

import websockets

from config import config
from core.bus import Event, EventBus
from core.store import Store

log = logging.getLogger("stage")

EMOTION_SMILE_MAP = {
    "happy": 0.85,
    "smug": 0.65,
    "shy": 0.45,
    "neutral": 0.15,
    "thinking": 0.05,
    "surprised": 0.10,
    "annoyed": -0.35,
    "sad": -0.45,
    "angry": -0.75,
}


class StageBridge:
    def __init__(self, store: Store, bus: EventBus) -> None:
        self.store = store
        self.bus = bus
        self.vts_connected = False
        self.obs_connected = False
        self.current_emotion = "neutral"
        self.current_scene = "Chat"
        self._rms = 0.0
        self._smile = 0.15
        self._vts_ws: Any = None
        self._obs_ws: Any = None
        self._hotkeys: dict[str, str] = {}  # nama_ekspresi_lower -> hotkeyID
        self._tasks: list[asyncio.Task] = []

        self.bus.on("stage.lipsync", self._on_lipsync)
        self.bus.on("stage.emotion", self._on_emotion)

    async def start(self) -> None:
        if config.vts_enabled:
            self._tasks.append(asyncio.create_task(self._vts_loop()))
        if config.obs_enabled:
            self._tasks.append(asyncio.create_task(self._obs_loop()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    def status(self) -> dict[str, Any]:
        return {
            "vts_enabled": config.vts_enabled,
            "vts_connected": self.vts_connected,
            "obs_enabled": config.obs_enabled,
            "obs_connected": self.obs_connected,
            "emotion": self.current_emotion,
            "scene": self.current_scene,
        }

    def _on_lipsync(self, ev: Event) -> None:
        self._rms = float(ev.payload.get("rms", 0.0))

    async def _on_emotion(self, ev: Event) -> None:
        emo = str(ev.payload.get("emotion", "neutral")).lower()
        self.current_emotion = emo
        self._smile = EMOTION_SMILE_MAP.get(emo, 0.1)
        if self.vts_connected and self._vts_ws:
            # Cari hotkey VTube Studio yang namanya mengandung nama emosi
            for hk_name, hk_id in self._hotkeys.items():
                if emo in hk_name:
                    await self._vts_send("HotkeyTriggerRequest", {"hotkeyID": hk_id})
                    break

    # ---------------- VTube Studio WebSocket (:8001) ----------------
    async def _vts_send(self, msg_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._vts_ws:
            return {}
        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"miyoria_{int(time.time() * 1000)}",
            "messageType": msg_type,
        }
        if data is not None:
            req["data"] = data
        await self._vts_ws.send(json.dumps(req))
        raw = await asyncio.wait_for(self._vts_ws.recv(), timeout=5.0)
        return json.loads(raw)

    async def _vts_loop(self) -> None:
        while True:
            try:
                async with websockets.connect(config.vts_ws_url) as ws:
                    self._vts_ws = ws
                    token = await self.store.get_kv("vts_auth_token", "")
                    if not token:
                        res = await self._vts_send(
                            "AuthenticationTokenRequest",
                            {"pluginName": "Miyoria-Py Stage", "pluginDeveloper": "MiyoriaLocal"},
                        )
                        token = res.get("data", {}).get("authenticationToken", "")
                        if token:
                            await self.store.set_kv("vts_auth_token", token)

                    auth = await self._vts_send(
                        "AuthenticationRequest",
                        {
                            "pluginName": "Miyoria-Py Stage",
                            "pluginDeveloper": "MiyoriaLocal",
                            "authenticationToken": token,
                        },
                    )
                    if not auth.get("data", {}).get("authenticated"):
                        log.warning("VTube Studio belum mengizinkan plugin (klik Allow di VTube Studio).")
                        await self.store.set_kv("vts_auth_token", "")
                        await asyncio.sleep(8.0)
                        continue

                    self.vts_connected = True
                    log.info("Terhubung ke VTube Studio Live2D!")
                    hk_res = await self._vts_send("HotkeysInCurrentModelRequest", {})
                    self._hotkeys = {
                        h["name"].lower(): h["hotkeyID"]
                        for h in hk_res.get("data", {}).get("availableHotkeys", [])
                    }

                    # Loop 25 FPS injeksi MouthOpen + MouthSmile + gerak napas/kepala halus
                    t0 = time.monotonic()
                    while True:
                        t = time.monotonic() - t0
                        sway_x = round(math.sin(t * 0.9) * 3.5, 2)
                        sway_z = round(math.cos(t * 0.65) * 2.5, 2)
                        params = [
                            {"id": "MouthOpen", "value": round(self._rms, 3)},
                            {"id": "MouthSmile", "value": round((self._smile + 1.0) / 2.0, 3)},
                            {"id": "FaceAngleX", "value": sway_x},
                            {"id": "FaceAngleZ", "value": sway_z},
                        ]
                        await self._vts_ws.send(
                            json.dumps(
                                {
                                    "apiName": "VTubeStudioPublicAPI",
                                    "apiVersion": "1.0",
                                    "requestID": "tick",
                                    "messageType": "InjectParameterDataRequest",
                                    "data": {"faceFound": True, "mode": "set", "parameterValues": params},
                                }
                            )
                        )
                        await self._vts_ws.recv()
                        await asyncio.sleep(0.04)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.vts_connected = False
                self._vts_ws = None
                log.debug("VTube Studio belum aktif (%s), coba lagi 10 detik...", e)
                await asyncio.sleep(10.0)

    # ---------------- OBS WebSocket v5 (:4455) ----------------
    async def switch_obs_scene(self, scene_name: str) -> bool:
        self.current_scene = scene_name
        await self.bus.emit("stage.scene_changed", scene=scene_name)
        if not self.obs_connected or not self._obs_ws:
            return False
        try:
            await self._obs_ws.send(
                json.dumps(
                    {
                        "op": 6,
                        "d": {
                            "requestType": "SetCurrentProgramScene",
                            "requestId": f"scene_{int(time.time())}",
                            "requestData": {"sceneName": scene_name},
                        },
                    }
                )
            )
            return True
        except Exception:
            return False

    async def _obs_loop(self) -> None:
        while True:
            try:
                async with websockets.connect(config.obs_ws_url) as ws:
                    hello = json.loads(await ws.recv())
                    auth_str = None
                    auth_info = hello.get("d", {}).get("authentication")
                    if auth_info and config.obs_ws_password:
                        secret = base64.b64encode(
                            hashlib.sha256((config.obs_ws_password + auth_info["salt"]).encode()).digest()
                        ).decode()
                        auth_str = base64.b64encode(
                            hashlib.sha256((secret + auth_info["challenge"]).encode()).digest()
                        ).decode()

                    ident: dict[str, Any] = {"rpcVersion": 1}
                    if auth_str:
                        ident["authentication"] = auth_str
                    await ws.send(json.dumps({"op": 1, "d": ident}))
                    identified = json.loads(await ws.recv())
                    if identified.get("op") == 2:
                        self._obs_ws = ws
                        self.obs_connected = True
                        log.info("Terhubung ke OBS Studio WebSocket v5!")
                        while True:
                            await ws.recv()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.obs_connected = False
                self._obs_ws = None
                await asyncio.sleep(12.0)
