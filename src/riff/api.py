"""FastAPI trigger server - exposes the ForceTrigger + Telemetry over HTTP/WS."""
from __future__ import annotations
import asyncio
import json
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .echo_filter import EchoFilter
from .telemetry import Telemetry, Event
from .trigger import ForceTrigger


def build_app(
    *,
    trigger: ForceTrigger,
    telemetry: Telemetry,
    echo: EchoFilter,
    state: dict[str, Any],
) -> FastAPI:
    app = FastAPI(title="Riff Trigger API")
    sockets: list[WebSocket] = []

    async def fanout(evt: Event) -> None:
        payload = json.dumps({"name": evt.name, "ts": evt.ts, "data": evt.data})
        dead: list[WebSocket] = []
        for ws in sockets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            sockets.remove(ws)

    telemetry.subscribe(fanout)

    @app.post("/trigger")
    async def trigger_now(payload: dict[str, Any] | None = None):
        hint = (payload or {}).get("hint", "") if payload else ""
        trigger.fire(hint=hint)
        return {"ok": True, "queued": True, "hint": hint}

    @app.post("/mute")
    async def mute(payload: dict[str, Any] | None = None):
        seconds = float((payload or {}).get("seconds", 30.0))
        echo.mute_for(seconds)
        return {"ok": True, "muted_for": seconds}

    @app.post("/unmute")
    async def unmute():
        echo.unmute()
        return {"ok": True}

    @app.get("/status")
    async def status():
        return {
            "persona": state.get("persona", "unknown"),
            "muted": echo.is_muted(),
            "events_recent": [
                {"name": e.name, "ts": e.ts, "data": e.data}
                for e in telemetry.recent(20)
            ],
        }

    @app.websocket("/events")
    async def events(ws: WebSocket):
        await ws.accept()
        sockets.append(ws)
        try:
            while True:
                # passive - we push from telemetry. heartbeat reads keep socket alive.
                await asyncio.sleep(15)
                await ws.send_text(json.dumps({"name": "ping", "ts": 0, "data": {}}))
        except WebSocketDisconnect:
            pass
        finally:
            if ws in sockets:
                sockets.remove(ws)

    return app
