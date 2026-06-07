import asyncio

import pytest

from src.riff.telemetry import Telemetry


async def test_emit_dispatches_to_subscribers():
    t = Telemetry()
    received = []

    async def listener(evt):
        received.append(evt.name)

    t.subscribe(listener)
    await t.emit("hello", k=1)
    assert received == ["hello"]


async def test_recent_returns_last_n():
    t = Telemetry(history=5)
    for i in range(10):
        await t.emit(f"e{i}")
    names = [e.name for e in t.recent(3)]
    assert names == ["e7", "e8", "e9"]


async def test_timer_round_trip():
    t = Telemetry()
    t.start("op")
    await asyncio.sleep(0.01)
    ms = t.stop("op")
    assert ms >= 5.0


async def test_stop_unknown_returns_zero():
    assert Telemetry().stop("missing") == 0.0


async def test_sync_listener_supported():
    t = Telemetry()
    captured = []
    t.subscribe(lambda e: captured.append(e.name))
    await t.emit("sync")
    assert captured == ["sync"]
