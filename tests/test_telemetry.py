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


# --- New tests ---


async def test_get_stats_empty():
    t = Telemetry()
    stats = t.get_stats()
    assert stats["stages"] == {}


async def test_get_stats_known_data():
    t = Telemetry()
    # 100 samples: 0, 1, 2, ..., 99
    for i in range(100):
        t.record_duration("retrieval", float(i))
    stats = t.get_stats()
    s = stats["stages"]["retrieval"]
    assert s["count"] == 100
    assert s["p50"] == 49.5  # median of 0..99
    assert s["p95"] == 95.0
    assert s["p99"] == 99.0
    assert s["mean"] == 49.5


async def test_record_duration_accumulates():
    t = Telemetry()
    t.record_duration("llm", 10.0)
    t.record_duration("llm", 20.0)
    t.record_duration("llm", 30.0)
    stats = t.get_stats()
    assert stats["stages"]["llm"]["count"] == 3
    assert stats["stages"]["llm"]["mean"] == 20.0


async def test_stop_without_start_returns_zero_no_error():
    t = Telemetry()
    result = t.stop("nonexistent")
    assert result == 0.0
    # No duration should be recorded
    assert "nonexistent" not in t._durations


async def test_reset_clears_everything():
    t = Telemetry()
    t.subscribe(lambda e: None)
    await t.emit("x")
    t.start("a")
    t.record_duration("stage", 5.0)
    t.reset()
    assert t.recent() == []
    assert t.get_stats() == {"stages": {}, "total_events": 0}
    assert t._listeners == []
    assert t._timers == {}


async def test_window_size_respected():
    t = Telemetry(window_size=5)
    for i in range(10):
        t.record_duration("s", float(i))
    # Only last 5 samples kept: 5,6,7,8,9
    stats = t.get_stats()
    assert stats["stages"]["s"]["count"] == 5
    assert stats["stages"]["s"]["mean"] == 7.0


async def test_multiple_stages_independent():
    t = Telemetry()
    t.record_duration("retrieval", 3.0)
    t.record_duration("retrieval", 5.0)
    t.record_duration("llm", 800.0)
    t.record_duration("llm", 1200.0)
    stats = t.get_stats()
    assert stats["stages"]["retrieval"]["count"] == 2
    assert stats["stages"]["llm"]["count"] == 2
    assert stats["stages"]["retrieval"]["mean"] == 4.0
    assert stats["stages"]["llm"]["mean"] == 1000.0


async def test_emit_fan_out_multiple_listeners():
    t = Telemetry()
    a, b = [], []
    t.subscribe(lambda e: a.append(e.name))
    t.subscribe(lambda e: b.append(e.name))
    await t.emit("ev1")
    await t.emit("ev2")
    assert a == ["ev1", "ev2"]
    assert b == ["ev1", "ev2"]


async def test_listener_exception_doesnt_crash_others():
    t = Telemetry()
    results = []

    def bad_listener(e):
        raise RuntimeError("boom")

    t.subscribe(bad_listener)
    t.subscribe(lambda e: results.append(e.name))
    await t.emit("safe")
    assert results == ["safe"]
