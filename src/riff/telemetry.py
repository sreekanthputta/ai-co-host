"""Telemetry - Observer pattern. Modules emit events, subscribers listen."""
from __future__ import annotations
import statistics
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    name: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)


Listener = Callable[[Event], Awaitable[None] | None]


class Telemetry:
    def __init__(self, history: int = 200, window_size: int = 1000):
        self._listeners: list[Listener] = []
        self._events: deque[Event] = deque(maxlen=history)
        self._timers: dict[str, float] = {}
        self._window_size = window_size
        self._durations: dict[str, deque[float]] = {}

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    async def emit(self, name: str, **data: Any) -> None:
        evt = Event(name=name, ts=time.time(), data=data)
        self._events.append(evt)
        for fn in list(self._listeners):
            try:
                res = fn(evt)
                if hasattr(res, "__await__"):
                    await res
            except Exception:
                pass

    def start(self, key: str) -> None:
        self._timers[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        start = self._timers.pop(key, None)
        if start is None:
            return 0.0
        duration_ms = (time.perf_counter() - start) * 1000.0
        self.record_duration(key, duration_ms)
        return duration_ms

    def record_duration(self, stage: str, duration_ms: float) -> None:
        if stage not in self._durations:
            self._durations[stage] = deque(maxlen=self._window_size)
        self._durations[stage].append(duration_ms)

    def get_stats(self) -> dict:
        stages = {}
        for stage, samples in self._durations.items():
            if not samples:
                continue
            sorted_s = sorted(samples)
            stages[stage] = {
                "p50": statistics.median(sorted_s),
                "p95": sorted_s[int(len(sorted_s) * 0.95)] if len(sorted_s) > 1 else sorted_s[0],
                "p99": sorted_s[int(len(sorted_s) * 0.99)] if len(sorted_s) > 1 else sorted_s[0],
                "mean": statistics.mean(sorted_s),
                "count": len(sorted_s),
            }
        total = sum(len(d) for d in self._durations.values()) + len(self._events)
        return {"stages": stages, "total_events": total}

    def reset(self) -> None:
        self._listeners.clear()
        self._events.clear()
        self._timers.clear()
        self._durations.clear()

    def recent(self, n: int = 20) -> list[Event]:
        return list(self._events)[-n:]
