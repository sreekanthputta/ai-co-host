"""Telemetry - Observer pattern. Modules emit events, subscribers listen."""
from __future__ import annotations
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
    def __init__(self, history: int = 200):
        self._listeners: list[Listener] = []
        self._events: deque[Event] = deque(maxlen=history)
        self._timers: dict[str, float] = {}

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    async def emit(self, name: str, **data: Any) -> None:
        evt = Event(name=name, ts=time.time(), data=data)
        self._events.append(evt)
        for fn in self._listeners:
            res = fn(evt)
            if hasattr(res, "__await__"):
                await res

    def start(self, key: str) -> None:
        self._timers[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        start = self._timers.pop(key, None)
        return 0.0 if start is None else (time.perf_counter() - start) * 1000.0

    def recent(self, n: int = 20) -> list[Event]:
        return list(self._events)[-n:]
