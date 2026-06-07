"""Shared protocols — all modules code against these, never against concrete classes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class MemoryDoc:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryResult:
    docs: list[MemoryDoc] = field(default_factory=list)


@dataclass
class LLMResponse:
    text: str
    score: float
    latency_ms: float = 0.0
    raw: str = ""


@dataclass
class DecisionResult:
    should_speak: bool
    text: str | None = None
    score: float = 0.0
    memory_context: str = ""
    latency_ms: float = 0.0
    gate_passed: str = ""


@dataclass
class TimingEvent:
    stage: str
    duration_ms: float
    metadata: dict = field(default_factory=dict)


class Memory(Protocol):
    async def index_turn(self, turn_id: str, text: str, speaker: str, source: str = "audio", metadata: dict | None = None) -> None: ...
    async def query(self, text: str, top_k: int = 5) -> MemoryResult: ...
    async def ensure_ready(self) -> None: ...


class LLMClient(Protocol):
    async def generate(self, system: str, user: str, max_tokens: int = 50) -> LLMResponse: ...
    async def generate_candidates(self, system: str, user: str, n: int = 3, max_tokens: int = 50) -> list[LLMResponse]: ...


class TelemetryPort(Protocol):
    def record(self, event: TimingEvent) -> None: ...
    def get_stats(self) -> dict: ...
