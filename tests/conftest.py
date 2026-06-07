"""Shared fixtures - fakes for every collaborator so unit tests stay hermetic."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Mock livekit.agents before any riff imports that depend on it.
class _FakeAgent:
    def __init__(self, **kwargs):
        pass

class _FakeAgentsModule:
    Agent = _FakeAgent
    ChatContext = object
    ChatMessage = object

_livekit_mock = MagicMock()
_livekit_mock.agents = _FakeAgentsModule
sys.modules.setdefault("livekit", _livekit_mock)
sys.modules.setdefault("livekit.agents", _FakeAgentsModule)

from src.riff.echo_filter import EchoFilter
from src.riff.memory import Hit, MemoryPort
from src.riff.persona import PersonaPack
from src.riff.telemetry import Telemetry
from src.riff.trigger import ForceTrigger


class FakeMemory:
    def __init__(self, gate: float = 0.9, callbacks: list[Hit] | None = None):
        self.gate = gate
        self.callbacks = callbacks or []
        self.remembered: list[tuple[str, str]] = []
        self.pushes = 0

    async def gate_score(self, text: str) -> float:
        return self.gate

    async def callback(self, text: str, k: int = 2) -> list[Hit]:
        return list(self.callbacks)[:k]

    async def remember(self, doc_id: str, text: str) -> None:
        self.remembered.append((doc_id, text))

    async def push(self) -> None:
        self.pushes += 1


class FakeLLM:
    def __init__(self, line: str = "great line", score: float = 0.9):
        from src.riff.llm_client import Punchline
        self.response = Punchline(line=line, score=score)
        self.calls: list[dict[str, Any]] = []

    async def punchline(self, system_prompt, context, max_tokens=50, temperature=0.85):
        self.calls.append({"system": system_prompt, "context": context, "max": max_tokens})
        return self.response


@pytest.fixture
def persona() -> PersonaPack:
    return PersonaPack(
        name="test",
        voice_id="v_test",
        system_prompt="be funny",
        similarity_threshold=0.7,
        cooldown_seconds=0.0,
        max_response_tokens=40,
        style_examples=("hello world",),
    )


@pytest.fixture
def memory() -> FakeMemory:
    return FakeMemory()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def telemetry() -> Telemetry:
    return Telemetry()


@pytest.fixture
def echo() -> EchoFilter:
    return EchoFilter()


@pytest.fixture
def trigger() -> ForceTrigger:
    return ForceTrigger()
