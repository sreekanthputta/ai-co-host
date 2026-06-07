"""Tests for RiffAgent - index_turn, process_chat_turn, on_user_turn_completed, get_transcript."""
from __future__ import annotations
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.riff.agent import RiffAgent
from src.riff.decision import Decision


class FakeDecision:
    def __init__(self, speak=True, line="funny line", score=0.9):
        self._decision = Decision(speak=speak, line=line, llm_score=score)
        self.evaluated: list[str] = []
        self.contexts: list[str] = []

    async def evaluate(self, text, *, force=False, context=""):
        self.evaluated.append(text)
        self.contexts.append(context)
        return self._decision

    def mark_spoken(self):
        pass


@pytest.fixture
def fake_decision():
    return FakeDecision()


@pytest.fixture
def agent(persona, memory, fake_decision, echo, telemetry, trigger):
    return RiffAgent(
        persona=persona,
        memory=memory,
        decision=fake_decision,
        echo=echo,
        telemetry=telemetry,
        trigger=trigger,
    )


# --- index_turn tests ---

async def test_index_turn_increments_counter(agent):
    await agent.index_turn("hello", "host")
    await agent.index_turn("world", "host")
    assert agent._turn == 2


async def test_index_turn_returns_turn_id(agent):
    tid = await agent.index_turn("hey", "guest")
    assert tid == "turn-1"
    tid2 = await agent.index_turn("yo", "guest")
    assert tid2 == "turn-2"


async def test_index_turn_calls_memory_remember(agent, memory):
    await agent.index_turn("nice joke", "host")
    assert ("turn-1", "[host] nice joke") in memory.remembered


async def test_index_turn_emits_telemetry(agent, telemetry):
    await agent.index_turn("test text", "audience", source="audio")
    evts = [e for e in telemetry.recent() if e.name == "turn.indexed"]
    assert len(evts) == 1
    assert evts[0].data["speaker"] == "audience"
    assert evts[0].data["source"] == "audio"
    assert evts[0].data["text"] == "test text"
    assert evts[0].data["turn"] == 1


# --- process_chat_turn tests ---

async def test_process_chat_turn_speaks(agent):
    result = await agent.process_chat_turn("hello world", "user1")
    assert result is not None
    assert result["reply"] == "funny line"
    assert result["turn_id"] == "turn-1"
    assert result["decision"] == "spoke"
    assert result["score"] == 0.9


async def test_process_chat_turn_silent(agent, fake_decision):
    fake_decision._decision = Decision(speak=False, reason="gate")
    result = await agent.process_chat_turn("boring", "user2")
    assert result is None


async def test_process_chat_turn_indexes_regardless(agent, memory, fake_decision):
    fake_decision._decision = Decision(speak=False, reason="cooldown")
    await agent.process_chat_turn("still indexed", "user3")
    assert ("turn-1", "[user3] still indexed") in memory.remembered


async def test_process_chat_turn_emits_chime_on_speak(agent, telemetry):
    await agent.process_chat_turn("get a joke", "chatter")
    evts = [e for e in telemetry.recent() if e.name == "chime.emitted"]
    assert len(evts) == 1
    assert evts[0].data["source"] == "chat"


# --- on_user_turn_completed tests ---

@dataclass
class FakeMessage:
    text_content: str | None = None


async def test_on_user_turn_completed_indexes(agent, memory):
    ctx = MagicMock()
    msg = FakeMessage(text_content="audience says hi")
    await agent.on_user_turn_completed(ctx, msg)
    assert ("turn-1", "[audience] audience says hi") in memory.remembered


async def test_on_user_turn_completed_suppresses_echo(agent, echo, memory, telemetry):
    echo.begin_speaking(5000, "echo text")
    ctx = MagicMock()
    msg = FakeMessage(text_content="echo text")
    await agent.on_user_turn_completed(ctx, msg)
    assert len(memory.remembered) == 0
    evts = [e for e in telemetry.recent() if e.name == "turn.suppressed"]
    assert len(evts) == 1


async def test_on_user_turn_completed_calls_decision(agent, fake_decision):
    ctx = MagicMock()
    msg = FakeMessage(text_content="some audience input")
    await agent.on_user_turn_completed(ctx, msg)
    assert "some audience input" in fake_decision.evaluated


async def test_on_user_turn_completed_empty_text_noop(agent, memory):
    ctx = MagicMock()
    msg = FakeMessage(text_content="   ")
    await agent.on_user_turn_completed(ctx, msg)
    assert len(memory.remembered) == 0


# --- get_transcript tests ---

async def test_get_transcript_empty(agent):
    assert agent.get_transcript() == []


async def test_get_transcript_returns_all(agent):
    await agent.index_turn("a", "s1", source="audio")
    await agent.index_turn("b", "s2", source="chat")
    await agent.index_turn("c", "s3", source="audio")
    result = agent.get_transcript()
    assert len(result) == 3
    assert result[0] == {"id": "turn-1", "speaker": "s1", "text": "a", "source": "audio"}
    assert result[2] == {"id": "turn-3", "speaker": "s3", "text": "c", "source": "audio"}


async def test_get_transcript_last_n(agent):
    await agent.index_turn("a", "s1")
    await agent.index_turn("b", "s2")
    await agent.index_turn("c", "s3")
    result = agent.get_transcript(last_n=2)
    assert len(result) == 2
    assert result[0]["text"] == "b"
    assert result[1]["text"] == "c"


async def test_get_transcript_has_all_fields(agent):
    await agent.index_turn("line", "speaker", source="chat")
    entry = agent.get_transcript()[0]
    assert "id" in entry
    assert "speaker" in entry
    assert "text" in entry
    assert "source" in entry


# --- ambient context tests ---


async def test_ambient_context_returns_empty_when_no_hits(agent):
    ctx = await agent._ambient_context("anything")
    assert ctx == ""


async def test_ambient_context_formats_hits(persona, echo, telemetry, trigger):
    from src.riff.memory import Hit

    class MemoryWithHits:
        def __init__(self):
            self.remembered = []
            self.pushes = 0

        async def gate_score(self, text):
            return 0.9

        async def callback(self, text, k=2):
            return [Hit(text="earlier joke about cats", score=0.8, id="h1")]

        async def remember(self, doc_id, text):
            self.remembered.append((doc_id, text))

        async def push(self):
            self.pushes += 1

    mem = MemoryWithHits()
    fd = FakeDecision()
    ag = RiffAgent(
        persona=persona,
        memory=mem,
        decision=fd,
        echo=echo,
        telemetry=telemetry,
        trigger=trigger,
    )
    ctx = await ag._ambient_context("tell me about cats")
    assert "earlier joke about cats" in ctx
    assert "CALLBACKS FROM THIS SHOW" in ctx


async def test_ambient_context_passed_to_decision(persona, echo, telemetry, trigger):
    from src.riff.memory import Hit

    class MemoryWithHits:
        def __init__(self):
            self.remembered = []
            self.pushes = 0

        async def gate_score(self, text):
            return 0.9

        async def callback(self, text, k=2):
            return [Hit(text="a memorable moment", score=0.85, id="h2")]

        async def remember(self, doc_id, text):
            self.remembered.append((doc_id, text))

        async def push(self):
            self.pushes += 1

    mem = MemoryWithHits()
    fd = FakeDecision()
    ag = RiffAgent(
        persona=persona,
        memory=mem,
        decision=fd,
        echo=echo,
        telemetry=telemetry,
        trigger=trigger,
    )
    await ag.process_chat_turn("test input text", "user1")
    assert len(fd.contexts) == 1
    assert "a memorable moment" in fd.contexts[0]


async def test_on_user_turn_completed_passes_context(persona, echo, telemetry, trigger):
    from src.riff.memory import Hit
    from unittest.mock import MagicMock

    class MemoryWithHits:
        def __init__(self):
            self.remembered = []
            self.pushes = 0

        async def gate_score(self, text):
            return 0.9

        async def callback(self, text, k=2):
            return [Hit(text="crowd callback", score=0.7, id="h3")]

        async def remember(self, doc_id, text):
            self.remembered.append((doc_id, text))

        async def push(self):
            self.pushes += 1

    mem = MemoryWithHits()
    fd = FakeDecision(speak=False)
    ag = RiffAgent(
        persona=persona,
        memory=mem,
        decision=fd,
        echo=echo,
        telemetry=telemetry,
        trigger=trigger,
    )
    ctx = MagicMock()
    msg = FakeMessage(text_content="audience input here")
    await ag.on_user_turn_completed(ctx, msg)
    assert len(fd.contexts) == 1
    assert "crowd callback" in fd.contexts[0]
