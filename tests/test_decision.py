import time

import pytest

from src.riff.decision import DecisionPipeline, TriggerGate


@pytest.fixture
def pipeline(persona, memory, llm, echo, telemetry):
    return DecisionPipeline(persona, memory, llm, echo, telemetry)


async def test_empty_input_returns_silence(pipeline):
    d = await pipeline.evaluate("   ")
    assert d.speak is False and d.reason == "empty"


async def test_low_gate_blocks_speech(pipeline, memory):
    memory.gate = 0.1
    d = await pipeline.evaluate("random remark with no funny shape")
    assert d.speak is False and d.reason == "gate"
    assert d.gate_score == pytest.approx(0.1)


async def test_high_gate_high_confidence_speaks(pipeline):
    d = await pipeline.evaluate("I work as an accountant in Tulsa")
    assert d.speak is True
    assert "great line" in d.line


async def test_low_llm_score_silences(pipeline, llm):
    from src.riff.llm_client import Punchline
    llm.response = Punchline(line="meh", score=0.2)
    d = await pipeline.evaluate("audience said something")
    assert d.speak is False and d.reason == "low_confidence"


async def test_force_bypasses_gate(pipeline, memory):
    memory.gate = 0.0
    d = await pipeline.evaluate("anything at all here", force=True)
    assert d.speak is True


async def test_cooldown_blocks_natural_speech(pipeline):
    pipeline._persona = pipeline._persona.__class__(
        name="t", voice_id="v", system_prompt="p", cooldown_seconds=10.0,
        similarity_threshold=0.5,
    )
    pipeline.mark_spoken()
    d = await pipeline.evaluate("input that would otherwise pass")
    assert d.speak is False and d.reason == "cooldown"


async def test_force_bypasses_cooldown(pipeline):
    pipeline.mark_spoken()
    d = await pipeline.evaluate("some input here", force=True)
    assert d.speak is True


async def test_echo_blocks_natural(pipeline, echo):
    echo.begin_speaking(2000, "hello world everyone here")
    d = await pipeline.evaluate("hello world everyone here")
    assert d.speak is False and d.reason == "echo"


async def test_llm_failure_returns_silence(pipeline, llm):
    async def boom(*a, **kw):
        raise RuntimeError("boom")
    llm.punchline = boom
    d = await pipeline.evaluate("input that is long enough")
    assert d.speak is False and d.reason == "low_confidence"


# --- TriggerGate tests ---

def test_trigger_gate_empty_text():
    g = TriggerGate(cooldown_seconds=0.0)
    assert g.should_proceed("") is False


def test_trigger_gate_short_text():
    g = TriggerGate(cooldown_seconds=0.0)
    assert g.should_proceed("yeah") is False


def test_trigger_gate_sufficient_words():
    g = TriggerGate(cooldown_seconds=0.0)
    assert g.should_proceed("hello how are you") is True


def test_trigger_gate_cooldown_blocks():
    g = TriggerGate(cooldown_seconds=10.0)
    g.record_speech()
    assert g.should_proceed("hello how are you") is False


def test_trigger_gate_record_then_proceed():
    g = TriggerGate(cooldown_seconds=10.0)
    g.record_speech()
    assert g.should_proceed("this is enough words") is False


def test_trigger_gate_force_open():
    g = TriggerGate(cooldown_seconds=10.0)
    g.record_speech()
    g.force_open()
    assert g.should_proceed("hello how are you") is True


def test_trigger_gate_in_cooldown():
    g = TriggerGate(cooldown_seconds=10.0)
    assert g.in_cooldown() is False
    g.record_speech()
    assert g.in_cooldown() is True


# --- DecisionPipeline upgrade tests ---

async def test_short_text_returns_too_short(pipeline):
    d = await pipeline.evaluate("hi")
    assert d.speak is False and d.reason == "too_short"


async def test_force_bypasses_word_count(pipeline):
    d = await pipeline.evaluate("hi", force=True)
    assert d.speak is True


async def test_format_context_with_callbacks():
    from src.riff.memory import Hit
    cbs = [Hit(text="someone said tacos", score=0.8, id="1")]
    ctx = DecisionPipeline.format_context("hello everyone", cbs)
    assert "Host/audience just said: hello everyone" in ctx
    assert "Earlier in the show:" in ctx
    assert "- someone said tacos" in ctx


async def test_format_context_with_tropes():
    from src.riff.memory import Hit
    tropes = [Hit(text="callback humor", score=0.7, id="2")]
    ctx = DecisionPipeline.format_context("test input", [], tropes)
    assert "Comedy patterns:" in ctx
    assert "- callback humor" in ctx


async def test_format_context_empty_lists():
    ctx = DecisionPipeline.format_context("just this", [])
    assert ctx == "Host/audience just said: just this"
