import time

import pytest

from src.riff.decision import DecisionPipeline


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
    d = await pipeline.evaluate("anything", force=True)
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
    d = await pipeline.evaluate("input", force=True)
    assert d.speak is True


async def test_echo_blocks_natural(pipeline, echo):
    echo.begin_speaking(2000, "hello world")
    d = await pipeline.evaluate("hello world")
    assert d.speak is False and d.reason == "echo"


async def test_llm_failure_returns_silence(pipeline, llm):
    async def boom(*a, **kw):
        raise RuntimeError("boom")
    llm.punchline = boom
    d = await pipeline.evaluate("input")
    assert d.speak is False and d.reason == "low_confidence"
