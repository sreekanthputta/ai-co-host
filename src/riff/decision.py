"""DecisionPipeline - decides whether/what to chime in response to a turn.

Two paths:
  1. Natural turn: gated by Moss similarity + cooldown + echo filter.
  2. Forced trigger: bypasses gate, always tries to produce a line.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

from .echo_filter import EchoFilter
from .llm_client import MinimaxClient, Punchline
from .memory import MemoryPort
from .persona import PersonaPack
from .telemetry import Telemetry


@dataclass
class Decision:
    speak: bool
    line: str = ""
    reason: str = ""
    gate_score: float = 0.0
    llm_score: float = 0.0


class TriggerGate:
    def __init__(self, cooldown_seconds: float = 8.0, min_word_count: int = 3):
        self._cooldown_seconds = cooldown_seconds
        self._min_word_count = min_word_count
        self._last_spoke_at: float = 0.0

    def should_proceed(self, text: str) -> bool:
        """Returns True if text passes cooldown + word count gates."""
        if not text or len(text.split()) < self._min_word_count:
            return False
        if (time.time() - self._last_spoke_at) < self._cooldown_seconds:
            return False
        return True

    def record_speech(self) -> None:
        self._last_spoke_at = time.time()

    def force_open(self) -> None:
        """Reset cooldown so next check passes."""
        self._last_spoke_at = 0.0

    def in_cooldown(self) -> bool:
        return (time.time() - self._last_spoke_at) < self._cooldown_seconds


class DecisionPipeline:
    def __init__(
        self,
        persona: PersonaPack,
        memory: MemoryPort,
        llm: MinimaxClient,
        echo: EchoFilter,
        telemetry: Telemetry,
    ):
        self._persona = persona
        self._memory = memory
        self._llm = llm
        self._echo = echo
        self._telemetry = telemetry
        self._last_spoke_at: float = 0.0

    def mark_spoken(self) -> None:
        self._last_spoke_at = time.time()

    def in_cooldown(self) -> bool:
        return (time.time() - self._last_spoke_at) < self._persona.cooldown_seconds

    @staticmethod
    def format_context(heard: str, callbacks: list, tropes: list = None) -> str:
        parts = [f"Host/audience just said: {heard}"]
        if callbacks:
            parts.append("Earlier in the show:\n" + "\n".join(f"- {h.text}" for h in callbacks))
        if tropes:
            parts.append("Comedy patterns:\n" + "\n".join(f"- {h.text}" for h in tropes))
        return "\n\n".join(parts)

    async def evaluate(self, heard: str, *, force: bool = False) -> Decision:
        if not heard.strip() and not force:
            return Decision(speak=False, reason="empty")

        if not force and len(heard.split()) < 3:
            return Decision(speak=False, reason="too_short")

        if not force and self._echo.looks_like_echo(heard):
            await self._telemetry.emit("decision.skip", reason="echo", heard=heard)
            return Decision(speak=False, reason="echo")

        if not force and self.in_cooldown():
            await self._telemetry.emit("decision.skip", reason="cooldown")
            return Decision(speak=False, reason="cooldown")

        gate = 1.0 if force else await self._memory.gate_score(heard)
        if not force and gate < self._persona.similarity_threshold:
            await self._telemetry.emit("decision.skip", reason="gate", score=gate)
            return Decision(speak=False, reason="gate", gate_score=gate)

        callbacks = await self._memory.callback(heard, k=2)
        callback_block = "\n".join(f"- {h.text}" for h in callbacks) or "(none)"
        context = (
            f"Host/audience just said: {heard}\n"
            f"Earlier in the show:\n{callback_block}"
        )

        self._telemetry.start("llm")
        punch = await self._safe_punchline(self._persona.render_system_prompt(), context)
        latency_ms = self._telemetry.stop("llm")
        await self._telemetry.emit(
            "decision.llm", latency_ms=latency_ms, score=punch.score, line=punch.line
        )

        if not punch.line or punch.score < 0.5:
            return Decision(
                speak=False, reason="low_confidence",
                gate_score=gate, llm_score=punch.score,
            )

        return Decision(
            speak=True, line=punch.line, reason="ok",
            gate_score=gate, llm_score=punch.score,
        )

    async def _safe_punchline(self, system: str, context: str) -> Punchline:
        try:
            return await self._llm.punchline(
                system_prompt=system,
                context=context,
                max_tokens=self._persona.max_response_tokens,
            )
        except Exception as exc:
            await self._telemetry.emit("decision.llm_error", error=repr(exc))
            return Punchline(line="", score=0.0, raw=repr(exc))
