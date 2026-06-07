"""RiffAgent - LiveKit Agent with Dependency Injection.

The agent itself stays thin: it routes events into the DecisionPipeline,
handles forced triggers, and hands lines to the AgentSession for TTS.
"""
from __future__ import annotations
import asyncio
import contextlib
from typing import Optional

from livekit.agents import Agent, ChatContext, ChatMessage

from .config import MemoryConfig
from .decision import DecisionPipeline
from .echo_filter import EchoFilter
from .memory import Hit, MemoryPort, format_memory_block
from .persona import PersonaPack
from .telemetry import Telemetry
from .trigger import ForceTrigger


class RiffAgent(Agent):
    def __init__(
        self,
        *,
        persona: PersonaPack,
        memory: MemoryPort,
        decision: DecisionPipeline,
        echo: EchoFilter,
        telemetry: Telemetry,
        trigger: ForceTrigger,
        config: MemoryConfig | None = None,
    ):
        super().__init__(instructions=persona.render_system_prompt())
        self._persona = persona
        self._memory = memory
        self._decision = decision
        self._echo = echo
        self._telemetry = telemetry
        self._trigger = trigger
        self._config = config or MemoryConfig()
        self._turn = 0
        self._session_ref = None
        self._trigger_task: Optional[asyncio.Task] = None
        self._last_heard: str = ""
        self._transcript_buffer: list[dict] = []

    def attach_session(self, agent_session) -> None:
        self._session_ref = agent_session
        if self._trigger_task is None:
            self._trigger_task = asyncio.create_task(self._watch_triggers())

    async def shutdown(self) -> None:
        if self._trigger_task:
            self._trigger_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._trigger_task
        await self._memory.push()

    async def index_turn(self, text: str, speaker: str, source: str = "audio", metadata: dict | None = None) -> str:
        """Index a turn into working memory. Returns turn_id."""
        self._turn += 1
        turn_id = f"turn-{self._turn}"
        await self._memory.remember(turn_id, f"[{speaker}] {text}")
        self._transcript_buffer.append({"id": turn_id, "speaker": speaker, "text": text, "source": source})
        await self._telemetry.emit("turn.indexed", turn=self._turn, text=text, speaker=speaker, source=source)
        return turn_id

    async def process_chat_turn(self, text: str, sender: str) -> dict | None:
        """Process a chat message. Returns reply dict or None if silent."""
        turn_id = await self.index_turn(text, sender, source="chat")

        memory_context = await self._ambient_context(text)
        decision = await self._decision.evaluate(text, force=False, context=memory_context)
        if not decision.speak:
            return None

        self._decision.mark_spoken()
        await self._telemetry.emit("chime.emitted", line=decision.line, source="chat")
        return {
            "reply": decision.line,
            "turn_id": turn_id,
            "decision": "spoke",
            "score": decision.llm_score,
            "latency_ms": 0,
        }

    def get_transcript(self, last_n: int = 20) -> list[dict]:
        """Return last N indexed turns."""
        return list(self._transcript_buffer)[-last_n:]

    async def _ambient_context(self, text: str) -> str:
        """Parallel retrieval across memory layers, budget-trimmed."""
        callbacks = await self._memory.callback(text, k=self._config.ambient_top_k)

        # Future: query episodic + semantic indexes separately
        return format_memory_block(
            working=callbacks,
            episodic=[],
            semantic=[],
            max_tokens=self._config.max_context_tokens,
            working_ratio=self._config.working_ratio,
            semantic_ratio=self._config.semantic_ratio,
            episodic_ratio=self._config.episodic_ratio,
            min_score=self._config.min_relevance_score,
        )

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        text = (new_message.text_content or "").strip()
        if not text:
            return

        if self._echo.is_muted() or self._echo.looks_like_echo(text):
            await self._telemetry.emit("turn.suppressed", reason="echo_or_muted", text=text)
            return

        self._last_heard = text
        await self.index_turn(text, speaker="audience", source="audio")

        memory_context = await self._ambient_context(text)
        decision = await self._decision.evaluate(text, force=False, context=memory_context)
        if decision.speak:
            await self._speak(decision.line)
        # The inversion: silence is the default outcome.

    async def _watch_triggers(self) -> None:
        while True:
            req = await self._trigger.wait()
            heard = req.hint or self._last_heard or "(no recent input)"
            await self._telemetry.emit("trigger.fired", hint=req.hint)
            memory_context = await self._ambient_context(heard)
            decision = await self._decision.evaluate(heard, force=True, context=memory_context)
            if decision.line:
                await self._speak(decision.line)
            else:
                await self._telemetry.emit("trigger.empty", reason=decision.reason)

    async def _speak(self, line: str) -> None:
        if not self._session_ref:
            return
        # Estimate ~70ms per char for TTS playback to bias the echo gate.
        estimated_ms = max(800, int(len(line) * 70))
        self._echo.begin_speaking(estimated_ms, line)
        self._decision.mark_spoken()
        await self._telemetry.emit("chime.emitted", line=line, est_ms=estimated_ms)
        await self._session_ref.say(line)
