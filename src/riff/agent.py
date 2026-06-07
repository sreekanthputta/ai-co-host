"""RiffAgent - LiveKit Agent with Dependency Injection.

The agent itself stays thin: it routes events into the DecisionPipeline,
handles forced triggers, and hands lines to the AgentSession for TTS.
"""
from __future__ import annotations
import asyncio
import contextlib
from typing import Optional

from livekit.agents import Agent, ChatContext, ChatMessage

from .decision import DecisionPipeline
from .echo_filter import EchoFilter
from .memory import MemoryPort
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
    ):
        super().__init__(instructions=persona.render_system_prompt())
        self._persona = persona
        self._memory = memory
        self._decision = decision
        self._echo = echo
        self._telemetry = telemetry
        self._trigger = trigger
        self._turn = 0
        self._session_ref = None
        self._trigger_task: Optional[asyncio.Task] = None
        self._last_heard: str = ""

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

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        text = (new_message.text_content or "").strip()
        if not text:
            return

        if self._echo.is_muted() or self._echo.looks_like_echo(text):
            await self._telemetry.emit("turn.suppressed", reason="echo_or_muted", text=text)
            return

        self._turn += 1
        self._last_heard = text
        await self._memory.remember(f"turn-{self._turn}", text)
        await self._telemetry.emit("turn.indexed", turn=self._turn, text=text)

        decision = await self._decision.evaluate(text, force=False)
        if decision.speak:
            await self._speak(decision.line)
        # The inversion: silence is the default outcome.

    async def _watch_triggers(self) -> None:
        while True:
            req = await self._trigger.wait()
            heard = req.hint or self._last_heard or "(no recent input)"
            await self._telemetry.emit("trigger.fired", hint=req.hint)
            decision = await self._decision.evaluate(heard, force=True)
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
