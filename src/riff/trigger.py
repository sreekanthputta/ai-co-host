"""ForceTrigger - Command pattern. Captures intent to chime regardless of gate."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass


@dataclass
class TriggerRequest:
    hint: str = ""
    bypass_gate: bool = True
    bypass_cooldown: bool = True


class ForceTrigger:
    """Single-slot command queue - newest force overrides any pending one."""

    def __init__(self):
        self._pending: TriggerRequest | None = None
        self._event = asyncio.Event()

    def fire(self, hint: str = "") -> None:
        self._pending = TriggerRequest(hint=hint)
        self._event.set()

    def take(self) -> TriggerRequest | None:
        req = self._pending
        self._pending = None
        self._event.clear()
        return req

    async def wait(self) -> TriggerRequest:
        await self._event.wait()
        req = self.take()
        assert req is not None
        return req
