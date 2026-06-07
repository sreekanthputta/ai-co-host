"""Riff entry point - wires every collaborator and starts the LiveKit worker.

    python voice_agent.py console     # talk via terminal
    python voice_agent.py dev         # accept inbound calls
"""
from __future__ import annotations
import asyncio
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from livekit.plugins import openai, silero
from livekit.agents import (
    AgentSession, JobContext, WorkerOptions, cli, inference,
)
from moss import MossClient

from src.riff.agent import RiffAgent
from src.riff.api import build_app
from src.riff.decision import DecisionPipeline
from src.riff.echo_filter import EchoFilter
from src.riff.llm_client import MinimaxClient
from src.riff.memory import MossMemory
from src.riff.persona import PersonaPack
from src.riff.telemetry import Telemetry
from src.riff.trigger import ForceTrigger

load_dotenv(override=True)

ROOT = Path(__file__).parent
PERSONA_PATH = os.getenv("RIFF_PERSONA", str(ROOT / "personas" / "riff.yaml"))
INDEX = os.getenv("MOSS_INDEX_NAME", "riff-tropes")
API_PORT = int(os.getenv("RIFF_API_PORT", "8765"))

MINIMAX_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.minimax.io/v1")
MINIMAX_API_KEY = os.environ["OPENAI_API_KEY"]


async def entrypoint(ctx: JobContext):
    persona = PersonaPack.from_yaml(PERSONA_PATH)
    telemetry = Telemetry()
    echo = EchoFilter()
    trigger = ForceTrigger()

    moss = MossClient(os.environ["MOSS_PROJECT_ID"], os.environ["MOSS_PROJECT_KEY"])
    await moss.load_index(INDEX)
    session_index = f"call-{ctx.room.name}"
    from moss import DocumentInfo as MossDoc
    try:
        await moss.create_index(session_index, [MossDoc(id="seed", text="session start")])
    except Exception:
        pass
    await moss.load_index(session_index)
    memory = MossMemory(moss, INDEX, session_index)

    llm = MinimaxClient(MINIMAX_BASE_URL, MINIMAX_API_KEY)
    decision = DecisionPipeline(persona, memory, llm, echo, telemetry)

    agent = RiffAgent(
        persona=persona, memory=memory, decision=decision,
        echo=echo, telemetry=telemetry, trigger=trigger,
    )

    state = {"persona": persona.name}
    api = build_app(trigger=trigger, telemetry=telemetry, echo=echo, state=state)
    api_server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=API_PORT, log_level="warning"))
    api_task = asyncio.create_task(api_server.serve())

    async def shutdown():
        api_server.should_exit = True
        await agent.shutdown()
        api_task.cancel()
    ctx.add_shutdown_callback(shutdown)

    await ctx.connect()

    agent_session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=openai.LLM(
            model="MiniMax-M3",
            base_url=MINIMAX_BASE_URL,
            api_key=MINIMAX_API_KEY,
            extra_body={"thinking": {"type": "disabled"}},
        ),
        tts=inference.TTS(model="cartesia/sonic", voice=persona.voice_id),
        vad=silero.VAD.load(),
    )
    agent.attach_session(agent_session)
    await agent_session.start(agent=agent, room=ctx.room)

    await telemetry.emit("session.started", persona=persona.name, room=ctx.room.name)
    # No greeting on stage - Riff stays silent until something earns a chime.


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
