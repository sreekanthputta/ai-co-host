"""MinimaxClient - thin async wrapper that returns a one-shot punchline.

LiveKit's AgentSession owns the streaming-into-TTS path for full responses;
this client is for the *forced-trigger* path where we generate a single line
synchronously and hand it to `agent_session.say(...)`.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Punchline:
    line: str
    score: float
    raw: str = ""


class MinimaxClient:
    def __init__(self, base_url: str, api_key: str, model: str = "MiniMax-M3"):
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def punchline(
        self,
        system_prompt: str,
        context: str,
        max_tokens: int = 50,
        temperature: float = 0.85,
    ) -> Punchline:
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{context}\n\n"
                        'Reply with strict JSON: {"line": "...", "score": 0.0-1.0}. '
                        "score = how confident you are this is funny. "
                        'If nothing good comes to mind, return {"line": "", "score": 0.0}.'
                    ),
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = (completion.choices[0].message.content or "").strip()
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> Punchline:
        match = _JSON_RE.search(raw)
        if not match:
            return Punchline(line="", score=0.0, raw=raw)
        try:
            data: dict[str, Any] = json.loads(match.group(0))
        except json.JSONDecodeError:
            return Punchline(line="", score=0.0, raw=raw)
        line = str(data.get("line", "") or "").strip()
        try:
            score = float(data.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        return Punchline(line=line, score=max(0.0, min(1.0, score)), raw=raw)
