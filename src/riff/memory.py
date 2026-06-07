"""MossMemory - Adapter over MossClient + live session.

Provides two views:
  - tropes index (read-only knowledge base) for similarity gating
  - session memory (per-show transcript) for callback retrieval
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Hit:
    text: str
    score: float
    id: str


class MemoryPort(Protocol):
    async def gate_score(self, text: str) -> float: ...
    async def callback(self, text: str, k: int = 2) -> list[Hit]: ...
    async def remember(self, doc_id: str, text: str) -> None: ...
    async def push(self) -> None: ...


class MossMemory:
    """Adapts the Moss SDK to the MemoryPort the decision pipeline expects."""

    def __init__(self, client, tropes_index: str, session):
        self._client = client
        self._tropes = tropes_index
        self._session = session

    async def gate_score(self, text: str) -> float:
        from moss import QueryOptions
        if not text.strip():
            return 0.0
        res = await self._client.query(self._tropes, text, QueryOptions(top_k=1))
        if not res.docs:
            return 0.0
        first = res.docs[0]
        return float(getattr(first, "score", 0.0) or 0.0)

    async def callback(self, text: str, k: int = 2) -> list[Hit]:
        from moss import QueryOptions
        res = await self._session.query(text, QueryOptions(top_k=k))
        return [
            Hit(text=d.text, score=float(getattr(d, "score", 0.0) or 0.0), id=str(getattr(d, "id", "")))
            for d in res.docs
        ]

    async def remember(self, doc_id: str, text: str) -> None:
        from moss import DocumentInfo
        await self._session.add_docs([DocumentInfo(id=doc_id, text=text)])

    async def push(self) -> None:
        try:
            await self._session.push_index()
        except RuntimeError:
            pass
