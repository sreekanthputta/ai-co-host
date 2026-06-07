"""MossMemory - Adapter over MossClient + live session.

Provides two views:
  - tropes index (read-only knowledge base) for similarity gating
  - session memory (per-show transcript) for callback retrieval
"""
from __future__ import annotations
from dataclasses import dataclass
from difflib import SequenceMatcher
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


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    return len(text) // chars_per_token


def budget_trim(docs: list[Hit], max_tokens: int, min_score: float = 0.4, chars_per_token: int = 4) -> list[str]:
    result: list[str] = []
    used = 0
    for doc in docs:
        if doc.score < min_score:
            continue
        cost = estimate_tokens(doc.text, chars_per_token)
        if used + cost > max_tokens:
            if not result:
                result.append(doc.text)
            break
        result.append(doc.text)
        used += cost
    return result


def deduplicate(docs: list[Hit], threshold: float = 0.90) -> list[Hit]:
    kept: list[Hit] = []
    for doc in docs:
        if any(SequenceMatcher(None, doc.text, k.text).ratio() > threshold for k in kept):
            continue
        kept.append(doc)
    return kept


def _qualifying_hits(hits: list[Hit], min_score: float) -> list[Hit]:
    return [h for h in hits if h.score >= min_score]


def format_memory_block(
    working: list[Hit],
    episodic: list[Hit],
    semantic: list[Hit],
    max_tokens: int = 8000,
    working_ratio: float = 0.5,
    semantic_ratio: float = 0.3,
    episodic_ratio: float = 0.2,
    min_score: float = 0.4,
) -> str:
    working_budget = int(max_tokens * working_ratio)
    semantic_budget = int(max_tokens * semantic_ratio)
    episodic_budget = int(max_tokens * episodic_ratio)

    w_texts = budget_trim(working, working_budget, min_score)
    e_texts = budget_trim(episodic, episodic_budget, min_score)
    s_texts = budget_trim(semantic, semantic_budget, min_score)

    sections: list[str] = []

    def _tier(texts: list[str], hits: list[Hit], high_header: str):
        qualified = _qualifying_hits(hits, min_score)
        paired = list(zip(texts, qualified))
        high = [t for t, h in paired if h.score >= 0.6]
        low = [t for t, h in paired if h.score < 0.6]
        if high:
            sections.append(f"{high_header}\n" + "\n".join(f"- {t}" for t in high))
        if low:
            sections.append("VAGUE RECALL (may be imprecise):\n" + "\n".join(f"- {t}" for t in low))

    _tier(w_texts, working, "CALLBACKS FROM THIS SHOW:")
    _tier(e_texts, episodic, "CALLBACKS FROM THIS SHOW:")
    _tier(s_texts, semantic, "COMEDY PATTERNS:")

    return "\n\n".join(sections) if sections else ""


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
