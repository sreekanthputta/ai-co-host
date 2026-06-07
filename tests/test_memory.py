import sys
import types

import pytest

from src.riff.memory import (
    MossMemory,
    Hit,
    estimate_tokens,
    budget_trim,
    deduplicate,
    format_memory_block,
)


class _Doc:
    def __init__(self, id, text, score):
        self.id = id
        self.text = text
        self.score = score


class _Result:
    def __init__(self, docs):
        self.docs = docs


class _FakeClient:
    def __init__(self, score=0.9, session_score=0.6):
        self.score = score
        self.session_score = session_score
        self.queries: list[tuple[str, str]] = []
        self.added_docs: list[tuple[str, list]] = []

    async def query(self, index, text, opts):
        self.queries.append((index, text))
        if index == "session":
            return _Result([_Doc("turn-1", "earlier line", self.session_score)])
        return _Result([_Doc("d1", "match", self.score)])

    async def add_docs(self, index, docs):
        self.added_docs.append((index, docs))


@pytest.fixture(autouse=True)
def _stub_moss(monkeypatch):
    """Stub the `moss` module so MossMemory imports its types without the real SDK."""
    if "moss" in sys.modules:
        return
    fake = types.ModuleType("moss")

    class _DocumentInfo:
        def __init__(self, id, text):
            self.id = id
            self.text = text

    class _QueryOptions:
        def __init__(self, top_k):
            self.top_k = top_k

    fake.DocumentInfo = _DocumentInfo
    fake.QueryOptions = _QueryOptions
    monkeypatch.setitem(sys.modules, "moss", fake)


async def test_gate_score_returns_first_hit():
    client = _FakeClient(score=0.84)
    mem = MossMemory(client, "tropes", "session")
    score = await mem.gate_score("Tulsa accountant")
    assert score == pytest.approx(0.84)
    assert client.queries == [("tropes", "Tulsa accountant")]


async def test_gate_empty_text_returns_zero():
    mem = MossMemory(_FakeClient(), "tropes", "session")
    assert await mem.gate_score("   ") == 0.0


async def test_callback_returns_hits():
    mem = MossMemory(_FakeClient(), "tropes", "session")
    hits = await mem.callback("earlier", k=1)
    assert len(hits) == 1
    assert hits[0].text == "earlier line"


async def test_remember_adds_doc_to_session_index():
    client = _FakeClient()
    mem = MossMemory(client, "tropes", "session")
    await mem.remember("turn-3", "what they said")
    assert len(client.added_docs) == 1
    assert client.added_docs[0][0] == "session"
    assert client.added_docs[0][1][0].id == "turn-3"


async def test_push_is_noop():
    mem = MossMemory(_FakeClient(), "tropes", "session")
    await mem.push()  # must not raise


# --- estimate_tokens tests ---

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_known():
    assert estimate_tokens("abcdefgh") == 2  # 8 // 4
    assert estimate_tokens("abcdefgh", chars_per_token=2) == 4


# --- budget_trim tests ---

def test_budget_trim_empty():
    assert budget_trim([], max_tokens=100) == []


def test_budget_trim_all_fit():
    docs = [Hit("hello", 0.9, "1"), Hit("world", 0.8, "2")]
    result = budget_trim(docs, max_tokens=100)
    assert result == ["hello", "world"]


def test_budget_trim_overflow_stops():
    docs = [
        Hit("a" * 20, 0.9, "1"),  # 5 tokens
        Hit("b" * 20, 0.8, "2"),  # 5 tokens
        Hit("c" * 20, 0.7, "3"),  # 5 tokens
    ]
    result = budget_trim(docs, max_tokens=9)
    assert result == ["a" * 20]


def test_budget_trim_below_min_score_dropped():
    docs = [Hit("good", 0.9, "1"), Hit("bad", 0.3, "2")]
    result = budget_trim(docs, max_tokens=1000)
    assert result == ["good"]


def test_budget_trim_first_doc_over_budget_included():
    docs = [Hit("a" * 100, 0.9, "1")]  # 25 tokens, budget is 5
    result = budget_trim(docs, max_tokens=5)
    assert result == ["a" * 100]


def test_budget_trim_first_doc_over_budget_but_low_score():
    docs = [Hit("a" * 100, 0.3, "1")]
    result = budget_trim(docs, max_tokens=5)
    assert result == []


# --- deduplicate tests ---

def test_deduplicate_empty():
    assert deduplicate([]) == []


def test_deduplicate_no_dupes():
    docs = [Hit("apples are red", 0.9, "1"), Hit("bananas are yellow", 0.8, "2")]
    assert deduplicate(docs) == docs


def test_deduplicate_exact():
    docs = [Hit("same text", 0.9, "1"), Hit("same text", 0.8, "2")]
    result = deduplicate(docs)
    assert len(result) == 1
    assert result[0].id == "1"


def test_deduplicate_95_percent_similar():
    base = "a" * 100
    docs = [Hit(base, 0.9, "1"), Hit(base[:95] + "bbbbb", 0.8, "2")]
    result = deduplicate(docs)
    assert len(result) == 1


def test_deduplicate_80_percent_similar_kept():
    base = "a" * 100
    docs = [Hit(base, 0.9, "1"), Hit(base[:80] + "b" * 20, 0.8, "2")]
    result = deduplicate(docs)
    assert len(result) == 2


def test_deduplicate_order_preserved():
    docs = [Hit("first", 0.9, "1"), Hit("second", 0.8, "2"), Hit("third", 0.7, "3")]
    result = deduplicate(docs)
    assert [h.id for h in result] == ["1", "2", "3"]


# --- format_memory_block tests ---

def test_format_memory_block_all_empty():
    assert format_memory_block([], [], []) == ""


def test_format_memory_block_only_working():
    hits = [Hit("callback joke", 0.8, "1")]
    result = format_memory_block(hits, [], [])
    assert "CALLBACKS FROM THIS SHOW:" in result
    assert "- callback joke" in result


def test_format_memory_block_all_layers():
    w = [Hit("working hit", 0.7, "1")]
    e = [Hit("episodic hit", 0.7, "2")]
    s = [Hit("semantic hit", 0.7, "3")]
    result = format_memory_block(w, e, s)
    assert "CALLBACKS FROM THIS SHOW:" in result
    assert "COMEDY PATTERNS:" in result
    assert "working hit" in result
    assert "episodic hit" in result
    assert "semantic hit" in result


def test_format_memory_block_tiered():
    w = [Hit("high conf", 0.8, "1"), Hit("low conf", 0.45, "2")]
    result = format_memory_block(w, [], [])
    assert "CALLBACKS FROM THIS SHOW:" in result
    assert "- high conf" in result
    assert "VAGUE RECALL (may be imprecise):" in result
    assert "- low conf" in result


def test_format_memory_block_budget_overflow():
    large = [Hit("x" * 1000, 0.9, str(i)) for i in range(50)]
    result = format_memory_block(large, [], [], max_tokens=100)
    assert "x" * 1000 in result  # first doc included even over budget
    lines = [l for l in result.split("\n") if l.startswith("- ")]
    assert len(lines) <= 5  # budget limits output


def test_format_memory_block_all_below_min_score():
    hits = [Hit("low", 0.2, "1"), Hit("lower", 0.1, "2")]
    assert format_memory_block(hits, hits, hits) == ""
