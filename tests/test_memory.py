import sys
import types

import pytest

from src.riff.memory import MossMemory


class _Doc:
    def __init__(self, id, text, score):
        self.id = id
        self.text = text
        self.score = score


class _Result:
    def __init__(self, docs):
        self.docs = docs


class _FakeClient:
    def __init__(self, score=0.9):
        self.score = score
        self.queries: list[tuple[str, str]] = []

    async def query(self, index, text, opts):
        self.queries.append((index, text))
        return _Result([_Doc("d1", "match", self.score)])


class _FakeSession:
    def __init__(self):
        self.docs = []
        self.pushed = 0
        self.queried: list[str] = []

    async def add_docs(self, docs):
        self.docs.extend(docs)

    async def query(self, text, opts):
        self.queried.append(text)
        return _Result([_Doc("turn-1", "earlier line", 0.6)])

    async def push_index(self):
        self.pushed += 1


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
    mem = MossMemory(client, "tropes", _FakeSession())
    score = await mem.gate_score("Tulsa accountant")
    assert score == pytest.approx(0.84)
    assert client.queries == [("tropes", "Tulsa accountant")]


async def test_gate_empty_text_returns_zero():
    mem = MossMemory(_FakeClient(), "tropes", _FakeSession())
    assert await mem.gate_score("   ") == 0.0


async def test_callback_returns_hits():
    mem = MossMemory(_FakeClient(), "tropes", _FakeSession())
    hits = await mem.callback("earlier", k=1)
    assert len(hits) == 1
    assert hits[0].text == "earlier line"


async def test_remember_pushes_doc():
    sess = _FakeSession()
    mem = MossMemory(_FakeClient(), "tropes", sess)
    await mem.remember("turn-3", "what they said")
    assert len(sess.docs) == 1
    assert sess.docs[0].id == "turn-3"


async def test_push_swallows_runtime_error():
    sess = _FakeSession()

    async def boom():
        raise RuntimeError("no session")
    sess.push_index = boom
    mem = MossMemory(_FakeClient(), "tropes", sess)
    await mem.push()  # must not raise
