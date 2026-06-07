# Parallel Build Plan — Sub-Agent Task Dispatch

This document breaks the Riff build into **independent, parallelizable modules**. Each task is self-contained: own files, own interfaces, own tests. No task blocks another unless explicitly marked.

---

## Ground Rules for All Agents

1. **Abstraction first** — define a Protocol/ABC interface, then implement against it.
2. **Unit tests mandatory** — every public function tested with fakes. Tests pass before done.
3. **No cross-module imports during build** — each module talks to interfaces, not concrete classes. Integration wiring happens last.
4. **File ownership** — each task owns specific files. No two tasks touch the same file.
5. **Consistent patterns** — all modules use `@dataclass` for config, `Protocol` for interfaces, `pytest` for tests.

---

## Dependency Graph

```
                    ┌─────────────┐
                    │  T1: Config │  (no deps, start immediately)
                    │  & Persona  │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ T2: Memory│   │ T3: LLM   │   │ T4: Echo  │   (parallel, no deps on each other)
    │  Adapter  │   │  Client   │   │  Filter   │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │                │                │
          │          ┌─────▼─────┐          │
          │          │ T5: Decis │          │
          └─────────►│  Pipeline ├◄─────────┘   (depends on T2, T3, T4 interfaces only)
                     └─────┬─────┘
                           │
                     ┌─────▼─────┐
                     │ T6: Agent │   (depends on T5 interface)
                     │  (core)   │
                     └─────┬─────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ T7: API   │   │ T8: Telem │   │ T9: Index │   (parallel, depend on T6 interface)
    │  Server   │   │  etry     │   │  Builder  │
    └───────────┘   └───────────┘   └───────────┘
                           │
                     ┌─────▼─────┐
                     │T10: Wire  │   (final: integrates everything)
                     │  & E2E    │
                     └───────────┘
```

---

## Task Definitions

### T1: Config & Persona Loader

**Files owned:**
- `src/riff/config.py`
- `src/riff/persona.py`
- `personas/riff.yaml`
- `tests/test_config.py`
- `tests/test_persona.py`

**Delivers:**
```python
# config.py
@dataclass
class MemoryConfig:
    max_context_tokens: int = 8000
    working_ratio: float = 0.5
    semantic_ratio: float = 0.3
    episodic_ratio: float = 0.2
    min_relevance_score: float = 0.4
    dedup_threshold: float = 0.90
    chars_per_token: int = 4
    moss_timeout_ms: int = 500
    ambient_top_k: int = 5
    deep_recall_top_k: int = 8

@dataclass
class AgentConfig:
    cooldown_seconds: float = 8.0
    min_word_count: int = 3
    quality_threshold: float = 7.0
    max_tokens_reply: int = 50

@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8765

# persona.py
@dataclass
class PersonaPack:
    name: str
    voice_id: str
    system_prompt: str
    memory_config: MemoryConfig
    agent_config: AgentConfig
    indexes: list[str]

    @classmethod
    def from_yaml(cls, path: Path) -> "PersonaPack": ...
```

**Tests:**
- YAML load + schema validation
- Missing fields → sensible defaults
- Invalid values → clear errors
- Config dataclass serialization round-trip

**Depends on:** Nothing. Start immediately.

---

### T2: Memory Adapter (Moss Wrapper)

**Files owned:**
- `src/riff/memory.py`
- `tests/test_memory.py`
- `tests/fakes/fake_memory.py`

**Delivers:**
```python
# memory.py
class MemoryResult:
    docs: list[MemoryDoc]

class MemoryDoc:
    id: str
    text: str
    score: float
    metadata: dict

class Memory(Protocol):
    async def index_turn(self, turn_id: str, text: str, speaker: str, source: str, metadata: dict | None) -> None: ...
    async def query(self, text: str, top_k: int) -> MemoryResult: ...
    async def ensure_ready(self) -> None: ...

class MossMemory:
    """Real implementation wrapping MossClient + session."""
    def __init__(self, client: MossClient, index_name: str, config: MemoryConfig): ...
    async def index_turn(...) -> None: ...
    async def query(...) -> MemoryResult: ...
    async def safe_query(...) -> MemoryResult: ...  # with timeout + fallback
    async def ensure_ready(self) -> None: ...

class WorkingMemory:
    """Session-scoped memory for current show."""
    ...

class SemanticMemory:
    """Cloud index for tropes/patterns."""
    ...

# Budget trimming + dedup (pure functions, easily testable)
def budget_trim(docs: list[MemoryDoc], max_tokens: int, min_score: float, chars_per_token: int) -> list[str]: ...
def deduplicate(docs: list[MemoryDoc], threshold: float) -> list[MemoryDoc]: ...
def format_memory_block(working: list, episodic: list, semantic: list, config: MemoryConfig) -> str: ...
```

**Tests:**
- `budget_trim`: respects token limit, drops below min_score, stops at budget
- `deduplicate`: removes near-duplicates, keeps unique
- `format_memory_block`: correct tiered formatting, empty inputs handled
- `safe_query`: returns empty on timeout (mock async timeout)
- `FakeMemory`: passes same interface contract tests as real

**Depends on:** T1 (MemoryConfig dataclass). Can start in parallel — just import the dataclass.

---

### T3: LLM Client

**Files owned:**
- `src/riff/llm_client.py`
- `tests/test_llm_client.py`
- `tests/fakes/fake_llm.py`

**Delivers:**
```python
# llm_client.py
@dataclass
class LLMResponse:
    text: str
    score: float | None  # self-rated quality score
    latency_ms: float

class LLMClient(Protocol):
    async def generate(self, system: str, user: str, max_tokens: int) -> LLMResponse: ...
    async def generate_candidates(self, system: str, user: str, n: int, max_tokens: int) -> list[LLMResponse]: ...

class MinimaxClient:
    """Real implementation using OpenAI-compatible API."""
    def __init__(self, base_url: str, api_key: str, model: str): ...
    async def generate(...) -> LLMResponse: ...
    async def generate_candidates(...) -> list[LLMResponse]: ...
    async def pre_warm(self) -> None: ...

class GroqClient:
    """Alternative: Groq for faster inference."""
    ...
```

**Tests:**
- `generate`: returns structured response, respects max_tokens in prompt
- `generate_candidates`: returns N candidates sorted by score
- `pre_warm`: doesn't error on cold start
- `FakeLLM`: returns scripted responses, tracks call count

**Depends on:** Nothing. Start immediately.

---

### T4: Echo Filter

**Files owned:**
- `src/riff/echo_filter.py`
- `tests/test_echo_filter.py`

**Delivers:**
```python
# echo_filter.py
class EchoFilter:
    def __init__(self, similarity_threshold: float = 0.7, suppression_window_ms: int = 300): ...
    def start_speaking(self, text: str) -> None: ...
    def stop_speaking(self) -> None: ...
    def is_muted(self) -> bool: ...
    def is_echo(self, incoming_text: str) -> bool: ...
    def should_process(self, incoming_text: str) -> bool: ...  # combines is_muted + is_echo
```

**Tests:**
- `is_muted`: True during TTS playback + suppression window
- `is_echo`: True when incoming text matches last spoken line >70%
- `should_process`: False when muted OR echo, True otherwise
- Edge: very short texts, empty strings, exact match vs fuzzy
- Timing: suppression window expiry

**Depends on:** Nothing. Start immediately.

---

### T5: Decision Pipeline

**Files owned:**
- `src/riff/decision.py`
- `tests/test_decision.py`

**Delivers:**
```python
# decision.py
@dataclass
class DecisionResult:
    should_speak: bool
    text: str | None
    score: float
    memory_context: str
    latency_ms: float
    gate_passed: str  # which gate let it through or blocked it

class TriggerGate:
    """Fast, no-LLM gate checks."""
    def __init__(self, config: AgentConfig): ...
    def should_proceed(self, text: str) -> bool: ...  # cooldown + word count
    def record_speech(self) -> None: ...  # resets cooldown
    def force_open(self) -> None: ...  # bypass for /trigger

class DecisionPipeline:
    def __init__(self, gate: TriggerGate, memory: Memory, llm: LLMClient, echo: EchoFilter, config: AgentConfig): ...
    async def evaluate(self, turn_text: str, speaker: str) -> DecisionResult: ...
```

**Tests:**
- `TriggerGate.should_proceed`: False during cooldown, False for short text, True after cooldown
- `TriggerGate.force_open`: bypasses all checks
- `DecisionPipeline.evaluate`: full flow with FakeMemory + FakeLLM
  - Low relevance retrieval → stays silent
  - High relevance + high LLM score → speaks
  - High relevance + low LLM score → stays silent (quality gate)
  - Echo detected → stays silent
  - Muted → stays silent

**Depends on:** T2 (Memory Protocol), T3 (LLMClient Protocol), T4 (EchoFilter). Uses only interfaces — can build against fakes immediately.

---

### T6: Agent Core

**Files owned:**
- `src/riff/agent.py`
- `tests/test_agent.py`

**Delivers:**
```python
# agent.py
class RiffAgent(Agent):
    def __init__(self, persona: PersonaPack, memory: Memory, decision: DecisionPipeline, telemetry: Telemetry): ...
    
    async def index_turn(self, text: str, speaker: str, source: str, metadata: dict | None) -> str: ...
    async def process_audio_turn(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None: ...
    async def process_chat_turn(self, text: str, sender: str) -> DecisionResult | None: ...
    async def on_user_turn_completed(self, turn_ctx, new_message) -> None: ...
```

**Tests:**
- `index_turn`: increments turn counter, calls memory.index_turn with correct args
- `process_audio_turn`: calls decision pipeline, speaks if result.should_speak
- `process_chat_turn`: same pipeline, returns text (no TTS)
- `on_user_turn_completed`: does NOT call super (inverted loop verified)
- Wiring: all dependencies injected, no direct Moss/LLM imports

**Depends on:** T1 (PersonaPack), T5 (DecisionPipeline interface). Can start once T5's interface is defined (not its full implementation).

---

### T7: API Server

**Files owned:**
- `src/riff/api.py`
- `tests/test_api.py`

**Delivers:**
```python
# api.py (FastAPI)
# POST /message       — single chat message, optional respond
# POST /message/batch — bulk index
# POST /trigger       — force chime
# POST /mute          — silence N seconds
# POST /unmute        — resume
# GET  /status        — current state
# GET  /transcript    — last N turns
# WS   /events       — live state stream
```

**Tests (via FastAPI TestClient, no real agent):**
- `/message` with `respond: true` → returns reply structure
- `/message` with `respond: false` → returns indexed confirmation
- `/message/batch` → indexes all, returns turn_ids
- `/trigger` → calls agent force chime
- `/mute` + `/unmute` → state transitions
- `/status` → returns current state JSON
- `/transcript` → returns last N turns
- Invalid payloads → 422 with clear errors

**Depends on:** T6 (Agent interface — injected as `app.state.agent`). Can build against a fake agent.

---

### T8: Telemetry

**Files owned:**
- `src/riff/telemetry.py`
- `tests/test_telemetry.py`

**Delivers:**
```python
# telemetry.py
@dataclass
class TimingEvent:
    stage: str  # "retrieval", "llm", "tts", "total"
    duration_ms: float
    metadata: dict

class Telemetry(Protocol):
    def record(self, event: TimingEvent) -> None: ...
    def get_stats(self) -> dict: ...  # p50, p95, counts

class InMemoryTelemetry:
    """Collects timing events, computes percentiles."""
    def __init__(self, window_size: int = 100): ...
    def record(self, event: TimingEvent) -> None: ...
    def get_stats(self) -> dict: ...
    def get_events(self, last_n: int = 20) -> list[TimingEvent]: ...
```

**Tests:**
- `record`: stores events, respects window_size
- `get_stats`: correct p50/p95 on known data
- Empty state → returns zeros, not errors
- Observer pattern: multiple listeners notified

**Depends on:** Nothing. Start immediately.

---

### T9: Index Builder

**Files owned:**
- `build_index.py`
- `data/comedy_tropes.json`
- `tests/test_build_index.py`

**Delivers:**
```python
# build_index.py
async def build_tropes_index(client: MossClient, data_path: Path, index_name: str) -> None: ...
async def validate_tropes_data(data_path: Path) -> list[dict]: ...  # schema validation

# data/comedy_tropes.json — structured comedy corpus
# Each entry: {"id": "...", "text": "[Context Header] content...", "metadata": {...}}
```

**Tests:**
- `validate_tropes_data`: valid JSON, required fields present, text has context headers
- Each entry has unique id
- Metadata has required keys (type, category, strength)
- Text starts with `[` context header pattern

**Depends on:** Nothing. Start immediately. (Uses real Moss client for integration, but unit tests validate data structure only.)

---

### T10: Integration Wiring & E2E

**Files owned:**
- `voice_agent.py` (entry point)
- `src/riff/__init__.py`
- `tests/test_integration.py` (marked `@pytest.mark.integration`)

**Delivers:**
- Wires all modules together with real implementations
- `entrypoint()` function that creates real MossClient, MinimaxClient, builds Agent, starts API
- E2E smoke test: send a /message, verify response shape

**Depends on:** ALL other tasks complete and tested.

---

## Parallel Execution Plan

### Wave 1 (all start immediately, zero dependencies)

| Task | Agent | Estimated Time |
|---|---|---|
| T1: Config & Persona | Agent A | 30 min |
| T3: LLM Client | Agent B | 30 min |
| T4: Echo Filter | Agent C | 30 min |
| T8: Telemetry | Agent D | 30 min |
| T9: Index Builder | Agent E | 45 min |

### Wave 2 (needs interfaces from Wave 1)

| Task | Agent | Estimated Time | Needs |
|---|---|---|---|
| T2: Memory Adapter | Agent A | 45 min | T1 config dataclass |
| T5: Decision Pipeline | Agent B | 45 min | T2, T3, T4 interfaces (not implementations) |

### Wave 3 (needs Wave 2)

| Task | Agent | Estimated Time | Needs |
|---|---|---|---|
| T6: Agent Core | Agent A | 45 min | T1, T5 interface |
| T7: API Server | Agent B | 45 min | T6 interface |

### Wave 4 (final integration)

| Task | Agent | Estimated Time | Needs |
|---|---|---|---|
| T10: Integration & E2E | Agent A | 30 min | All |

### Total wall-clock time: ~2.5 hours (vs ~6 hours sequential)

---

## Interface Contracts (shared across all agents)

All agents must agree on these interfaces. Defined upfront so parallel work doesn't diverge.

```python
# src/riff/protocols.py — ALL agents reference this file (read-only for them)

from typing import Protocol
from dataclasses import dataclass


@dataclass
class MemoryDoc:
    id: str
    text: str
    score: float
    metadata: dict


@dataclass
class MemoryResult:
    docs: list[MemoryDoc]


@dataclass
class LLMResponse:
    text: str
    score: float | None
    latency_ms: float


@dataclass
class DecisionResult:
    should_speak: bool
    text: str | None
    score: float
    memory_context: str
    latency_ms: float
    gate_passed: str


@dataclass
class TimingEvent:
    stage: str
    duration_ms: float
    metadata: dict


class Memory(Protocol):
    async def index_turn(self, turn_id: str, text: str, speaker: str, source: str, metadata: dict | None = None) -> None: ...
    async def query(self, text: str, top_k: int = 5) -> MemoryResult: ...
    async def ensure_ready(self) -> None: ...


class LLMClient(Protocol):
    async def generate(self, system: str, user: str, max_tokens: int = 50) -> LLMResponse: ...
    async def generate_candidates(self, system: str, user: str, n: int = 3, max_tokens: int = 50) -> list[LLMResponse]: ...


class Telemetry(Protocol):
    def record(self, event: TimingEvent) -> None: ...
    def get_stats(self) -> dict: ...
```

---

## File Ownership Matrix

No two tasks write to the same file. This prevents merge conflicts in parallel work.

| File | Owner Task |
|---|---|
| `src/riff/__init__.py` | T10 |
| `src/riff/protocols.py` | Pre-created (shared, read-only) |
| `src/riff/config.py` | T1 |
| `src/riff/persona.py` | T1 |
| `src/riff/memory.py` | T2 |
| `src/riff/llm_client.py` | T3 |
| `src/riff/echo_filter.py` | T4 |
| `src/riff/decision.py` | T5 |
| `src/riff/agent.py` | T6 |
| `src/riff/api.py` | T7 |
| `src/riff/telemetry.py` | T8 |
| `build_index.py` | T9 |
| `data/comedy_tropes.json` | T9 |
| `personas/riff.yaml` | T1 |
| `voice_agent.py` | T10 |
| `tests/test_config.py` | T1 |
| `tests/test_persona.py` | T1 |
| `tests/test_memory.py` | T2 |
| `tests/test_llm_client.py` | T3 |
| `tests/test_echo_filter.py` | T4 |
| `tests/test_decision.py` | T5 |
| `tests/test_agent.py` | T6 |
| `tests/test_api.py` | T7 |
| `tests/test_telemetry.py` | T8 |
| `tests/test_build_index.py` | T9 |
| `tests/test_integration.py` | T10 |
| `tests/fakes/fake_memory.py` | T2 |
| `tests/fakes/fake_llm.py` | T3 |
| `tests/conftest.py` | T10 |

---

## Definition of Done (per task)

- [ ] All owned files created
- [ ] Protocol/interface respected (type-checks against `protocols.py`)
- [ ] Unit tests written for every public function
- [ ] `pytest tests/test_{module}.py` passes with 0 failures
- [ ] No imports of other task's concrete implementations (only protocols)
- [ ] Docstring on the module explaining its responsibility (one line)
