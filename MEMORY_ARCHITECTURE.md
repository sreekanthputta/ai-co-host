# Memory Architecture — Riff's Living Memory System

**Goal:** Make Riff remember like a human comedian — callbacks from earlier in the set, patterns from past shows, and audience details that make crowd work personal.

---

## The Problem

Raw transcript retrieval is useless for comedy:

```
Bad:  "Yeah I think we should go with option B for that"
      → LLM has no idea: who? when? what option B? useless.

Good: "[Show 2024-06-07 | Audience: Maya, dental hygienist from Tulsa, single]
       Host said he hasn't had a date since the Obama administration."
      → LLM can now callback: "Has anyone in Tulsa flossed since the Obama administration?"
```

**Every retrieved chunk must be independently interpretable.**

---

## Three Memory Stores

### 1. Working Memory — "This Show"

| | |
|---|---|
| **Index** | `show-{room}` (Moss session) |
| **Content** | Every turn from current show, verbatim + speaker |
| **Lifecycle** | Created at show start, pushed to cloud at show end, then discarded |
| **Purpose** | Same-show callbacks ("she said X 5 minutes ago") |

```python
# Indexed on every turn:
DocumentInfo(
    id=f"turn-{n}",
    text=f"[{speaker}] {text}",
    metadata={"speaker": speaker, "timestamp": str(elapsed_seconds)}
)
```

### 2. Episodic Memory — "Past Shows"

| | |
|---|---|
| **Index** | `riff-episodes` (Moss cloud) |
| **Content** | Processed highlights from past shows — what landed, who was there |
| **Lifecycle** | Persists forever. Old low-importance entries get compressed monthly. |
| **Purpose** | Cross-show callbacks, learning what works |

```python
# Post-show consolidation extracts these:
DocumentInfo(
    id="show-20260607-maya-callback",
    text="[2026-06-07 | YC Hackathon Show] Callback that landed: connected 'dental hygienist from Tulsa' "
         "with 'Obama administration dating drought' → 'Has anyone in Tulsa flossed since the Obama "
         "administration?' Audience erupted. Formula: [profession] + [time reference] = unexpected link.",
    metadata={
        "type": "highlight",
        "date": "2026-06-07",
        "show": "yc-hackathon",
        "landed": "true",
        "importance": "5",
        "people": "maya",
    }
)
```

### 3. Semantic Memory — "What I Know About Comedy"

| | |
|---|---|
| **Index** | `riff-tropes` (Moss cloud) |
| **Content** | Comedy patterns, formulas, tropes, pop-culture references |
| **Lifecycle** | Persistent. Updated (upserted) as Riff learns what works. |
| **Purpose** | Give the LLM comedy structure to hang jokes on |

```python
# Pre-built from data/comedy_tropes.json:
DocumentInfo(
    id="trope-callback-formula",
    text="[Comedy Pattern: Callback] Bring back an earlier detail in a new context. "
         "Formula: [specific audience detail] + [current topic] = unexpected connection. "
         "Best when 2+ minutes have passed. The longer the gap, the bigger the laugh.",
    metadata={"type": "pattern", "category": "callback", "strength": "5"}
)
```

---

## Ambient Retrieval — The Core Mechanism

**No tool calls. No "should I search?" decision. Every turn, auto-query before LLM.**

```
User speaks → STT → [Moss queries fire in parallel, <10ms] → results injected as context → LLM decides to speak or stay silent → TTS

NOT:
User speaks → STT → LLM thinks → "I should search" → Moss query → results → LLM generates → TTS
                                   ^^^^^^^^^^^^^^^^
                                   500-1500ms wasted
```

### Implementation

```python
async def on_user_turn_completed(self, turn_ctx, new_message):
    query = new_message.text_content
    if not query or not query.strip():
        return

    # 1. Index into working memory (always)
    self._turn += 1
    await self.session.add_docs([
        DocumentInfo(id=f"turn-{self._turn}", text=f"[{speaker}] {query}")
    ])

    # 2. Gate check (cooldown, word count) — skip retrieval if we won't speak anyway
    if not self.trigger_gate.should_proceed(query):
        return

    # 3. Ambient retrieval — all three stores in parallel
    working_hits, episodic_hits, semantic_hits = await asyncio.gather(
        self.safe_query(self.session, query, top_k=3),
        self.safe_query(self.moss, "riff-episodes", query, top_k=2),
        self.safe_query(self.moss, "riff-tropes", query, top_k=2),
    )

    # 4. Budget-trim, deduplicate, format
    memory_block = self.format_memory(working_hits, episodic_hits, semantic_hits)

    # 5. Decision pipeline with memory context
    candidate = await self.maybe_riff(query, memory_block)
    if candidate:
        await self.agent_session.say(candidate)
```

---

## Token Budget System

### Config

```python
@dataclass
class MemoryConfig:
    max_context_tokens: int = 8000    # hard cap on injected memory
    working_ratio: float = 0.5        # same-show callbacks get most budget
    semantic_ratio: float = 0.3       # tropes/patterns
    episodic_ratio: float = 0.2       # past shows
    min_relevance_score: float = 0.4  # below this = don't inject (garbage filter)
    dedup_threshold: float = 0.90     # near-duplicate removal
    chars_per_token: int = 4          # rough token estimation
    moss_timeout_ms: int = 500        # skip if slow (don't block comedy for retrieval)
```

### Budget Allocation

```
8000 tokens total
├── 4000 tokens (50%) → working memory (same-show callbacks — most valuable)
├── 2400 tokens (30%) → semantic (comedy patterns/tropes)
└── 1600 tokens (20%) → episodic (past shows)

Each pool fills by relevance score until budget is exhausted.
Below min_relevance_score → dropped.
Near-duplicates → dropped.
```

### Budget Trim Logic

```python
def budget_trim(docs, max_tokens: int, min_score: float = 0.4, chars_per_token: int = 4) -> list[str]:
    """Take docs in relevance order until budget is spent. Drop low-confidence and dupes."""
    result = []
    used = 0
    seen_texts = []

    for doc in docs:
        if doc.score < min_score:
            break  # docs are sorted by score, so all remaining are worse
        if any(similarity(doc.text, s) > 0.90 for s in seen_texts):
            continue  # near-duplicate
        cost = len(doc.text) // chars_per_token
        if used + cost > max_tokens:
            break
        result.append(doc.text)
        seen_texts.append(doc.text)
        used += cost

    return result
```

---

## Confidence-Tiered Injection

The LLM sees memory with confidence signals — just like humans distinguish "I clearly remember" from "I vaguely recall":

```python
def format_memory(working_hits, episodic_hits, semantic_hits):
    parts = []

    # High confidence: same-show, recent
    confident_callbacks = [d for d in working_hits if d.score >= 0.6]
    if confident_callbacks:
        parts.append("CALLBACKS FROM THIS SHOW:\n" +
                     "\n".join(f"- {d.text}" for d in confident_callbacks))

    # Medium confidence: patterns and past shows
    patterns = [d for d in semantic_hits if d.score >= 0.5]
    if patterns:
        parts.append("COMEDY PATTERNS (relevant):\n" +
                     "\n".join(f"- {d.text}" for d in patterns))

    # Low confidence: vague matches
    vague = [d for d in working_hits + episodic_hits if 0.4 <= d.score < 0.6]
    if vague:
        parts.append("VAGUE RECALL (may be imprecise):\n" +
                     "\n".join(f"- {d.text}" for d in vague))

    return "\n\n".join(parts) if parts else ""
```

---

## Resilience

```python
async def safe_query(self, client_or_session, *args, **kwargs):
    """Never let retrieval failure block the comedy."""
    try:
        return await asyncio.wait_for(
            client_or_session.query(*args, **kwargs),
            timeout=self.config.moss_timeout_ms / 1000
        )
    except (asyncio.TimeoutError, Exception):
        return type('Empty', (), {'docs': []})()  # empty result, agent continues
```

| Failure | Behavior |
|---|---|
| Moss timeout (>500ms) | Skip memory, agent still riffs (just no callbacks) |
| All results below 0.4 | Inject nothing — clean context for LLM |
| Budget overflow | Lowest-scored chunks dropped (never truncated) |
| Session index missing | Create empty at show start |
| Echo (own TTS retrieved) | `echo_filter.py` blocks + last-line similarity filter |
| Duplicate memories | Dedup at 90% text similarity |

---

## Ingestion: Comedy Corpus (`data/comedy_tropes.json`)

### Structure

Each entry is **self-contained with baked-in context**:

```json
[
  {
    "id": "trope-callback-formula",
    "text": "[Comedy Pattern: Callback] Bring back an earlier detail in a new context. Formula: [specific audience detail] + [current topic] = unexpected connection. Best when 2+ minutes have passed.",
    "metadata": {"type": "pattern", "category": "callback", "strength": "5"}
  },
  {
    "id": "trope-rule-of-three",
    "text": "[Comedy Pattern: Rule of Three] Set up a pattern with two items, break it with the third. The subversion is the joke. Works for lists, examples, escalations.",
    "metadata": {"type": "pattern", "category": "structure", "strength": "5"}
  },
  {
    "id": "trope-heightening",
    "text": "[Comedy Pattern: Heightening] Take the audience's detail and exaggerate it one level further each time you reference it. 'Dental hygienist' → 'tooth cop' → 'the mouth police'. Escalate until absurd.",
    "metadata": {"type": "pattern", "category": "technique", "strength": "4"}
  },
  {
    "id": "ref-obama-time",
    "text": "[Pop Culture Reference: Obama Administration] 2009-2017. Use as a time anchor for 'it's been a long time' jokes. Works because everyone remembers it, nobody argues the dates.",
    "metadata": {"type": "reference", "category": "time_anchor", "strength": "3"}
  }
]
```

### Why This Structure

1. **Context header in brackets** — `[Comedy Pattern: X]` tells the LLM exactly what it's looking at
2. **Self-contained** — makes sense without surrounding context
3. **Formula + explanation** — LLM can apply the pattern to current moment
4. **Metadata enables filtering** — during tight budget, only pull `strength >= 4`

### Indexing

```bash
python build_index.py   # reads data/comedy_tropes.json → Moss cloud index 'riff-tropes'
```

---

## Post-Show Consolidation (Stretch Goal)

After each show, extract durable memories for future shows:

```python
async def consolidate_show(transcript: str, show_meta: dict):
    extraction = await llm.generate(f"""
    From this comedy show transcript, extract:
    
    1. HIGHLIGHTS: What callbacks/jokes landed (audience laughed)?
       For each: the setup, the punchline, why it worked, audience details involved.
    
    2. PATTERNS LEARNED: Any new comedy patterns discovered?
       Things that worked that aren't already in the tropes corpus.
    
    3. WHAT BOMBED: What fell flat? (so we don't repeat it)
    
    Show: {show_meta['title']} on {show_meta['date']}
    Transcript:
    {transcript}
    
    Return JSON with highlights[], patterns[], bombed[]
    """)

    # Highlights → episodic memory
    for h in extraction["highlights"]:
        await moss.add_docs("riff-episodes", [DocumentInfo(
            id=f"show-{show_meta['date']}-{slugify(h['setup'][:30])}",
            text=f"[{show_meta['date']} | {show_meta['title']}] {h['description']}",
            metadata={
                "type": "highlight",
                "date": show_meta["date"],
                "landed": "true",
                "importance": str(h["importance"]),
            }
        )])

    # New patterns → semantic memory (upsert = evolves over time)
    for p in extraction["patterns"]:
        await moss.add_docs("riff-tropes", [DocumentInfo(
            id=f"learned-{slugify(p['name'])}",
            text=f"[Learned Pattern: {p['name']}] {p['description']}",
            metadata={"type": "learned", "source_show": show_meta["date"]}
        )], MutationOptions(upsert=True))
```

**Over many shows, Riff builds its own comedy corpus — learning from experience, not just pre-loaded data.**

---

## Retrieval Evals

### Test Cases

```json
[
  {
    "scenario": "callback_opportunity",
    "setup": "Maya, dental hygienist, Tulsa introduced at turn 3",
    "query_at_turn_12": "So Maya, what do you actually do all day?",
    "must_retrieve": ["Maya", "dental hygienist", "Tulsa"],
    "must_not_retrieve": ["unrelated trope"]
  },
  {
    "scenario": "trope_match",
    "query": "I haven't been on a date since 2016",
    "must_retrieve": ["time anchor", "callback formula"],
    "must_not_retrieve": ["audience name extraction"]
  },
  {
    "scenario": "irrelevant_filler",
    "query": "yeah",
    "must_retrieve": [],
    "rationale": "below min_relevance_score, nothing injected"
  },
  {
    "scenario": "budget_overflow",
    "setup": "50 relevant turns in working memory",
    "query": "tell me about everyone",
    "assert": "total injected tokens <= max_context_tokens"
  }
]
```

### Metrics

| Metric | Target | Measured How |
|---|---|---|
| Recall@5 | >85% | expected chunks appear in top 5 |
| Precision@5 | >60% | % of returned chunks that are relevant |
| Latency P95 | <15ms | end-to-end Moss query time |
| Token efficiency | >70% | useful_tokens / total_injected |
| False injection | <10% | memory injected on irrelevant queries |
| Budget compliance | 100% | never exceeds `max_context_tokens` |

### Running

```bash
python -m evals.run_memory_eval          # retrieval accuracy
python -m evals.run_comedy_quality_eval  # does good retrieval → good comedy?
```

---

## Hybrid Approach: Ambient + Deep Recall Tool

**90% of turns**: ambient retrieval handles it (auto-search, inject context, done).

**10% of turns**: LLM needs more specific recall → uses `deep_recall` tool:

```python
@function_tool
async def deep_recall(self, context: RunContext, query: str) -> str:
    """Search show memory deeper when automatic recall wasn't enough.
    Use when you vaguely remember something but ambient context didn't include it."""
    results = await self.moss.query("show-{room}", query, QueryOptions(top_k=8))
    texts = budget_trim(results.docs, self.config.max_context_tokens, min_score=0.3)
    return "\n".join(f"- {t}" for t in texts) or "Nothing found."
```

This gives Riff a fallback without the latency penalty on 90% of turns.

---

## Chat Message API — Website/App Integration

Riff doesn't only listen to audio. It can also receive **text messages via API** — from a website chat, a Slack integration, a mobile app, or any frontend. These messages flow through the same memory + retrieval pipeline as audio turns.

### Input Sources

```
┌──────────────────┐     ┌──────────────────┐
│  Microphone      │     │  Website/App     │
│  (LiveKit STT)   │     │  (API calls)     │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         │  on_user_turn_completed│  POST /message
         │                        │
         ▼                        ▼
┌──────────────────────────────────────────────┐
│        UNIFIED TURN HANDLER                  │
│  1. Index into working memory                │
│  2. Ambient retrieval (parallel, <10ms)      │
│  3. Budget-trim + format context             │
│  4. Decision pipeline (speak or stay silent) │
└──────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
    TTS (audio)              JSON response (text)
```

Both paths hit the **exact same pipeline**. The only difference is output: audio turns produce TTS, chat messages return text.

### API Endpoints

#### `POST /message` — Single chat message

```json
// Request
{
  "text": "Hey Riff, what did Maya say earlier?",
  "sender": "audience-member-42",
  "respond": true,
  "metadata": {
    "source": "website-chat",
    "timestamp": "2026-06-07T22:15:00Z"
  }
}

// Response (if respond: true)
{
  "reply": "Maya? The dental hygienist from Tulsa? She said she's single. I'd swipe right but I don't have thumbs.",
  "memory_used": ["turn-3: Maya, dental hygienist, Tulsa", "turn-7: Maya is single"],
  "latency_ms": 1240,
  "decision": "spoke",
  "score": 8.2
}

// Response (if respond: false — just index, no reply)
{
  "indexed": true,
  "turn_id": "turn-47"
}
```

#### `POST /message/batch` — Bulk-inject chat history

For page loads or session reconnects — catch up Riff on what happened in chat:

```json
// Request
{
  "messages": [
    {"text": "Anyone here from out of state?", "sender": "host", "timestamp": "2026-06-07T22:10:00Z"},
    {"text": "I'm from Tulsa!", "sender": "maya", "timestamp": "2026-06-07T22:10:05Z"},
    {"text": "What do you do?", "sender": "host", "timestamp": "2026-06-07T22:10:08Z"},
    {"text": "Dental hygienist", "sender": "maya", "timestamp": "2026-06-07T22:10:12Z"}
  ]
}

// Response
{
  "indexed": 4,
  "turn_ids": ["turn-43", "turn-44", "turn-45", "turn-46"]
}
```

No response generated — just memory ingestion. Riff now has context for future turns.

### Implementation

```python
# src/riff/api.py

@app.post("/message")
async def receive_message(body: MessageRequest):
    agent: RiffAgent = app.state.agent

    # 1. Index into working memory (same as audio turns)
    turn_id = await agent.index_turn(
        text=body.text,
        speaker=body.sender,
        source="chat",
        metadata=body.metadata,
    )

    if not body.respond:
        return {"indexed": True, "turn_id": turn_id}

    # 2. Run full pipeline: ambient retrieval + decision
    result = await agent.process_chat_turn(body.text, body.sender)

    return {
        "reply": result.text if result else None,
        "memory_used": result.memory_context if result else [],
        "latency_ms": result.latency_ms if result else 0,
        "decision": "spoke" if result else "silent",
        "score": result.score if result else 0,
    }


@app.post("/message/batch")
async def receive_batch(body: BatchMessageRequest):
    agent: RiffAgent = app.state.agent
    turn_ids = []

    for msg in body.messages:
        turn_id = await agent.index_turn(
            text=msg.text,
            speaker=msg.sender,
            source="chat",
            metadata={"timestamp": msg.timestamp},
        )
        turn_ids.append(turn_id)

    return {"indexed": len(turn_ids), "turn_ids": turn_ids}
```

### Agent Unified Turn Handler

```python
class RiffAgent(Agent):
    async def index_turn(self, text: str, speaker: str, source: str = "audio", metadata: dict = None) -> str:
        """Index a turn from any source into working memory."""
        self._turn += 1
        turn_id = f"turn-{self._turn}"

        doc_text = f"[{speaker}] {text}"
        doc_metadata = {"speaker": speaker, "source": source, "turn": str(self._turn)}
        if metadata:
            doc_metadata.update(metadata)

        await self.session.add_docs([DocumentInfo(
            id=turn_id,
            text=doc_text,
            metadata=doc_metadata,
        )])

        return turn_id

    async def process_chat_turn(self, text: str, sender: str):
        """Same pipeline as audio, but returns text instead of TTS."""
        # Gate check
        if not self.trigger_gate.should_proceed(text):
            return None

        # Ambient retrieval
        working, episodic, semantic = await asyncio.gather(
            self.safe_query(self.session, text, top_k=3),
            self.safe_query(self.moss, "riff-episodes", text, top_k=2),
            self.safe_query(self.moss, "riff-tropes", text, top_k=2),
        )

        memory_block = self.format_memory(working, episodic, semantic)

        # Generate (same LLM call, just don't TTS it)
        candidate = await self.generate_candidate(text, memory_block)
        return candidate  # text response, no audio

    # Audio path (existing) calls same internals:
    async def on_user_turn_completed(self, turn_ctx, new_message):
        await self.index_turn(new_message.text_content, speaker="audience", source="audio")
        # ... rest of decision pipeline, but outputs to TTS
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Chat and audio share the same working memory index | A chat message "I'm from Tulsa" should be retrievable during audio callbacks and vice versa |
| `respond: false` mode exists | Sometimes the website just wants Riff to *know* something (e.g., user profile info) without expecting a reply |
| Batch endpoint doesn't trigger responses | Bulk history catch-up shouldn't fire 20 replies — just index silently |
| Response includes `memory_used` | Website can show "Riff remembered: ..." UI for transparency |
| Same decision pipeline (cooldown, quality gate) | Chat Riff shouldn't be more aggressive than audio Riff |
| WebSocket `/events` broadcasts `message_received` | Website gets real-time updates when audio turns happen too |

### Website Integration Example

```javascript
// Website chat widget
const RIFF_API = "http://localhost:8765";

// User sends a chat message
async function sendMessage(text) {
  const res = await fetch(`${RIFF_API}/message`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ text, sender: "user", respond: true }),
  });
  const data = await res.json();

  if (data.decision === "spoke") {
    appendToChat("riff", data.reply);
  }
  // else: Riff chose to stay silent (same behavior as audio)
}

// On page load, catch up on audio transcript
async function syncTranscript() {
  const res = await fetch(`${RIFF_API}/transcript?last=50`);
  const turns = await res.json();
  // Display in chat UI so website user sees what's been said on mic
}

// Real-time updates via WebSocket
const ws = new WebSocket("ws://localhost:8765/events");
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === "chime_emitted") {
    appendToChat("riff", data.text);  // Riff spoke (from audio trigger), show in chat too
  }
  if (data.type === "turn_indexed" && data.source === "audio") {
    appendToChat(data.speaker, data.text);  // Audio turn, mirror in chat
  }
};
```

---

## Speed Stack (Optimized for Comedy Timing)

| Component | Choice | Why |
|---|---|---|
| STT | Groq Whisper (chunked on RMS silence) | Fastest batch transcription, pseudo-streaming via 200ms silence detection |
| Retrieval | Moss ambient (parallel, all 3 indexes) | <10ms, no network hop, no LLM decision needed |
| LLM | Groq Llama 3.3 70B OR Minimax M3 | Fast token generation for one-liner candidates |
| TTS | Cartesia Sonic | Low latency, already personality-tuned |
| Total target | <1500ms user-speech-end → Riff-audio-starts | Comedy timing window |

### Latency Budget

```
STT finish:         ~300ms (Groq Whisper chunk)
Moss retrieval:      ~10ms (ambient, parallel)
LLM generation:    ~800ms (first tokens, streaming)
TTS first audio:   ~150ms (Cartesia streaming)
─────────────────────────────
Total:            ~1260ms (within comedy timing window)
```
