# PRD — "Riff" · The Crowd-Work Co-Host

**Hackathon:** YC × Moss Conversational AI Hackathon (build June 6–7, 2026)
**Stack (required):** Moss (real-time semantic retrieval, <10ms) · Minimax (LLM) · LiveKit (voice infra)

---

## 1. The Pitch

A voice AI that sits on a stool next to a standup comedian doing **crowd work** — listens continuously to the comic and the audience volunteers, and **decides on its own when to chime in.** It's not a chatbot. It's a co-host with its own mic. Mostly silent. Lands one-liners. Pulls callbacks across the set. Like Norm Macdonald sitting on Conan's couch — except Conan is a working comedian and Norm has read the entire internet.

> "It's not a voice assistant with a microphone. It's a co-host with a kill-switch on its own mouth."

## 2. Why This Wins

- **Demo is unforgettable.** A live comedian doing crowd work on stage with an AI co-host that isn't on a leash. Judges will share the clip.
- **Crowd work is the right form for an LLM.** Crowd work is one-liner observational comedy built around real-time facts about audience members — exactly the shape an LLM does well. We're not asking it to sustain a character or build a narrative scene. We're asking it to land single-line jabs grounded in retrieved context.
- **Moss's <10ms is *load-bearing*.** Crowd-work timing dies past ~300ms. The callback retrieval ("she said her ex was a chiropractor 12 minutes ago") *must* happen at conversational latency or the joke is dead on arrival. This is the only category where slow retrieval visibly breaks the product.
- **Inverted agent architecture.** Every other team will build response-driven voice agents (turn-based STT→LLM→TTS). We invert it: continuous listening + autonomous speak-or-stay-silent decision. Sponsors will see a novel agent shape, not a chat clone.
- **Sponsor-friendly Twitter moment.** Crowd-work clips already dominate TikTok and Reels. "An AI co-host roasted me from row 3" is a screenshot Moss/Minimax/LiveKit will all repost.

---

## 3. The Core Mechanic — "Speak When It Wants"

Standard voice agents follow: **user-turn → LLM-reply → TTS.** We can't use that loop. The agent must:

1. Listen to *everyone* in the room continuously (host + audience + scene partners).
2. After every turn, evaluate: *is this a moment worth interjecting?*
3. If yes, generate a candidate line, **self-rate** it for funniness, and only speak if it clears a quality bar.
4. Enforce a cooldown so it never hogs the stage.

### The Decision Loop

```
   ┌──────────────────────────────────────────┐
   │  TRANSCRIPT STREAM (Deepgram, always on) │
   └────────────────┬─────────────────────────┘
                    │  every finalized turn
                    ▼
   ┌──────────────────────────────────────────┐
   │ 1. INDEX-IT  →  moss_session.add_docs()  │  every line goes into Moss
   └────────────────┬─────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────┐
   │ 2. TRIGGER GATE                          │
   │   • cooldown elapsed (≥8s since last)?   │
   │   • turn longer than 3 words?            │
   │   • not currently mid-someone-speaking?  │
   └────────────────┬─────────────────────────┘
                    ▼ (yes)
   ┌──────────────────────────────────────────┐
   │ 3. OPPORTUNITY RETRIEVAL  (Moss × 2)     │
   │   • session.query(last_turn) → callbacks │
   │   • cloud_index.query(topic) → tropes    │
   │   ↑ this is where <10ms matters          │
   └────────────────┬─────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────┐
   │ 4. JOKE GENERATION  (Minimax)            │
   │   prompt: last 60s + retrieved hits      │
   │   output: 3 candidate lines + self-score │
   └────────────────┬─────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────┐
   │ 5. QUALITY GATE  (score ≥ 7 to ship)     │
   │   • discard if mid → silence is funnier  │
   └────────────────┬─────────────────────────┘
                    ▼
   ┌──────────────────────────────────────────┐
   │ 6. SPEAK  (LiveKit TTS, Cartesia voice)  │
   │   set last-speech-timestamp, cooldown    │
   └──────────────────────────────────────────┘
```

### Why each gate exists
- **Cooldown (8s)** — humans don't quip every line. Without this, the agent ruins every scene.
- **Word-count floor** — skip "yeah," "uh-huh," filler.
- **Quality gate (self-rate ≥7)** — Minimax rates its own candidates. Comedy that scores 5 is worse than silence. Restraint is the entire product.
- **Cloud index hits** — supplies tropes the LLM can hang on (callback templates, common improv patterns, comedy beats).

---

## 4. Architecture

### 4.1 Components

| Component | Tech | Role |
|---|---|---|
| Audio in/out | LiveKit room | Single mic in, Riff TTS out. Host repeats audience replies back through the same mic — Riff sees one unified transcript stream. |
| STT | Deepgram nova-3 (LiveKit `inference.STT`) | Continuous transcript stream, partial + final. |
| Decision LLM | Minimax M3 | "Should I speak?" classifier + joke generator + self-rater (one combined call to save latency). |
| Comedy corpus | Moss cloud index `riff-tropes` | Improv setups, callback templates, comedy beats, pop-culture refs. Pre-built. |
| Show memory | Moss live session `show-{room}` | Every turn indexed live; powers callbacks. |
| TTS | Cartesia sonic via LiveKit `inference.TTS`, voice `v_mVfV2aqK755e` | Reuse the voice from the workshop starter — already wired, already personality-tuned. |
| VAD | Silero | Don't speak over a human. |

### 4.2 Why Each Sponsor Is Essential

- **Moss is essential** because callbacks are the entire product. The agent must retrieve "what was said 14 minutes ago about the audience member from Tulsa" before the moment passes. **Vector DBs with 100ms latency cannot do this** — the joke arrives after the laugh window. Moss's two-tier setup (cloud index for tropes + live session for show memory) is exactly what improv needs. Removing Moss = the agent only references the last 60 seconds = no callbacks = it's not improv anymore.
- **Minimax is essential** because the LLM has to do three creative tasks per gate: opportunity-classify, generate 3 candidates, self-rate. A weaker model produces dad jokes. A frontier model with low-latency output is non-negotiable.
- **LiveKit is essential** because we need multi-participant audio (host + audience + AI in one room), continuous full-duplex listening, and TTS playback in the same channel. We're not "calling" the agent — we're sharing a stage with it. LiveKit's room model is the only piece that gives us that.

### 4.3 Inverting the LiveKit Default Loop

The starter's `AgentSession` auto-replies on every user turn via the default agent step. We override:

```python
class RiffAgent(Agent):
    async def on_user_turn_completed(self, turn_ctx, new_message):
        # 1. Always index into Moss session (memory grows)
        await self.session_idx.add_docs([DocumentInfo(...)])
        # 2. Run our decision pipeline — DO NOT call super().generate_reply
        candidate = await self.maybe_riff(new_message)
        if candidate:
            await self.agent_session.say(candidate)
        # else: stay silent. This is the inversion.
```

Default behavior: reply to every turn. Our behavior: reply only when `maybe_riff` returns non-None. This is the entire architectural novelty.

---

## 5. The 3-Minute Demo

**Setup:** laptop on a podium, one mic plugged in, speakers facing the audience. The host (me) does crowd work; when an audience member answers, the host **repeats the answer into the mic** before responding — exactly like a podcast host paraphrases caller questions, or like a comedian narrating a venue without a working audience-mic. Riff hears one continuous transcript stream and chimes in through the speakers.

| Time | Beat |
|---|---|
| **0:00** | Host: "Tonight I've got a co-host. His name is Riff. He's an AI. He gets to talk whenever he wants. Riff, you good?" |
| **0:05** | Riff: "Yeah. I'll mostly stay out of your way." (one line, then silence — sets the rules) |
| **0:20** | Host opens with crowd work: "Hey, who are we, where are we from, what do we do?" |
| **0:30** | Audience volunteer answers; host repeats into the mic: "Maya — dental hygienist — from Tulsa. Got it." Riff stays silent (recognizes setup, not punchline). |
| **0:50** | Host roasts Tulsa for 30 seconds. Riff stays silent. |
| **1:20** | Host pivots to dating: "Maya — and she just told me she's single, just out of a long relationship." Riff stays silent. |
| **1:40** | Host: "I'm telling you, I have not had a good first date since the Obama administration." → **Riff (chimes in): "That's not a dry spell, that's a presidency."** (one-liner, lands) |
| **1:55** | Audience laughs. Host laughs, points at Riff: "Okay, okay, fine." Beat resumes. Riff goes silent again. |
| **2:15** | Host circles back to Maya: "Maya, what does a dental hygienist actually do all day?" |
| **2:30** | **Riff (callback, unprompted): "Maya, has anyone in Tulsa flossed since the Obama administration? Asking for a friend."** (callbacks Tulsa + dental hygienist + Obama-administration setup — this is the Moss moment) |
| **2:45** | Audience erupts. Judges write down "callback worked across 2 minutes." |
| **2:55** | Host: "Riff, we should have you on every week." Riff: "I'd rather not. Nights are when I think." (closes the bit) |

The demo proves three things in 3 minutes: **(a) it stays silent,** **(b) it lands a joke when it speaks,** **(c) it pulls callbacks across long-range memory** (the Moss moment).

---

## 6. 20-Hour Build Plan

### Hour 0–2 · Wire it up
- Reuse `moss/moss-workshop/starter/.env` via `load_dotenv()` — all keys already set.
- Reuse Cartesia voice `v_mVfV2aqK755e` and STT `deepgram/nova-3`.
- Verify `python voice_agent.py console` runs end-to-end before any modification.
- Fork starter → `riff/` directory.

### Hour 2–5 · Modular skeleton + inverted loop
- Lay out `src/riff/` modules (see §11) as empty stubs with type-checked signatures.
- Implement `agent.py` (subclass `Agent`, disable auto-reply, index turns into Moss session).
- Implement `decision.py` trigger gate (cooldown, word-count) — instant, no LLM.
- Sanity check: agent stays silent for a 5-minute conversation.

### Hour 5–9 · Decision pipeline + Moss-similarity gate
- `memory.py` (Moss adapter) + `llm_client.py` (Minimax streaming wrapper, `max_tokens=50`, `thinking: disabled`).
- Decision flow: trigger gate → Moss retrieval → if top-similarity ≥ 0.75 → `agent_session.generate_reply(...)` with persona prompt + callbacks. LiveKit auto-streams LLM → TTS.
- Pre-warm Minimax with a dummy call at boot.

### Hour 9–11 · Trigger API (FastAPI on localhost:8765)
- `POST /trigger` → forced chime; bypasses cooldown + similarity gate; same generate_reply flow.
- `POST /mute {seconds}` / `POST /unmute` / `GET /status` / `GET /transcript` / `WS /events`.
- This unblocks the Electron renderer to start consuming live state.

### Hour 11–14 · Electron app
- `electron/main.js` spawns the Python agent as subprocess, manages lifecycle.
- Renderer: state pill (Idle / Listening / Thinking / Speaking), live transcript scroll, three buttons (Force Chime In / Mute 30s / Switch Persona), latency strip.
- All wiring via `localhost:8765` (fetch + WebSocket /events).

### Hour 14–16 · Memory corpus + echo defense
- `data/comedy_tropes.json` (~80 entries: callback templates, setup→punchline patterns, current-events refs). `build_index.py` → cloud index `riff-tropes`.
- `echo_filter.py`: mute STT during TTS playback, 300ms tail, last-line similarity filter (>70%).

### Hour 16–18 · Unit tests
- `tests/test_persona.py`, `test_memory.py`, `test_decision.py`, `test_echo_filter.py`, `test_telemetry.py`, `test_trigger.py`.
- Pure unit (no live keys). Integration tests marked `@pytest.mark.integration` — not run in default suite.

### Hour 18–20 · Demo polish + dry runs
- Tune cooldown + similarity threshold via mock dialogue.
- Pre-warmed canned-bit fallback if pipeline > 1.5s.
- Three full dry-runs of the 3-min demo. Time silences. Time laughs.

### MVP cut-line (if we run out of time)
- ✅ Must have: inverted loop, Moss similarity gate, generate_reply streaming, comedy_tropes index, echo defense, trigger API, Electron app showing state + force-chime button.
- ⚠️ Nice: full Electron polish, latency telemetry display, panic-button hotkey, self-rating quality gate.
- ❌ Cut first: speculative pre-generation, Pitch persona, persona-switch UI, audience-name extraction.

---

## 7. The Risk

**The single biggest risk:** the agent speaks at the wrong moment on stage and kills a beat. Demo dies in real time.

**Mitigation, in order of cost:**
1. **Stage panic button.** Host wears a discreet button (LiveKit data channel) that mutes Riff for 30 seconds. Single biggest safety net.
2. **Aggressive cooldown** (12s) for the demo run — sacrifices density for reliability.
3. **High quality threshold** (8 instead of 7) — speaks less often but lands more.
4. **Pre-warmed canned-bit fallback.** If LLM call exceeds 1.5s, we don't speak that turn at all (silent failure > bad joke).

**Secondary risks:**
- STT lag → mitigated by Deepgram nova-3 (already <300ms).
- **Echo loop** (laptop speaker plays Riff's TTS → laptop mic re-transcribes Riff's own voice → Riff riffs on its own line). Real risk because we're using one device for both input and output. Three layers of defense:
  1. **Mute STT during TTS playback.** When `agent_session.say(...)` starts, suspend the input pipeline; resume on TTS complete.
  2. **Suppression window** after TTS ends (300ms) — kills any tail-end audio that bled in.
  3. **Last-line filter** — if a freshly transcribed turn matches Riff's last spoken line by >70% similarity, discard it.
- Moss session size → fine, one show is at most ~10k tokens of transcript.

---

## 8. Stretch Goals (if everything ships early)
- Audience-name auto-extraction → "the dental hygienist Maya from row 3" callbacks.
- Topic-tracker → recognizes when the host pivots and avoids stale callbacks.
- Persona switch hotword → "Riff, be Larry David" reloads a different style index.
- Live captions screen for the audience showing Riff's "thinking but staying silent" moments — turns the inversion into part of the show.

---

## 9. Locked Decisions

- ✅ **Framing:** crowd-work co-host (not improv troupe, not generic stage partner).
- ✅ **Keys:** reuse `moss/moss-workshop/starter/.env` via `load_dotenv()`. No new keys needed.
- ✅ **Voice:** reuse Cartesia `v_mVfV2aqK755e` from the starter — already wired and personality-tuned.
- ✅ **STT:** Deepgram `nova-3` via LiveKit `inference.STT` (same as starter).
- ✅ **LLM:** Minimax M3 via the OpenAI-compatible base URL in `.env`.
- ✅ **Code quality:** All code fully abstracted (interfaces + DI). Every function unit-tested. Tests must pass before moving to the next feature. No untested code ships.

## 10. Locked Setup
- ✅ **Demo host:** you. Solo. You'll rehearse the bit and tee up callback fodder.
- ✅ **Physical setup:** laptop + one mic + laptop speakers. Host repeats audience answers into the mic so Riff sees one unified transcript stream.
- ✅ **Echo handling:** input pipeline suspended during Riff's TTS playback, plus 300ms tail and last-line dedupe filter.
- ✅ **Ships as:** Electron desktop app. Python agent runs as a managed subprocess; UI talks to it via localhost FastAPI.

---

## 11. Code Architecture (SOLID, modular)

```
mossHackathon/
├── PRD.md
├── voice_agent.py              ← thin entry point; loads persona, starts agent + API server
├── pyproject.toml
├── personas/
│   └── riff.yaml               ← name, voice_id, system_prompt, cooldown, threshold, indexes
├── data/
│   └── comedy_tropes.json
├── src/riff/
│   ├── __init__.py
│   ├── persona.py              ← PersonaPack dataclass + YAML loader (factory)
│   ├── memory.py               ← MossMemory adapter (Moss cloud + session). Defines abstract Memory interface for testability
│   ├── decision.py             ← DecisionPipeline. Gates: cooldown → word-count → Moss similarity → LLM. Strategy injected via PersonaPack
│   ├── llm_client.py           ← MinimaxClient. Streaming wrapper with pre-warm + max_tokens cap
│   ├── echo_filter.py          ← EchoFilter. Mute-during-TTS state + last-line similarity dedupe
│   ├── telemetry.py            ← Telemetry. Observer pattern; collects timing events at each pipeline stage
│   ├── trigger.py              ← ForceTrigger. Command pattern; encapsulates a forced-speak request
│   ├── api.py                  ← FastAPI app. Handlers delegate to Trigger, EchoFilter (mute), Telemetry (status)
│   └── agent.py                ← RiffAgent (subclass LiveKit Agent). Wires Memory + Decision + EchoFilter + Telemetry via DI
├── tests/
│   ├── conftest.py             ← shared fixtures (FakePersona, FakeMemory, FakeLLM)
│   ├── test_persona.py
│   ├── test_memory.py
│   ├── test_decision.py
│   ├── test_echo_filter.py
│   ├── test_telemetry.py
│   └── test_trigger.py
└── electron/
    ├── package.json
    ├── main.js                 ← spawns Python agent subprocess, manages lifecycle
    ├── preload.js              ← contextBridge IPC surface
    └── renderer/
        ├── index.html
        ├── app.js              ← talks to localhost:8765
        └── styles.css
```

### Design patterns (used purposefully, not for show)

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `PersonaPack` configures `DecisionPipeline` | Open/Closed: add personas without touching engine code |
| **Adapter** | `MossMemory` wraps `MossClient` against an internal `Memory` interface | Lets tests swap in `FakeMemory` |
| **Observer** | `Telemetry` subscribes to pipeline stage events | Latency tracking without coupling to logic |
| **Command** | `ForceTrigger` encapsulates a forced-speak request | API and Electron both invoke the same command |
| **Dependency Injection** | `RiffAgent(memory, llm, decision, echo_filter, telemetry)` | Tests inject fakes; production wires real services |
| **Factory** | `PersonaPack.from_yaml(path)` | Centralized validation + defaults |

### SOLID checklist
- **S** — each module owns one concern (memory ≠ decision ≠ echo ≠ telemetry).
- **O** — adding a persona = adding a YAML file. No engine code changes.
- **L** — `FakeMemory` and `MossMemory` are interchangeable behind the `Memory` interface.
- **I** — `Memory` exposes `add(doc)` / `query(text)`; nothing else. Don't bloat.
- **D** — `RiffAgent` depends on the abstract `Memory` / `LLMClient` interfaces, not on `MossClient` / `OpenAI` directly.

---

## 12. Trigger API (`src/riff/api.py`)

FastAPI on `localhost:8765`. Single source of truth for runtime state.

| Method | Path | Body | Purpose |
|---|---|---|---|
| `POST` | `/message` | `{"text": "...", "sender": "user", "respond": true}` | Inject a chat message. Indexed into memory, runs ambient retrieval + decision pipeline. If `respond: true`, returns Riff's text reply (no TTS). If `respond: false`, just indexes silently. |
| `POST` | `/message/batch` | `{"messages": [{"text": "...", "sender": "...", "timestamp": "..."}]}` | Bulk-inject chat history (e.g., page load). All indexed into working memory. No response generated. |
| `POST` | `/trigger` | `{"hint": "..."}` (optional) | Force a chime now. Bypasses cooldown + similarity gate. Same `generate_reply` flow → LiveKit streams LLM → TTS. Returns `latency_ms`. |
| `POST` | `/mute` | `{"seconds": 30}` | Suspend speech for N seconds (panic button). |
| `POST` | `/unmute` | — | Resume. |
| `GET`  | `/status` | — | `{persona, last_spoke_at, queue_depth, mute_until}` |
| `GET`  | `/transcript` | `?last=20` | Last N turns from rolling buffer. |
| `WS`   | `/events` | — | Live state stream: `state_changed`, `turn_indexed`, `chime_decision`, `chime_emitted`, `message_received`. Electron/website subscribes. |

**Trigger flow (forced chime):**
1. Capture last-60s transcript window from `RiffAgent.transcript_buffer`.
2. Fetch top callback hits from `MossMemory.query(...)`.
3. Build forcing prompt: persona system + transcript + callbacks + `"Drop a one-liner now."`.
4. `agent_session.generate_reply(user_input=prompt)` — LiveKit pipelines LLM tokens → TTS audio (~150ms to first audio).
5. Telemetry records `forced_chime_latency_ms`. Return.

The trigger API path is the same flow as autonomous chiming, just with the gates skipped — keeps the codepath single, no duplicate logic.

---

## 13. Electron App

**Goal:** desktop software shell. Looks like a real product to judges. Lets the host visually monitor Riff and force chimes manually.

### Window layout (single window, ~600×400)

```
┌──────────────────────────────────────────────────────────────┐
│  RIFF                          [● LISTENING]   p50: 940ms   │ ← state pill, latency strip
├──────────────────────────────────────────────────────────────┤
│  Transcript                                                  │
│  > host: ...so I haven't had a date since the Obama admin    │
│  > riff: That's not a dry spell, that's a presidency.        │ ← last 10 turns, scrolling
│  > host: Maya, what does a dental hygienist do all day?      │
│  > riff: Has anyone in Tulsa flossed since the Obama admin?  │
├──────────────────────────────────────────────────────────────┤
│   [ ⚡ Force Chime In ]   [ 🔇 Mute 30s ]   [ Persona: Riff ▼ ]│
└──────────────────────────────────────────────────────────────┘
```

### Architecture

```
┌──────────────────┐  spawns  ┌────────────────────────┐
│  Electron main   │ ───────▶ │  Python voice_agent.py │
│  (main.js)       │          │   (subprocess)         │
└──────────────────┘          │   FastAPI :8765        │
        │                     │   LiveKit room         │
        │ contextBridge       │   Moss + Minimax       │
        ▼                     └────────────────────────┘
┌──────────────────┐  fetch + WS  ▲
│  Renderer        │ ─────────────┘
│  (HTML + JS)     │
└──────────────────┘
```

- `main.js` manages subprocess lifecycle (spawn, log capture, graceful shutdown on quit).
- `preload.js` exposes a tiny IPC surface; renderer talks to FastAPI directly via fetch + WebSocket.
- No native deps. No bundler beyond what `electron-builder` provides.

### MVP cut-line for Electron
- ✅ State pill + transcript scroll + Force Chime button + Mute button.
- ⚠️ Latency strip + persona dropdown.
- ❌ Cut first: animations, persona-switch UI, settings pane.

---

## 14. Test Strategy

### Two layers
1. **Unit (default `pytest`)** — pure logic, no network, mocks for Moss/LLM/LiveKit. Runs in ~1s. CI-safe.
2. **Integration (`pytest -m integration`)** — hits real Moss + Minimax. Requires `.env`. Skipped by default.

### Coverage targets

| Module | Tested | How |
|---|---|---|
| `persona.py` | YAML load, schema validation, defaults | unit |
| `memory.py` | adapter contract via `FakeMemory` | unit; integration with real Moss |
| `decision.py` | each gate's behavior in isolation: cooldown elapsed/not, word-count under/over, similarity above/below | unit, mocked memory + LLM |
| `echo_filter.py` | mute-during-TTS state machine; similarity dedupe at boundary | unit |
| `telemetry.py` | observer fan-out; p50/p95 calc on synthetic samples | unit |
| `trigger.py` | force-trigger bypasses gates; calls generate_reply with correct prompt | unit, mocked agent_session |
| `agent.py` | DI wiring; on_turn → memory.add called; auto-reply suppressed | unit, mocked everything |
| `api.py` | endpoint shapes; auth absent ok for localhost | unit via FastAPI TestClient |

### Fixtures (`conftest.py`)
- `fake_persona` — minimal valid PersonaPack
- `fake_memory` — in-memory dict-backed Memory implementation
- `fake_llm` — returns scripted responses
- `fake_telemetry` — captures events for assertions

No tests for LiveKit AgentSession — that's framework code, smoke-tested by running `voice_agent.py console`.

---

## 15. Memory Architecture — "Living Memory" System

The show memory isn't just a transcript dump. It's a **layered, human-like memory system** designed so Riff can recall callbacks naturally, understand context from past shows, and never inject irrelevant noise.

### 15.1 Design Principles

1. **Every retrieved chunk must be independently interpretable** — if Riff pulls a memory, it must make sense without surrounding context.
2. **Ambient retrieval, not tool-driven** — Moss queries fire automatically on every turn *before* the LLM, saving one full LLM round-trip (~500-1500ms). No "decide to search" latency.
3. **Token-budgeted injection** — never exceed configurable max (default 8000 tokens). Relevance-ranked, budget-trimmed, deduplicated.
4. **Graceful degradation** — if Moss times out (>500ms), the agent continues without memory. Silence is better than lag.

### 15.2 Three Memory Layers

```
┌─────────────────────────────────────────────────┐
│  WORKING MEMORY  (Moss session, ephemeral)      │
│  Raw turns from current show. Lives/dies with   │
│  the session. Powers same-show callbacks.       │
├─────────────────────────────────────────────────┤
│  EPISODIC MEMORY  (Moss cloud index, persistent)│
│  "What happened" — past shows, past audience    │
│  interactions, great callbacks that landed.     │
│  Compressed over time.                          │
├─────────────────────────────────────────────────┤
│  SEMANTIC MEMORY  (Moss cloud index, persistent)│
│  "What I know" — comedy patterns, tropes,       │
│  what topics work, audience demographics.       │
│  Evolves and gets updated.                      │
└─────────────────────────────────────────────────┘
```

| Layer | Index Name | Content | Lifecycle |
|---|---|---|---|
| Working | `show-{room}` | Every turn, verbatim | Dies after show |
| Episodic | `riff-episodes` | Processed highlights from past shows | Compressed over time |
| Semantic | `riff-tropes` | Comedy patterns, tropes, what works | Updated, never timestamped |

### 15.3 Ambient Retrieval (Pre-LLM, Automatic)

Every turn, *before* the LLM sees anything:

```python
async def on_user_turn_completed(self, turn_ctx, new_message):
    query = new_message.text_content
    
    # 1. Always index into working memory
    await self.session.add_docs([DocumentInfo(id=f"turn-{n}", text=query)])
    
    # 2. Ambient retrieval — parallel, <10ms total
    callbacks, tropes = await asyncio.gather(
        safe_query(self.session, query, top_k=3),      # same-show callbacks
        safe_query(self.moss, "riff-tropes", query, top_k=2),  # comedy patterns
    )
    
    # 3. Budget-trim and inject as context
    memory_block = budget_trim_and_format(callbacks, tropes, max_tokens=self.config.max_context_tokens)
    if memory_block:
        turn_ctx.add_message(ChatMessage.create(role="system", text=memory_block))
    
    # 4. Run decision pipeline (NOT super().generate_reply)
    candidate = await self.maybe_riff(new_message, memory_block)
    if candidate:
        await self.agent_session.say(candidate)
```

### 15.4 Token Budget System

```python
@dataclass
class MemoryConfig:
    max_context_tokens: int = 8000    # hard cap, configurable
    working_ratio: float = 0.5        # 50% for same-show callbacks
    semantic_ratio: float = 0.3       # 30% for tropes/patterns
    episodic_ratio: float = 0.2       # 20% for past show memory
    min_relevance_score: float = 0.4  # below this = don't inject
    chars_per_token: int = 4          # rough estimation
    moss_timeout_ms: float = 500      # skip if Moss is slow
```

**How it works:**
- Each pool fills in relevance order until budget runs out.
- Chunks below `min_relevance_score` are dropped entirely.
- Near-duplicate chunks (>90% overlap) are deduplicated.
- If total memory fits under budget, all relevant chunks are included.
- If over budget, lowest-scored chunks are dropped (not truncated mid-sentence).

### 15.5 Confidence-Tiered Context

Retrieved memories are presented to the LLM with confidence signals:

```
STRONG CALLBACKS (from this show):
- Maya from Tulsa, dental hygienist, single, just out of a long relationship
- Host hasn't had a date since the Obama administration

COMEDY PATTERNS (relevant):
- Callback formula: [original detail] + [new context] = unexpected connection
- Rule of three: setup, reinforce, subvert

VAGUE RECALL (may be imprecise):
- Someone mentioned chiropractors earlier (not sure who)
```

This mirrors how humans say "I think someone said..." vs "I clearly remember..."

### 15.6 Resilience

| Failure Mode | Handling |
|---|---|
| Moss timeout (>500ms) | Skip memory, agent still works (just no callbacks) |
| Irrelevant results | Relevance floor (0.4) filters garbage |
| Duplicate chunks | Dedup by text similarity before injection |
| Token overflow | Budget system hard-caps at configured max |
| Session index missing | Create empty on first access |
| Echo (own speech retrieved) | Last-line filter already in `echo_filter.py` |

### 15.7 Post-Show Consolidation (Stretch Goal)

After each show, an LLM pass extracts durable memories:

```python
async def consolidate_show(transcript, show_meta):
    extraction = await llm.generate(f"""
    From this comedy show transcript, extract:
    1. What callbacks landed (audience laughed)
    2. What topics worked / bombed
    3. Audience details worth remembering for next show
    4. New comedy patterns discovered
    
    Show: {show_meta}
    Transcript: {transcript}
    """)
    
    # Index into episodic (what happened) and semantic (what we learned)
    await moss.add_docs("riff-episodes", episodes)
    await moss.add_docs("riff-tropes", new_patterns, MutationOptions(upsert=True))
```

This means Riff gets *better* across shows — remembers what worked, learns audience patterns, builds a personal comedy corpus.

### 15.8 Ingestion Strategy for Comedy Corpus

The `riff-tropes` cloud index is pre-built from `data/comedy_tropes.json`. Each entry is structured as a **self-contained, context-rich chunk**:

```json
{
  "id": "trope-callback-formula",
  "text": "[Comedy Pattern: Callback] Bring back an earlier detail in a new context. Formula: take [specific thing audience member said] + connect to [current topic] = unexpected link. Best when 2+ minutes have passed. Example: 'Maya the dental hygienist' + 'Obama administration' = 'Has anyone in Tulsa flossed since the Obama administration?'",
  "metadata": {"type": "pattern", "category": "callback", "strength": "5"}
}
```

**Why this structure works:**
- Baked-in context header (`[Comedy Pattern: Callback]`) — LLM knows what it's looking at
- Self-contained — no need for "surrounding lines"
- Formula + example — LLM can apply the pattern to current context
- Metadata enables filtering (only high-strength patterns during tight budget)

---

## 16. Retrieval Eval Framework

### 16.1 What to Measure

| Metric | Target | What breaks if missed |
|---|---|---|
| Recall@5 | >85% | Riff misses obvious callback opportunities |
| Precision@5 | >60% | Riff's context is polluted with irrelevant noise |
| Retrieval latency P95 | <15ms | Breaks real-time comedy timing |
| Token efficiency | >70% useful | Wastes budget on junk, crowds out good callbacks |
| False injection rate | <10% | Riff references things that never happened |

### 16.2 Eval Dataset

```python
# evals/memory_eval.json
[
    {
        "scenario": "callback_opportunity",
        "show_transcript": "...(Maya, dental hygienist, Tulsa)...",
        "query": "So Maya, what do you actually do all day?",
        "expected_retrievals": ["turn with Maya intro", "dental hygienist detail"],
        "should_not_retrieve": ["unrelated trope about dating"]
    },
    {
        "scenario": "irrelevant_query",
        "query": "yeah",
        "expected_retrievals": [],
        "rationale": "filler words should not trigger retrieval injection"
    },
    {
        "scenario": "trope_match",
        "query": "I haven't been on a date since 2016",
        "expected_retrievals": ["callback formula", "time-based exaggeration pattern"],
        "should_not_retrieve": ["audience name extraction pattern"]
    }
]
```

### 16.3 Running Evals

```bash
# After changing chunking, alpha, top_k, or min_score:
python -m evals.run_memory_eval

# Output:
# ✓ [callback_opportunity] recall=100% precision=80% 3.2ms
# ✓ [irrelevant_query]     correctly returned nothing    2.1ms
# ✓ [trope_match]          recall=100% precision=100%   2.8ms
# ────────────────────────────────────────────────────────────
# Recall@5: 95%  Precision: 82%  P50: 2.9ms  P95: 4.1ms
```

### 16.4 End-to-End Comedy Quality Eval

Beyond retrieval accuracy, test whether retrieved context produces good comedy:

```python
async def eval_comedy_output(scenario, retrieved_context):
    """Does the LLM produce a funny, relevant line given this context?"""
    response = await llm.generate(
        persona_prompt + f"\nContext:\n{retrieved_context}\n\nDrop a one-liner."
    )
    
    # LLM-as-judge rates the output
    score = await judge_llm.generate(f"""
        Rate this comedy line 1-10 for:
        - Relevance to the conversation (uses the context?)
        - Timing (would it land in the moment?)
        - Originality (not a generic quip?)
        
        Context: {scenario}
        Line: {response}
    """)
    return score
```

### 16.5 When to Run Evals

- After changing `comedy_tropes.json` content or structure
- After changing `alpha` (hybrid search weight)
- After changing `top_k` or `min_relevance_score`
- After changing `max_context_tokens` budget
- After changing chunk sizes or context header format
- Before every demo dry-run
