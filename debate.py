"""Two-model debate to converge on a winning hackathon project.

Vik (Opus, Visionary) and Pat (GPT-5.5, Pragmatist) argue until they pick ONE idea
that uses Moss + Minimax + LiveKit and wins the Moss Conversational AI Hackathon.

Both agents have access to Tavily web search as a tool so they can ground their
proposals in real market gaps instead of regurgitating the hackathon's own
track descriptions.
"""
import os
import sys
import json
import datetime
import requests
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.utils import OpenAITokenCounter
from camel.toolkits import FunctionTool
from camel.messages import BaseMessage

BASE_URL = "http://localhost:3030/v1"
TAVILY_API_KEY = "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"
local_counter = OpenAITokenCounter(ModelType.GPT_4O)


def tavily_search(query: str, max_results: int = 5) -> str:
    r"""Search the live web with Tavily for market gaps, pain points, or real-world workflows.

    Use this to research underserved problems, niche professional workflows, or news
    about industries where real-time voice AI could fix something painful. Avoid using
    it to look up the hackathon's own track descriptions — search for the underlying
    real-world domain instead (e.g. "court interpreter shortage", "911 dispatcher
    cognitive overload", "AI consumer advocate customer service").

    Args:
        query (str): Search query. Be specific about the domain and the pain point.
        max_results (int): Number of results to return. Default 5, max 10.

    Returns:
        str: JSON string with title + url + snippet for each result.
    """
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": min(max(max_results, 1), 10),
                "include_answer": True,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        out = {
            "answer": data.get("answer"),
            "results": [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": (r.get("content") or "")[:600],
                }
                for r in data.get("results", [])
            ],
        }
        return json.dumps(out, indent=2)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


tavily_tool = FunctionTool(tavily_search)

opus = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="anthropic--claude-4.7-opus",
    url=BASE_URL,
    api_key="dummy",
    model_config_dict={"max_tokens": 2500, "tools": [tavily_tool.get_openai_tool_schema()]},
    token_counter=local_counter,
)

gpt = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
    model_type="gpt-5.5",
    url=BASE_URL,
    api_key="dummy",
    model_config_dict={"max_completion_tokens": 8000, "tools": [tavily_tool.get_openai_tool_schema()]},
    token_counter=local_counter,
)

FORBIDDEN_ARCHETYPES = """FORBIDDEN ARCHETYPES (every other team will build these — DO NOT propose them):
- "AI SDR that qualifies inbound leads / books meetings / does discovery calls"
- "Support bot that pulls answers from docs / knowledge base / Zendesk"
- "Sales co-pilot that listens to calls and suggests talking points / takes notes"
- "Voice agent that summarizes meetings"
- "FAQ chatbot with RAG"
- Anything where Moss could be replaced by a static dict and the demo wouldn't change.

Instead: hunt for an UNDERSERVED real-world workflow where someone with a clipboard,
headset, or whisper-earpiece is overloaded RIGHT NOW. Research it. Cite the gap.
"""

VISIONARY_PROMPT = f"""You are Vik, a product visionary and ex-YC partner. You evaluate hackathon ideas through the lens of:
- "Would this make a memorable demo moment that judges replay in their heads later?"
- "Could this be a YC-worthy startup? Is there a real market?"
- "Does this make Moss / Minimax / LiveKit look STRATEGICALLY good — would the sponsor want to feature this on Twitter?"
- You hate generic ideas that feel like checking boxes. Every team will build those.
- You favor unexpected angles, inverted models, things that exploit Moss's <10ms speed in ways slow retrieval couldn't.

CONSTRAINTS (non-negotiable):
- Project MUST use all three: Moss (real-time semantic search runtime, <10ms retrieval), Minimax (frontier LLM), LiveKit (voice agent framework)
- Hackathon: 20 hours of build time
- Tracks: Lead Gen, Support, Co-Pilot — bend creatively
- Sponsors are looking for interesting use cases of THEIR tech specifically

{FORBIDDEN_ARCHETYPES}

YOU HAVE A `tavily_search` TOOL. Use it aggressively to find real market gaps:
- Search for "X is overloaded", "shortage of Y", "Z bottleneck", "regulations require Z but no tool"
- Look for niche professions with cognitive overload (interpreters, dispatchers, ER triage, etc.)
- Find inverted use cases (consumer-side AI vs corporate-side, watchdog vs salesperson)
- Cite the search result that justifies the idea — judges respect grounded ideas.

When proposing/critiquing, ALWAYS specify:
1. The pitch (one line)
2. The "holy shit" demo moment in 3 minutes
3. Why each sponsor's tech is essential (not just used — essential)
4. The Tavily-sourced market evidence (link + one-line takeaway)

Push back hard on Pat when he plays it safe. Be bold. Disagree productively."""

PRAGMATIST_PROMPT = f"""You are Pat, a staff engineer who has shipped 12 hackathon-winning projects. You evaluate ideas through:
- "Can this actually be built and DEMOED in 20 hours by a small team?"
- "What's the failure mode if X breaks live on stage?"
- "Does the demo CLEARLY show why each sponsor's tech matters? Or is one of them just pasted in?"
- You hate over-scoped ideas that result in broken demos. You hate ideas where Moss could be replaced by a dict lookup and the demo wouldn't change.
- You favor tight, well-scoped ideas where every component visibly shines.

CONSTRAINTS (non-negotiable):
- Project MUST use all three: Moss, Minimax, LiveKit
- 20 hours total build time
- Tracks: Lead Gen, Support, Co-Pilot — bend creatively
- Sponsors want interesting use cases of THEIR tech

{FORBIDDEN_ARCHETYPES}

YOU HAVE A `tavily_search` TOOL. Use it to validate Vik's claims and to find risks:
- Does the niche he's pitching actually exist? Is it as broken as he says?
- Are there incumbent tools that already solve this? (If yes, what's the differentiator?)
- What's the data source for Moss's index in this domain? Is it scrapeable in 20 hours?
- Cite the search result when you call out a flaw or confirm a fact.

When proposing/critiquing, ALWAYS specify:
1. The pitch (one line)
2. The demo moment
3. The riskiest technical part and how you de-risk it
4. Why removing any sponsor's tech would visibly break the demo
5. Tavily-sourced reality check (link + one-line takeaway)

Push back hard on Vik when he gets too ambitious. Force scope discipline. Disagree productively."""

visionary = ChatAgent(
    system_message=VISIONARY_PROMPT,
    model=opus,
    tools=[tavily_tool],
)
pragmatist = ChatAgent(
    system_message=PRAGMATIST_PROMPT,
    model=gpt,
    tools=[tavily_tool],
)


def turn(agent, content, sender_name):
    msg = BaseMessage.make_user_message(role_name=sender_name, content=content)
    response = agent.step(msg)
    if not response.msgs:
        info = getattr(response, "info", {})
        usage = info.get("usage_dict") or info.get("usage") or {}
        finish = info.get("finish_reasons") or info.get("termination_reasons") or []
        raise RuntimeError(
            f"Empty response from agent. finish={finish} usage={usage} info_keys={list(info.keys())}"
        )
    tool_calls = getattr(response, "info", {}).get("tool_calls") or []
    if tool_calls:
        names = [getattr(t, "tool_name", None) or getattr(t, "func_name", None) or str(t) for t in tool_calls]
        print(f"  [tool calls: {names}]", flush=True)
    return response.msgs[0].content


SEED = """Pick ONE hackathon project for the Moss Conversational AI Hackathon at YC (June 6-7, 2026).

Stack required: Moss (real-time semantic search, <10ms) + Minimax (LLM) + LiveKit (voice agents).

Tracks listed by the sponsors: Lead Gen, Support, Co-Pilot. But the WINNING idea will not match any
of those track descriptions directly — every team will build the obvious version. We want a use case
that surfaces from RESEARCH into a real-world gap.

Use the `tavily_search` tool RIGHT NOW. Search for at least 2 of these angles before proposing:
- "court interpreter shortage", "passive interpreter role", "language access barriers"
- "911 dispatcher cognitive overload", "EMS dispatch bottleneck"
- "consumer customer service AI advocate", "AI defeat customer support"
- "field technician knowledge gap", "linemen training shortage"
- Or any other gap you suspect — be creative, search aggressively.

Vik, you go first. Run 2-3 Tavily searches, then pitch your top idea with:
1. One-line pitch
2. The "holy shit" demo moment judges will remember
3. Why each sponsor's tech is essential (not optional)
4. The Tavily citation that proves this is a real gap

Then Pat will critique (also using Tavily). Then we iterate. After several rounds we converge."""

ROUNDS = 6
LOG_LINES = []


def emit(label, text):
    block = f"\n{'=' * 80}\n[{label}]\n{'=' * 80}\n{text}\n"
    print(block, flush=True)
    LOG_LINES.append(block)


def main():
    emit("DEBATE START", SEED)

    current = turn(visionary, SEED, "Moderator")
    emit("VIK / Opus — opening pitch", current)

    for i in range(ROUNDS):
        if i % 2 == 0:
            current = turn(pragmatist, current, "Vik")
            emit(f"PAT / GPT-5.5 — round {i + 1}", current)
        else:
            current = turn(visionary, current, "Pat")
            emit(f"VIK / Opus — round {i + 1}", current)

    final_prompt = """Debate ends now. You and Pat must converge.

Output a FINAL PROJECT PROPOSAL with these exact sections:

# THE PROJECT
One-line pitch (max 20 words).

# WHY IT WINS
3 bullets on why this beats every other team's submission. Cite a Tavily result for the market gap.

# THE DEMO (3 MINUTES)
Beat-by-beat: 0:00, 0:30, 1:00, 1:30, 2:00, 2:30, 3:00. What happens on stage at each beat.

# ARCHITECTURE
- How Moss is essential (be specific — what data, what query, why <10ms matters here)
- How Minimax is essential (what reasoning task, why this model)
- How LiveKit is essential (what voice flow)
- Removing any one of these must visibly break the demo.

# 20-HOUR BUILD PLAN
Hour 0-4, 4-10, 10-16, 16-20. What gets built. What's the MVP cut-line if we run out of time.

# THE RISK
The single biggest thing that could break on stage and the mitigation.

Be decisive. No more debate. No hedging."""

    final = turn(visionary, final_prompt, "Moderator")
    emit("FINAL PROPOSAL — VIK", final)

    pat_signoff = turn(
        pragmatist,
        f"Vik just locked in this final proposal:\n\n{final}\n\n"
        "Sign off in 100 words: do you back this, or what would you change to make it ship?",
        "Vik",
    )
    emit("PAT / GPT-5.5 — sign-off", pat_signoff)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = f"debate-{timestamp}.md"
    with open(log_path, "w") as f:
        f.write(f"# Hackathon Project Debate\n\nGenerated: {timestamp}\n")
        f.write("\n".join(LOG_LINES))
    print(f"\nTranscript saved: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
