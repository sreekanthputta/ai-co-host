"""One-time index builder. Loads comedy tropes (gating KB) into Moss.

    python build_index.py
"""
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from moss import MossClient, DocumentInfo

from src.riff.persona import PersonaPack

load_dotenv(override=True)

ROOT = Path(__file__).parent
PERSONA_PATH = os.getenv("RIFF_PERSONA", str(ROOT / "personas" / "riff.yaml"))
INDEX = os.getenv("MOSS_INDEX_NAME", "riff-tropes")


async def main():
    persona = PersonaPack.from_yaml(PERSONA_PATH)
    tropes_path = Path(persona.tropes_path or "data/comedy_tropes.json")
    if not tropes_path.is_absolute():
        tropes_path = ROOT / tropes_path

    client = MossClient(os.environ["MOSS_PROJECT_ID"], os.environ["MOSS_PROJECT_KEY"])
    entries = json.loads(tropes_path.read_text(encoding="utf-8"))
    docs = [DocumentInfo(id=e["id"], text=e["text"]) for e in entries]

    print(f"indexing {len(docs)} tropes into '{INDEX}'...")
    await client.create_index(INDEX, docs, model_id="moss-minilm")
    print(f"done. now run: python voice_agent.py console")


if __name__ == "__main__":
    asyncio.run(main())
