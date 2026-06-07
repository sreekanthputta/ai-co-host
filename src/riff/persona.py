"""PersonaPack - Strategy pattern. Swap personality without touching the pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class PersonaPack:
    name: str
    voice_id: str
    system_prompt: str
    similarity_threshold: float = 0.75
    cooldown_seconds: float = 4.0
    max_response_tokens: int = 50
    tropes_path: str | None = None
    style_examples: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PersonaPack":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        examples = tuple(data.pop("style_examples", []) or ())
        return cls(style_examples=examples, **data)

    def render_system_prompt(self) -> str:
        if not self.style_examples:
            return self.system_prompt
        examples_block = "\n".join(f"- {e}" for e in self.style_examples)
        return f"{self.system_prompt}\n\nStyle examples:\n{examples_block}"
