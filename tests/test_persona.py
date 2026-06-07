from pathlib import Path

import pytest

from src.riff.persona import PersonaPack


def test_render_includes_examples(persona):
    out = persona.render_system_prompt()
    assert "be funny" in out
    assert "hello world" in out
    assert "Style examples" in out


def test_render_no_examples_returns_prompt():
    p = PersonaPack(name="x", voice_id="v", system_prompt="raw prompt")
    assert p.render_system_prompt() == "raw prompt"


def test_from_yaml_round_trip(tmp_path: Path):
    yaml_text = """
name: probe
voice_id: v_x
system_prompt: hello
similarity_threshold: 0.8
cooldown_seconds: 2.5
max_response_tokens: 30
style_examples:
  - one
  - two
"""
    p = tmp_path / "p.yaml"
    p.write_text(yaml_text)
    pack = PersonaPack.from_yaml(p)
    assert pack.name == "probe"
    assert pack.style_examples == ("one", "two")
    assert pack.similarity_threshold == 0.8


def test_persona_immutable(persona):
    with pytest.raises(Exception):
        persona.name = "changed"  # type: ignore[misc]
