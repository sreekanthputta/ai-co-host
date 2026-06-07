from src.riff.llm_client import MinimaxClient, Punchline


def test_parse_strict_json():
    p = MinimaxClient._parse('{"line": "Tulsa accountant", "score": 0.9}')
    assert p.line == "Tulsa accountant"
    assert p.score == 0.9


def test_parse_with_surrounding_text():
    p = MinimaxClient._parse('thinking...\n{"line": "x", "score": 0.5}\n')
    assert p.line == "x"


def test_parse_invalid_json_returns_zero():
    p = MinimaxClient._parse("totally not json")
    assert p.line == "" and p.score == 0.0


def test_parse_clamps_score():
    p = MinimaxClient._parse('{"line": "y", "score": 5.0}')
    assert p.score == 1.0
    p2 = MinimaxClient._parse('{"line": "z", "score": -0.2}')
    assert p2.score == 0.0


def test_parse_handles_string_score():
    p = MinimaxClient._parse('{"line": "a", "score": "bad"}')
    assert p.score == 0.0
