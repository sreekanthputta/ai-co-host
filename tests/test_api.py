"""Tests for the Riff API server endpoints."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.riff.api import build_app
from src.riff.echo_filter import EchoFilter
from src.riff.telemetry import Telemetry
from src.riff.trigger import ForceTrigger


_turn_counter = 0


async def fake_handler(text, sender, respond, metadata):
    global _turn_counter
    _turn_counter += 1
    return {
        "reply": "test reply" if respond else None,
        "turn_id": f"turn-{_turn_counter}",
        "decision": "spoke" if respond else "silent",
        "latency_ms": 100 if respond else 0,
    }


def make_client(message_handler=fake_handler, transcript=None):
    global _turn_counter
    _turn_counter = 0
    state = {"persona": "comedian", "transcript": transcript or []}
    app = build_app(
        trigger=ForceTrigger(),
        telemetry=Telemetry(),
        echo=EchoFilter(),
        state=state,
        message_handler=message_handler,
    )
    return TestClient(app)


def test_message_with_respond():
    client = make_client()
    r = client.post("/message", json={"text": "hello", "respond": True})
    assert r.status_code == 200
    data = r.json()
    assert data["reply"] == "test reply"
    assert data["decision"] == "spoke"
    assert data["turn_id"].startswith("turn-")
    assert data["latency_ms"] == 100


def test_message_no_respond():
    client = make_client()
    r = client.post("/message", json={"text": "just noting", "respond": False})
    assert r.status_code == 200
    data = r.json()
    assert data["reply"] is None
    assert data["decision"] == "silent"
    assert data["latency_ms"] == 0


def test_message_missing_text():
    client = make_client()
    r = client.post("/message", json={"sender": "user"})
    assert r.status_code == 422


def test_message_no_handler():
    client = make_client(message_handler=None)
    r = client.post("/message", json={"text": "hello"})
    assert r.status_code == 501
    assert r.json()["error"] == "message handler not configured"


def test_batch_messages():
    client = make_client()
    msgs = [
        {"text": "msg1", "sender": "alice"},
        {"text": "msg2", "sender": "bob"},
        {"text": "msg3", "sender": "charlie"},
    ]
    r = client.post("/message/batch", json={"messages": msgs})
    assert r.status_code == 200
    data = r.json()
    assert data["indexed"] == 3
    assert len(data["turn_ids"]) == 3


def test_batch_empty():
    client = make_client()
    r = client.post("/message/batch", json={"messages": []})
    assert r.status_code == 200
    data = r.json()
    assert data["indexed"] == 0
    assert data["turn_ids"] == []


def test_transcript_returns_turns():
    turns = [
        {"id": "turn-1", "speaker": "host", "text": "hello", "source": "audio"},
        {"id": "turn-2", "speaker": "guest", "text": "hi", "source": "chat"},
    ]
    client = make_client(transcript=turns)
    r = client.get("/transcript")
    assert r.status_code == 200
    assert r.json()["turns"] == turns


def test_transcript_last_limit():
    turns = [{"id": f"turn-{i}", "speaker": "host", "text": f"t{i}", "source": "audio"} for i in range(10)]
    client = make_client(transcript=turns)
    r = client.get("/transcript?last=5")
    assert r.status_code == 200
    data = r.json()
    assert len(data["turns"]) == 5
    assert data["turns"][0]["id"] == "turn-5"


def test_trigger_still_works():
    client = make_client()
    r = client.post("/trigger", json={"hint": "now"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["hint"] == "now"


def test_mute_and_status():
    client = make_client()
    r = client.post("/mute", json={"seconds": 60})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["muted"] is True
