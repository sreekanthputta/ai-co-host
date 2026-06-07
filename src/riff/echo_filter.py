"""Echo defence - mute STT during TTS playback + last-line near-dedupe."""
from __future__ import annotations
import time
from difflib import SequenceMatcher


class EchoFilter:
    def __init__(self, dedupe_threshold: float = 0.7, tail_ms: int = 300):
        self._muted_until: float = 0.0
        self._last_spoken: str = ""
        self._dedupe_threshold = dedupe_threshold
        self._tail_ms = tail_ms

    def begin_speaking(self, expected_duration_ms: int, text: str) -> None:
        now_ms = time.time() * 1000.0
        self._muted_until = (now_ms + expected_duration_ms + self._tail_ms) / 1000.0
        self._last_spoken = text or ""

    def mute_for(self, seconds: float) -> None:
        self._muted_until = max(self._muted_until, time.time() + seconds)

    def unmute(self) -> None:
        self._muted_until = 0.0

    def is_muted(self) -> bool:
        return time.time() < self._muted_until

    def looks_like_echo(self, heard: str) -> bool:
        if not self._last_spoken or not heard.strip():
            return False
        ratio = SequenceMatcher(None, heard.lower().strip(), self._last_spoken.lower().strip()).ratio()
        return ratio >= self._dedupe_threshold
