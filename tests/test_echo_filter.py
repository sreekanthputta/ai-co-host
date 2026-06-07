import time

from src.riff.echo_filter import EchoFilter


def test_not_muted_by_default():
    assert EchoFilter().is_muted() is False


def test_mute_for_window():
    e = EchoFilter()
    e.mute_for(0.05)
    assert e.is_muted() is True
    time.sleep(0.08)
    assert e.is_muted() is False


def test_unmute_clears_window():
    e = EchoFilter()
    e.mute_for(10.0)
    assert e.is_muted()
    e.unmute()
    assert not e.is_muted()


def test_begin_speaking_mutes_for_duration():
    e = EchoFilter(tail_ms=0)
    e.begin_speaking(50, "hello world")
    assert e.is_muted()
    time.sleep(0.08)
    assert not e.is_muted()


def test_echo_dedupes_near_match():
    e = EchoFilter(dedupe_threshold=0.7)
    e.begin_speaking(100, "Tulsa accountant. Bold move bringing math.")
    assert e.looks_like_echo("Tulsa accountant. Bold move bringing math.") is True


def test_echo_passes_through_distinct_input():
    e = EchoFilter(dedupe_threshold=0.7)
    e.begin_speaking(100, "Tulsa accountant. Bold move bringing math.")
    assert e.looks_like_echo("I work in Tulsa as an accountant") is False


def test_no_last_spoken_means_no_echo():
    e = EchoFilter()
    assert e.looks_like_echo("anything") is False
