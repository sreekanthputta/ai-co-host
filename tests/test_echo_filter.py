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


def test_should_process_false_when_muted():
    e = EchoFilter()
    e.mute_for(10.0)
    assert e.should_process("hello world") is False


def test_should_process_false_when_echo():
    e = EchoFilter(dedupe_threshold=0.7, tail_ms=0)
    e.begin_speaking(30, "That joke landed like a lead balloon")
    time.sleep(0.05)
    assert not e.is_muted()
    assert e.should_process("That joke landed like a lead balloon") is False


def test_should_process_true_otherwise():
    e = EchoFilter(dedupe_threshold=0.7, tail_ms=0)
    e.begin_speaking(30, "That joke landed like a lead balloon")
    time.sleep(0.05)
    assert e.should_process("Something completely different here") is True


def test_stop_speaking_starts_tail_window():
    e = EchoFilter(tail_ms=50)
    e.stop_speaking()
    assert e.is_muted()
    time.sleep(0.08)
    assert not e.is_muted()


def test_looks_like_echo_empty_string():
    e = EchoFilter()
    e.begin_speaking(100, "some text")
    assert e.looks_like_echo("") is False


def test_looks_like_echo_very_short_text():
    e = EchoFilter()
    e.begin_speaking(100, "some text")
    assert e.looks_like_echo("hi") is False
    assert e.looks_like_echo("  a ") is False


def test_looks_like_echo_none():
    e = EchoFilter()
    e.begin_speaking(100, "some text")
    assert e.looks_like_echo(None) is False


def test_looks_like_echo_exact_match():
    e = EchoFilter(dedupe_threshold=0.7)
    e.begin_speaking(100, "The crowd goes wild")
    assert e.looks_like_echo("The crowd goes wild") is True


def test_looks_like_echo_above_threshold():
    e = EchoFilter(dedupe_threshold=0.7)
    e.begin_speaking(100, "The crowd goes wild tonight")
    assert e.looks_like_echo("The crowd goes wild tonigh") is True


def test_looks_like_echo_below_threshold():
    e = EchoFilter(dedupe_threshold=0.7)
    e.begin_speaking(100, "The crowd goes wild tonight")
    assert e.looks_like_echo("Something entirely unrelated here") is False


def test_begin_speaking_mute_transitions_to_false():
    e = EchoFilter(tail_ms=0)
    e.begin_speaking(30, "quick line")
    assert e.is_muted()
    time.sleep(0.05)
    assert not e.is_muted()


def test_multiple_mute_for_latest_expiry_wins():
    e = EchoFilter()
    e.mute_for(0.02)
    e.mute_for(0.10)
    time.sleep(0.05)
    assert e.is_muted()
    time.sleep(0.08)
    assert not e.is_muted()


def test_multiple_mute_for_earlier_does_not_shorten():
    e = EchoFilter()
    e.mute_for(0.10)
    e.mute_for(0.01)
    time.sleep(0.05)
    assert e.is_muted()
