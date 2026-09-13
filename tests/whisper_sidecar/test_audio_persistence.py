import logging
import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_backend import audio_persistence
from whisper_backend.audio_persistence import TurnAudioCache


class _NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


def _enabled_cache(monkeypatch, *, max_buffer_bytes):
    monkeypatch.setattr(audio_persistence.threading, "Thread", _NoopThread)
    return TurnAudioCache(
        enabled=True,
        max_buffer_bytes=max_buffer_bytes,
    )


def test_turn_audio_cache_keeps_exact_cap_then_discards_whole_overflow_once(
    monkeypatch,
    caplog,
):
    cache = _enabled_cache(monkeypatch, max_buffer_bytes=8)
    cache.start_turn(41)
    cache.accumulate(b"1234")
    cache.accumulate(b"5678")
    assert bytes(cache._buffer) == b"12345678"

    with caplog.at_level(logging.WARNING, logger=audio_persistence.__name__):
        cache.accumulate(b"9")
        cache.accumulate(b"ignored")

    assert cache._buffer == bytearray()
    assert cache._turn_discarded is True
    assert cache.enqueue_turn(41, "completed", 24_000) is False
    warnings = [
        record.getMessage() for record in caplog.records
        if "server-cache sidecar exceeded cap" in record.getMessage()
    ]
    assert len(warnings) == 1
    assert "turn 41" in warnings[0]
    assert "transcription continues" in warnings[0]
    cache.close()


def test_turn_audio_cache_resets_discard_latch_for_next_turn(monkeypatch, caplog):
    cache = _enabled_cache(monkeypatch, max_buffer_bytes=4)

    with caplog.at_level(logging.WARNING, logger=audio_persistence.__name__):
        cache.start_turn(51)
        cache.accumulate(b"12345")
        cache.accumulate(b"still ignored")
        cache.start_turn(52)
        cache.accumulate(b"1234")
        assert bytes(cache._buffer) == b"1234"
        assert cache._turn_discarded is False
        cache.accumulate(b"5")

    assert cache._buffer == bytearray()
    assert cache._turn_discarded is True
    warnings = [
        record.getMessage() for record in caplog.records
        if "server-cache sidecar exceeded cap" in record.getMessage()
    ]
    assert len(warnings) == 2
    assert "turn 51" in warnings[0]
    assert "turn 52" in warnings[1]
    cache.close()
