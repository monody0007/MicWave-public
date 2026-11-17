import pathlib
import re
import sys
import time


from micwave_core.audio_persistence import TurnAudioCache


def _wait_for(path: pathlib.Path):
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path.name}")


def test_enqueue_turn_file_returns_the_basename_the_worker_writes(tmp_path):
    cache = TurnAudioCache(enabled=True, cache_dir=str(tmp_path), filename_prefix="phone_")
    try:
        cache.start_turn(7)
        cache.accumulate(b"\x01\x00" * 100)
        name = cache.enqueue_turn_file(7, "completed", 24000)
        assert name is not None
        assert re.fullmatch(r"phone_\d{8}_\d{6}_\d{6}_T7_completed\.wav", name)
        _wait_for(tmp_path / name)
        assert [p.name for p in tmp_path.iterdir()] == [name]
    finally:
        cache.close()


def test_enqueue_turn_file_none_when_nothing_queued(tmp_path):
    disabled = TurnAudioCache(enabled=False, cache_dir=str(tmp_path))
    disabled.start_turn(1)
    disabled.accumulate(b"\x01\x00")
    assert disabled.enqueue_turn_file(1, "completed", 24000) is None

    cache = TurnAudioCache(enabled=True, cache_dir=str(tmp_path))
    try:
        cache.start_turn(2)
        assert cache.enqueue_turn_file(2, "completed", 24000) is None  # empty buffer
        cache.accumulate(b"\x01\x00")
        assert cache.enqueue_turn_file(2, "completed", 24000) is not None
        cache.accumulate(b"\x01\x00")
        assert cache.enqueue_turn_file(2, "completed", 24000) is None  # dedup
        assert cache.enqueue_turn(2, "completed", 24000) is False
    finally:
        cache.close()
