import wave

import pytest

from audio_spool import AudioSpoolError, TurnAudioSpool


def _stream_duration(spool, *, seconds, head, tail, chunk_bytes=64 * 1024):
    total = 48_000 * seconds
    assert len(head) % 2 == 0 and len(tail) % 2 == 0
    spool.append(head)
    remaining = total - len(head) - len(tail)
    zero_chunk = b"\0" * (chunk_bytes - chunk_bytes % 2)
    while remaining:
        chunk = zero_chunk[: min(remaining, len(zero_chunk))]
        spool.append(chunk)
        remaining -= len(chunk)
    spool.append(tail)
    return total


@pytest.mark.parametrize("seconds", [600, 1000])
def test_long_turn_spool_streams_full_head_tail_and_exact_archive(tmp_path, seconds):
    head = b"HEAD" * 8
    tail = b"TAIL" * 8
    spool = TurnAudioSpool(
        tmp_path / "spool",
        turn_id=seconds,
        sample_rate=24_000,
        channels=1,
    )
    expected_bytes = _stream_duration(
        spool,
        seconds=seconds,
        head=head,
        tail=tail,
    )
    assert spool.byte_count == expected_bytes
    assert spool.frame_count == expected_bytes // 2

    archived = spool.archive_wav(tmp_path / "recent", outcome="completed")
    with wave.open(str(archived), "rb") as handle:
        assert handle.getframerate() == 24_000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() * 2 == expected_bytes
        assert handle.readframes(len(head) // 2) == head
        handle.setpos(handle.getnframes() - len(tail) // 2)
        assert handle.readframes(len(tail) // 2) == tail
    assert spool.path.exists()
    spool.cleanup()
    assert not spool.path.exists()
    assert archived.exists()


def test_spool_supports_multiple_atomic_archives_without_loading_turn(tmp_path):
    spool = TurnAudioSpool(tmp_path / "spool", turn_id=7, sample_rate=24_000)
    spool.append(b"\x01\x00" * 100)
    failed = spool.archive_wav(tmp_path / "failed", outcome="transport_failed")
    recent = spool.archive_wav(tmp_path / "recent", outcome="transport_failed")
    assert failed.read_bytes() == recent.read_bytes()
    assert spool.byte_count == 200


def test_spool_rejects_unaligned_pcm_and_append_after_seal(tmp_path):
    spool = TurnAudioSpool(tmp_path, turn_id=1, sample_rate=24_000)
    with pytest.raises(AudioSpoolError, match="unaligned"):
        spool.append(b"x")
    spool.append(b"\x00\x00")
    spool.seal()
    with pytest.raises(AudioSpoolError, match="sealed"):
        spool.append(b"\x00\x00")

