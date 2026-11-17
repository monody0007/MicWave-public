"""Silence-aligned soft segment boundary (task 1049).

One long provider buffer made gpt-4o-transcribe answer the content instead of
transcribing it. The server now reuses its planned segment rollover once the
active segment reaches a soft target, at the first >=300 ms low-energy window
(or at soft target + grace without one). Frames here are 20 ms PCM16 (960 B)
whose int16 amplitude doubles as an identity tag, so audio conservation across
segments is asserted by exact frame equality.
"""
import asyncio
import logging
import pathlib
import struct
import sys

import pytest


from micwave_core import whisper_adapter as dotenv_loader
from micwave_core.whisper_server import (
    TranscriptionTurnSession,
    TranscriptionTurnSessionConfig,
)
from micwave_core.whisper_adapter import QuietRunTracker

from .test_transcription_server import (
    FakeClient,
    FakeWebSocket,
    _assert_s1,
    _client_text_view,
    _idles,
    make_config,
    make_session,
)

SAMPLE_RATE = 24_000
FRAME_SAMPLES = 480                      # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2
SOFT_SEC = 1.0
GRACE_SEC = 0.5
SOFT_FRAMES = 50                         # 1.0 s soft target
GRACE_FRAMES = 25                        # +0.5 s grace


def frame(amplitude: int) -> bytes:
    """One 20 ms constant-amplitude frame; its RMS is |amplitude|."""
    return struct.pack("<h", amplitude) * FRAME_SAMPLES


def loud(seed: int) -> bytes:
    return frame(1000 + seed)            # far above the 150 RMS threshold


def quiet(seed: int) -> bytes:
    return frame(1 + seed % 100)         # <=100, below the threshold


def soft_session(**overrides):
    config = dict(soft_segment_sec=SOFT_SEC, soft_segment_grace_sec=GRACE_SEC)
    config.update(overrides)
    return make_session(**config)


def install_providers(session, count):
    """Hand out `count` fake providers in order; record each commit."""
    providers = [FakeClient() for _ in range(count)]
    committed = [asyncio.Event() for _ in providers]
    for provider, event in zip(providers, committed):
        async def _commit(event=event):
            event.set()

        provider.commit_audio = _commit
    replacements = iter(providers[1:])

    async def _init(turn_id=None, **kwargs):
        client = next(replacements)
        session._client = client
        session._provider_generation += 1
        session._provider_audio_bytes = 0
        session._begin_provider_epoch_locked()
        session._register_client_handlers(client)
        return True

    session._client = providers[0]
    session._provider_generation = 100
    session._register_client_handlers(providers[0])
    session._init_or_reuse_client = _init
    session._RECONNECT_BACKOFF_BASE_SEC = 0
    return providers, committed


async def complete(provider, item_id, text):
    await provider.handlers["input_audio_buffer.committed"]({"item_id": item_id})
    await provider.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": item_id, "transcript": text})


async def land_rollover(session):
    task = session._segment_rollover_task
    assert task is not None
    await asyncio.wait_for(asyncio.shield(task), 1)


async def stop_and_complete(session, committed_event, provider, item_id, text):
    stop = asyncio.create_task(session._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(committed_event.wait(), 1)
    await complete(provider, item_id, text)
    await asyncio.wait_for(stop, 1)


async def feed(session, frames):
    for audio in frames:
        await session._handle_audio_bytes(audio)


def boundary_logs(caplog, reason=None):
    marker = "Soft segment boundary"
    lines = [r.getMessage() for r in caplog.records if marker in r.getMessage()]
    if reason is not None:
        lines = [line for line in lines if f"reason={reason} " in line]
    return lines


# ── trigger at silence ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("first_text", "second_text", "expected"),
    [
        ("前段内容", "后段内容", "前段内容后段内容"),
        ("First part.", "Second part.", "First part. Second part."),
        ("设计这个Control", "Plane怎么样", "设计这个Control Plane怎么样"),
    ],
)
@pytest.mark.asyncio
async def test_soft_boundary_fires_at_silence_after_target_and_joins_text(
    caplog, first_text, second_text, expected,
):
    s = soft_session()
    providers, committed = install_providers(s, 2)
    first, second = providers
    # A 20-frame pause at frames 10-29 is *before* the soft target: ignored.
    # Speech 30-44 then a pause from frame 45: the 15th quiet frame is frame 59.
    frames = (
        [loud(i) for i in range(10)]
        + [quiet(i) for i in range(20)]
        + [loud(i) for i in range(10, 25)]
        + [quiet(i) for i in range(20, 75)]
    )
    assert len(frames) == 100
    with caplog.at_level(logging.INFO):
        await feed(s, frames[:59])
        assert not committed[0].is_set()      # 14 quiet frames: not yet 300 ms
        await feed(s, frames[59:60])          # 15th quiet frame, segment 1.2 s
        await asyncio.wait_for(committed[0].wait(), 1)
    assert first.sent_audio == frames[:60]    # whole chunks, none split
    assert len(boundary_logs(caplog, "silence")) == 1
    assert boundary_logs(caplog, "forced") == []
    assert s._rollover_tail_chunks == []

    await feed(s, frames[60:70])              # arrives while the prefix decodes
    await complete(first, "seg-A", first_text)
    await land_rollover(s)
    assert second.sent_audio == frames[60:70]
    assert s._finalized is False
    assert _client_text_view(s) == first_text

    await feed(s, frames[70:])
    await stop_and_complete(s, committed[1], second, "seg-B", second_text)

    assert second.sent_audio == frames[60:]
    assert first.sent_audio + second.sent_audio == frames   # nothing lost/duplicated
    assert _client_text_view(s) == expected
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.asyncio
async def test_pause_shorter_than_300ms_does_not_trigger():
    s = soft_session()
    providers, committed = install_providers(s, 2)
    # Past the soft target the speech keeps interrupting 14-frame (280 ms) pauses.
    frames = [loud(i) for i in range(SOFT_FRAMES)]
    while len(frames) < SOFT_FRAMES + GRACE_FRAMES - 1:
        k = len(frames)
        frames += [quiet(k + i) for i in range(14)] + [loud(k)]
    frames = frames[: SOFT_FRAMES + GRACE_FRAMES - 1]
    await feed(s, frames)
    assert not committed[0].is_set()
    assert s._segment_rollover_pending is False


@pytest.mark.asyncio
async def test_silence_straddling_unaligned_chunks_is_still_found():
    """Real upload chunks (160 ms) never line up with 20 ms analysis windows."""
    s = soft_session()
    providers, committed = install_providers(s, 2)
    audio = b"".join([loud(i) for i in range(48)] + [quiet(i) for i in range(40)])
    pieces = [audio[i:i + 1_234] for i in range(0, len(audio), 1_234)]
    sent_pieces = []
    for piece in pieces:
        await s._handle_audio_bytes(piece)
        sent_pieces.append(piece)
        if committed[0].is_set():
            break
    assert committed[0].is_set()
    cut = sum(len(p) for p in sent_pieces)
    assert b"".join(providers[0].sent_audio) == audio[:cut]
    quiet_bytes = cut - 48 * FRAME_BYTES
    assert 15 * FRAME_BYTES <= quiet_bytes < 15 * FRAME_BYTES + 2 * 1_234


# ── forced cut ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forced_cut_at_soft_target_plus_grace_without_silence(caplog):
    s = soft_session()
    providers, committed = install_providers(s, 2)
    first, second = providers
    frames = [loud(i) for i in range(120)]
    force = SOFT_FRAMES + GRACE_FRAMES
    with caplog.at_level(logging.INFO):
        await feed(s, frames[: force - 1])
        assert not committed[0].is_set()      # past the target, still no pause
        await feed(s, frames[force - 1:force])
        await asyncio.wait_for(committed[0].wait(), 1)
    assert first.sent_audio == frames[:force]
    assert len(boundary_logs(caplog, "forced")) == 1
    assert boundary_logs(caplog, "silence") == []

    await feed(s, frames[force:force + 5])
    await complete(first, "seg-F1", "甲")
    await land_rollover(s)
    await feed(s, frames[force + 5:])
    await stop_and_complete(s, committed[1], second, "seg-F2", "乙")

    assert first.sent_audio + second.sent_audio == frames
    assert _client_text_view(s) == "甲乙"
    _assert_s1(s)


@pytest.mark.asyncio
async def test_silence_just_before_forced_cut_wins_over_forced_cut(caplog):
    s = soft_session()
    providers, committed = install_providers(s, 2)
    frames = [loud(i) for i in range(60)] + [quiet(i) for i in range(15)]
    with caplog.at_level(logging.INFO):
        await feed(s, frames[:-1])
        assert not committed[0].is_set()
        await feed(s, frames[-1:])            # frame 75 is also the forced point
        await asyncio.wait_for(committed[0].wait(), 1)
    assert len(boundary_logs(caplog, "silence")) == 1
    assert boundary_logs(caplog, "forced") == []


# ── disabled / thresholds / short turns ──────────────────────────────────────


@pytest.mark.asyncio
async def test_zero_disables_soft_boundary_even_with_long_silence():
    s = soft_session(soft_segment_sec=0.0)
    providers, committed = install_providers(s, 2)
    assert s._soft_thresholds is None
    assert s._segment_quiet_tracker is None
    await feed(s, [quiet(i) for i in range(300)])     # 6 s of pause at 1 s target
    await feed(s, [loud(i) for i in range(300)])
    assert not committed[0].is_set()
    assert s._segment_rollover_pending is False
    assert len(providers[0].sent_audio) == 600


@pytest.mark.asyncio
async def test_soft_target_at_or_above_hard_cap_is_inert():
    s = make_session(max_turn_audio_bytes=50 * FRAME_BYTES, soft_segment_sec=60.0)
    assert s._soft_thresholds is None
    assert s._segment_quiet_tracker is None


@pytest.mark.asyncio
async def test_hard_cap_still_rolls_over_with_soft_boundary_enabled():
    hard = 100 * FRAME_BYTES
    s = soft_session(
        max_turn_audio_bytes=hard,
        soft_segment_grace_sec=10.0,          # forced point (11 s) beyond the cap
        soft_segment_silence_rms=1.0,         # nothing below 1 RMS: no silence cut
    )
    providers, committed = install_providers(s, 2)
    first, second = providers
    frames = [loud(i) for i in range(160)]
    await feed(s, frames[:100])
    assert not committed[0].is_set()
    await feed(s, frames[100:101])            # crossing frame belongs to the tail
    await asyncio.wait_for(committed[0].wait(), 1)
    assert first.sent_audio == frames[:100]
    assert s._rollover_tail_chunks == [frames[100]]
    await feed(s, frames[101:110])
    await complete(first, "seg-H1", "甲")
    await land_rollover(s)
    await feed(s, frames[110:])
    await stop_and_complete(s, committed[1], second, "seg-H2", "乙")
    assert first.sent_audio + second.sent_audio == frames
    assert _client_text_view(s) == "甲乙"
    _assert_s1(s)


@pytest.mark.asyncio
async def test_short_turn_is_unchanged_with_soft_boundary_enabled(caplog):
    s = soft_session()
    providers, committed = install_providers(s, 1)
    # 0.9 s turn (below the 1.0 s target) containing a >300 ms pause and ending
    # in silence: the soft rule must stay out of the way.
    frames = (
        [loud(i) for i in range(10)]
        + [quiet(i) for i in range(20)]
        + [loud(i) for i in range(10, 15)]
        + [quiet(i) for i in range(15, 18)]
    )
    assert len(frames) < SOFT_FRAMES
    with caplog.at_level(logging.INFO):
        await feed(s, frames)
        await stop_and_complete(s, committed[0], providers[0], "short", "短句")
    assert boundary_logs(caplog) == []
    assert providers[0].sent_audio == frames
    assert _client_text_view(s) == "短句"
    _assert_s1(s)


@pytest.mark.asyncio
async def test_default_config_keeps_59_seconds_of_continuous_speech_in_one_segment(caplog):
    """Default 60 s target: 59 s of continuous speech stays one segment."""
    s = make_session()                        # config defaults: 60 s + 30 s grace
    assert s._soft_thresholds.soft_bytes == 60 * SAMPLE_RATE * 2
    providers, committed = install_providers(s, 1)
    speech = loud(1)
    with caplog.at_level(logging.INFO):
        for _ in range(59 * 50):
            await s._handle_audio_bytes(speech)
    assert boundary_logs(caplog) == []
    assert not committed[0].is_set()


# ── stop races and consecutive soft boundaries ───────────────────────────────


@pytest.mark.asyncio
async def test_stop_right_after_soft_boundary_keeps_committed_text_with_empty_tail():
    s = soft_session()
    providers, committed = install_providers(s, 2)
    first, second = providers
    frames = [loud(i) for i in range(45)] + [quiet(i) for i in range(15)]
    await feed(s, frames)
    await asyncio.wait_for(committed[0].wait(), 1)
    assert s._rollover_tail_chunks == []

    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.sleep(0)
    assert not stop.done()
    await complete(first, "seg-E1", "整段内容")
    await asyncio.wait_for(stop, 2)

    assert first.sent_audio == frames
    assert second.sent_audio == []
    assert _client_text_view(s) == "整段内容"
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.asyncio
async def test_three_segments_cut_at_silence_each_keep_every_frame_once(caplog):
    s = soft_session()
    providers, committed = install_providers(s, 3)
    block = lambda base: (        # 55 loud + 15 quiet frames: pause ends each block
        [loud(base + i) for i in range(55)] + [quiet(base + i) for i in range(15)]
    )
    frames = block(0) + block(100) + block(200)[:40]
    texts = ["甲段内容", "乙段内容", "丙段内容"]
    with caplog.at_level(logging.INFO):
        await feed(s, frames[:70])
        await asyncio.wait_for(committed[0].wait(), 1)
        await complete(providers[0], "seg-T1", texts[0])
        await land_rollover(s)

        await feed(s, frames[70:140])
        await asyncio.wait_for(committed[1].wait(), 1)
        await complete(providers[1], "seg-T2", texts[1])
        await land_rollover(s)

        await feed(s, frames[140:])
        await stop_and_complete(s, committed[2], providers[2], "seg-T3", texts[2])

    assert len(boundary_logs(caplog, "silence")) == 2
    assert providers[0].sent_audio == frames[:70]
    assert providers[1].sent_audio == frames[70:140]
    assert providers[2].sent_audio == frames[140:]
    assert _client_text_view(s) == "".join(texts)
    _assert_s1(s)


@pytest.mark.asyncio
async def test_pause_inside_rollover_tail_counts_toward_the_next_boundary(caplog):
    """The tail's pause carries over, and decides the cut by silence, not by force.

    The first segment ends loud (a forced cut), so a tracker that kept the old
    segment's state sees no quiet run. The tail is strictly below the forced
    point (70 of 75 frames) and ends with a 300 ms pause, so only a promoted
    tail tracker can make the next chunk a *silence* boundary; a fresh tracker
    would measure 20 ms and wait for the forced point instead.
    """
    s = soft_session()
    providers, committed = install_providers(s, 3)
    first, second, third = providers
    prefix = [loud(i) for i in range(SOFT_FRAMES + GRACE_FRAMES)]
    with caplog.at_level(logging.INFO):
        await feed(s, prefix)
        await asyncio.wait_for(committed[0].wait(), 1)
        assert len(boundary_logs(caplog, "forced")) == 1
        tail = [loud(i) for i in range(100, 155)] + [quiet(i) for i in range(15)]
        assert SOFT_FRAMES <= len(tail) < SOFT_FRAMES + GRACE_FRAMES
        await feed(s, tail)
        await complete(first, "seg-P1", "甲")
        await land_rollover(s)
        # Satisfied by the promoted tail already, but a boundary is only ever
        # taken on a live chunk: nothing is committed yet.
        assert not committed[1].is_set()
        more = [quiet(15)]
        await feed(s, more)
        await asyncio.wait_for(committed[1].wait(), 1)
    assert second.sent_audio == tail + more
    assert len(second.sent_audio) < SOFT_FRAMES + GRACE_FRAMES
    assert len(boundary_logs(caplog, "silence")) == 1
    assert len(boundary_logs(caplog, "forced")) == 1       # only the first one


@pytest.mark.asyncio
async def test_loud_rollover_tail_does_not_inherit_the_old_segments_quiet_run(caplog):
    """Negative control: the previous segment's pause must not leak into the tail."""
    s = soft_session()
    providers, committed = install_providers(s, 3)
    first, second, third = providers
    prefix = [loud(i) for i in range(45)] + [quiet(i) for i in range(15)]
    with caplog.at_level(logging.INFO):
        await feed(s, prefix)
        await asyncio.wait_for(committed[0].wait(), 1)
        assert len(boundary_logs(caplog, "silence")) == 1  # old run: 15 quiet frames
        tail = [loud(i) for i in range(100, 160)]          # ends loud, 60 >= target
        assert SOFT_FRAMES <= len(tail) < SOFT_FRAMES + GRACE_FRAMES
        await feed(s, tail)
        await complete(first, "seg-N1", "甲")
        await land_rollover(s)
        await feed(s, [quiet(20)])         # 20 ms of pause: far from 300 ms
        assert not committed[1].is_set()
        assert s._segment_rollover_pending is False
        await feed(s, [quiet(21 + i) for i in range(13)])  # 14 quiet frames in all
        assert not committed[1].is_set()
        await feed(s, [quiet(40)])                         # the 15th: 300 ms
        await asyncio.wait_for(committed[1].wait(), 1)
    assert len(boundary_logs(caplog, "silence")) == 2
    assert boundary_logs(caplog, "forced") == []
    assert len(second.sent_audio) == len(tail) + 15


# ── slow-prefix fixture ──────────────────────────────────────────────────────


FORCED_FRAMES = SOFT_FRAMES + GRACE_FRAMES     # 75


async def settle(rounds=10):
    for _ in range(rounds):
        await asyncio.sleep(0)


async def open_slow_prefix(s, committed):
    """Segment A cut at the forced point; its terminal is withheld by the caller."""
    prefix = [loud(i) for i in range(FORCED_FRAMES)]
    await feed(s, prefix)
    await asyncio.wait_for(committed[0].wait(), 1)
    return prefix


# ── a nonempty final tail below the provider minimum is not dropped ──────────

MIN_COMMIT_BYTES = 4_800                       # 100 ms, the provider minimum


def assert_padded(sent, original):
    """`sent` = the original frames, then one explicit all-zero pad to 100 ms."""
    assert sent[:len(original)] == original
    pad = b"".join(sent[len(original):])
    assert pad == bytes(MIN_COMMIT_BYTES - len(original) * FRAME_BYTES)
    assert len(sent) == len(original) + 1
    assert sum(len(f) for f in sent) == MIN_COMMIT_BYTES


@pytest.mark.parametrize("n_frames", [1, 2, 3, 4])
@pytest.mark.asyncio
async def test_short_tail_stop_before_prefix_completes_is_padded_and_committed(n_frames):
    s = soft_session()
    providers, committed = install_providers(s, 2)
    prefix = await open_slow_prefix(s, committed)
    tail = [loud(100 + i) for i in range(n_frames)]       # 20-80 ms of real audio
    await feed(s, tail)
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.sleep(0)
    await complete(providers[0], "seg-T1", "甲")
    await land_rollover(s)
    await asyncio.wait_for(committed[1].wait(), 1)        # the tail is committed ...
    assert_padded(providers[1].sent_audio, tail)
    assert s._provider_audio_bytes == s._turn_audio_bytes == MIN_COMMIT_BYTES
    assert _idles(s) == []                             # ... before any idle
    await complete(providers[1], "seg-T2", "乙")          # ... and its text lands
    await asyncio.wait_for(stop, 2)
    assert _client_text_view(s) == "甲乙"
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.parametrize("n_frames", [1, 2, 3, 4])
@pytest.mark.asyncio
async def test_short_tail_stop_after_promotion_is_padded_and_committed(n_frames):
    s = soft_session()
    providers, committed = install_providers(s, 2)       # a rebuild would run dry
    prefix = await open_slow_prefix(s, committed)
    tail = [loud(100 + i) for i in range(n_frames)]
    await feed(s, tail)
    await complete(providers[0], "seg-A1", "甲")
    await land_rollover(s)                                # promoted, still recording
    assert providers[1].sent_audio == tail and not committed[1].is_set()
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(committed[1].wait(), 1)
    assert_padded(providers[1].sent_audio, tail)         # pad sent on the live provider
    assert s._provider_audio_bytes == s._turn_audio_bytes == MIN_COMMIT_BYTES
    assert _idles(s) == []
    await complete(providers[1], "seg-A2", "乙")
    await asyncio.wait_for(stop, 2)
    assert _client_text_view(s) == "甲乙"
    _assert_s1(s)


@pytest.mark.asyncio
async def test_short_tail_pad_is_replayed_after_a_failed_pad_send_on_stop_before_completion():
    s = soft_session()
    providers, committed = install_providers(s, 3)
    await open_slow_prefix(s, committed)
    tail = [loud(100 + i) for i in range(3)]
    await feed(s, tail)
    original_send = providers[1].send_audio

    async def drop_the_pad(chunk):
        return False if not any(chunk) else await original_send(chunk)

    providers[1].send_audio = drop_the_pad
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.sleep(0)
    await complete(providers[0], "seg-F1", "甲")
    await land_rollover(s)
    await asyncio.wait_for(committed[2].wait(), 1)       # recommitted on the next provider
    assert not committed[1].is_set()
    assert_padded(providers[2].sent_audio, tail)         # replay carried original + pad
    await complete(providers[2], "seg-F2", "乙")
    await asyncio.wait_for(stop, 2)
    assert _client_text_view(s) == "甲乙"
    _assert_s1(s)


@pytest.mark.asyncio
async def test_short_tail_pad_is_replayed_when_the_provider_dropped_before_stop():
    s = soft_session()
    providers, committed = install_providers(s, 3)
    await open_slow_prefix(s, committed)
    tail = [loud(100 + i) for i in range(2)]
    await feed(s, tail)
    await complete(providers[0], "seg-G1", "甲")
    await land_rollover(s)
    providers[1]._open = False                            # live provider is gone
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(committed[2].wait(), 1)
    assert not committed[1].is_set()
    assert_padded(providers[2].sent_audio, tail)
    await complete(providers[2], "seg-G2", "乙")
    await asyncio.wait_for(stop, 2)
    assert _client_text_view(s) == "甲乙"
    _assert_s1(s)


@pytest.mark.asyncio
async def test_five_frame_tail_is_exactly_the_minimum_and_needs_no_pad():
    s = soft_session()
    providers, committed = install_providers(s, 2)
    await open_slow_prefix(s, committed)
    tail = [loud(100 + i) for i in range(5)]
    await feed(s, tail)
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.sleep(0)
    await complete(providers[0], "seg-E1", "甲")
    await land_rollover(s)
    await asyncio.wait_for(committed[1].wait(), 1)
    assert providers[1].sent_audio == tail               # no pad appended
    await complete(providers[1], "seg-E2", "乙")
    await asyncio.wait_for(stop, 2)
    assert _client_text_view(s) == "甲乙"


@pytest.mark.parametrize("n_frames", [1, 4])
@pytest.mark.parametrize("stop_before_completion", [True, False])
@pytest.mark.asyncio
async def test_all_zero_short_tail_still_ends_with_insufficient_audio(
    n_frames, stop_before_completion,
):
    """Unchanged: digital silence is never padded into a commit."""
    s = soft_session()
    providers, committed = install_providers(s, 2)
    await open_slow_prefix(s, committed)
    tail = [frame(0) for _ in range(n_frames)]
    await feed(s, tail)
    stop = None
    if stop_before_completion:
        stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
        await asyncio.sleep(0)
    await complete(providers[0], "seg-Y1", "甲")
    await land_rollover(s)
    if stop is None:
        stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(stop, 2)
    assert not committed[1].is_set()
    assert sum(len(f) for f in providers[1].sent_audio) <= n_frames * FRAME_BYTES
    assert _client_text_view(s) == "甲"                   # the delivered text stands
    _assert_s1(s)


@pytest.mark.asyncio
async def test_whole_turn_shorter_than_the_minimum_still_ends_early():
    """Unchanged: nothing delivered yet, so a tiny turn is not padded/committed."""
    s = soft_session()
    providers, committed = install_providers(s, 1)
    await feed(s, [loud(i) for i in range(3)])
    await asyncio.wait_for(s._handle_stop_recording({"turn_id": 900}), 2)
    assert not committed[0].is_set()
    assert _client_text_view(s) == ""
    _assert_s1(s)


# ── the pad is gated on "an earlier segment existed", not on delivered text ──


@pytest.mark.parametrize("stop_before_promotion", [True, False])
@pytest.mark.parametrize("n_frames", [1, 4])
@pytest.mark.asyncio
async def test_short_tail_after_an_empty_result_prefix_is_padded_and_committed(
    n_frames, stop_before_promotion,
):
    """A prefix that came back blank delivered no text, but the turn was long."""
    s = soft_session()
    providers, committed = install_providers(s, 3)
    await open_slow_prefix(s, committed)
    tail = [loud(100 + i) for i in range(n_frames)]
    await feed(s, tail)
    stop = None
    if stop_before_promotion:
        stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
        await settle()
    await complete(providers[0], "seg-Z1", "")             # blank: one retry is allowed
    await asyncio.wait_for(committed[1].wait(), 1)         # the prefix is re-sent and re-committed
    await complete(providers[1], "seg-Z2", "")             # blank again: the empty result stands
    await land_rollover(s)
    assert s._committed_transcript_text == ""              # nothing was delivered
    if stop is None:
        assert providers[2].sent_audio == tail and not committed[2].is_set()
        stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(committed[2].wait(), 1)
    assert_padded(providers[2].sent_audio, tail)
    assert _idles(s) == []
    await complete(providers[2], "seg-Z3", "乙")
    await asyncio.wait_for(stop, 2)
    assert _client_text_view(s) == "乙"
    assert len(_idles(s)) == 1
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.parametrize("n_frames", [1, 4])
@pytest.mark.asyncio
async def test_short_tail_after_a_retired_silent_prefix_is_padded_and_committed(n_frames):
    """An all-zero prefix is retired, not transcribed; the speech after it still counts."""
    s = soft_session()
    providers, committed = install_providers(s, 2)
    await feed(s, [frame(0) for _ in range(SOFT_FRAMES)])  # digital silence at the soft target
    assert not committed[0].is_set()                        # retired, never committed
    assert s._committed_transcript_text == ""
    tail = [loud(100 + i) for i in range(n_frames)]
    await feed(s, tail)
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(committed[1].wait(), 1)
    assert_padded(providers[1].sent_audio, tail)
    await complete(providers[1], "seg-R1", "乙")
    await asyncio.wait_for(stop, 2)
    assert _client_text_view(s) == "乙"
    assert len(_idles(s)) == 1
    _assert_s1(s)


@pytest.mark.asyncio
async def test_all_zero_short_tail_after_a_retired_silent_prefix_still_ends_early():
    """Control: digital silence is never padded into a commit, prior segment or not."""
    s = soft_session()
    providers, committed = install_providers(s, 2)
    await feed(s, [frame(0) for _ in range(SOFT_FRAMES)])
    await feed(s, [frame(0) for _ in range(3)])
    await asyncio.wait_for(s._handle_stop_recording({"turn_id": 900}), 2)
    assert not committed[0].is_set() and not committed[1].is_set()
    assert _client_text_view(s) == ""
    _assert_s1(s)


# ── tracker unit behaviour ───────────────────────────────────────────────────


def test_quiet_run_tracker_counts_trailing_quiet_windows_only():
    t = QuietRunTracker(sample_rate=SAMPLE_RATE, rms_threshold=150)
    t.feed(frame(1000) + frame(5) + frame(6))
    assert t.run_bytes == 2 * FRAME_BYTES
    t.feed(frame(900))
    assert t.run_bytes == 0
    t.feed(frame(0) * 3)
    assert t.run_bytes == 3 * FRAME_BYTES


def test_quiet_run_tracker_threshold_is_strict_rms():
    t = QuietRunTracker(sample_rate=SAMPLE_RATE, rms_threshold=150)
    t.feed(frame(149))
    assert t.run_bytes == FRAME_BYTES
    t.feed(frame(150))
    assert t.run_bytes == 0
    t.feed(frame(-100))                      # sign is irrelevant
    assert t.run_bytes == FRAME_BYTES


def test_quiet_run_tracker_is_chunking_invariant():
    audio = b"".join(
        [frame(500)] * 3 + [frame(7)] * 20 + [frame(800)] + [frame(3)] * 17
    )
    whole = QuietRunTracker(sample_rate=SAMPLE_RATE)
    whole.feed(audio)
    for size in (1, 2, 3, 7, 959, 960, 961, 4_800, 7_680):
        t = QuietRunTracker(sample_rate=SAMPLE_RATE)
        for i in range(0, len(audio), size):
            t.feed(audio[i:i + size])
        assert t.run_bytes == whole.run_bytes == 17 * FRAME_BYTES, size


def test_quiet_run_tracker_ignores_incomplete_trailing_window():
    t = QuietRunTracker(sample_rate=SAMPLE_RATE)
    t.feed(frame(2) * 2 + bytes(100))        # 2 quiet windows + 100 B carry
    assert t.run_bytes == 2 * FRAME_BYTES
    t.feed(frame(1000)[:FRAME_BYTES - 100])  # completes a loud window
    assert t.run_bytes == 0


# ── config surface ───────────────────────────────────────────────────────────


def test_soft_segment_config_defaults_and_env(monkeypatch):
    for name in (
        "WHISPERWAVE_SOFT_SEGMENT_SEC",
        "WHISPERWAVE_SOFT_SEGMENT_GRACE_SEC",
        "WHISPERWAVE_SOFT_SEGMENT_SILENCE_RMS",
    ):
        monkeypatch.delenv(name, raising=False)
    config = TranscriptionTurnSessionConfig.from_env()
    assert (config.soft_segment_sec, config.soft_segment_grace_sec) == (60.0, 30.0)
    assert config.soft_segment_silence_rms == 150.0
    assert config.max_turn_audio_bytes == 86_400_000   # hard cap untouched

    monkeypatch.setenv("WHISPERWAVE_SOFT_SEGMENT_SEC", "0")
    monkeypatch.setenv("WHISPERWAVE_SOFT_SEGMENT_GRACE_SEC", "12.5")
    monkeypatch.setenv("WHISPERWAVE_SOFT_SEGMENT_SILENCE_RMS", "220")
    config = TranscriptionTurnSessionConfig.from_env()
    assert config.soft_segment_sec == 0.0
    assert config.soft_segment_grace_sec == 12.5
    assert config.soft_segment_silence_rms == 220.0

    monkeypatch.setenv("WHISPERWAVE_SOFT_SEGMENT_SEC", "-5")
    assert TranscriptionTurnSessionConfig.from_env().soft_segment_sec == 0.0
    monkeypatch.setenv("WHISPERWAVE_SOFT_SEGMENT_SEC", "abc")
    assert TranscriptionTurnSessionConfig.from_env().soft_segment_sec == 60.0


def test_soft_segment_env_keys_are_dotenv_allowlisted():
    for key in (
        "WHISPERWAVE_SOFT_SEGMENT_SEC",
        "WHISPERWAVE_SOFT_SEGMENT_GRACE_SEC",
        "WHISPERWAVE_SOFT_SEGMENT_SILENCE_RMS",
    ):
        assert key in dotenv_loader.DOTENV_ALLOWED_KEYS


SEC_ENV = "WHISPERWAVE_SOFT_SEGMENT_SEC"
GRACE_ENV = "WHISPERWAVE_SOFT_SEGMENT_GRACE_SEC"
RMS_ENV = "WHISPERWAVE_SOFT_SEGMENT_SILENCE_RMS"


def _clear_soft_env(monkeypatch):
    for name in (SEC_ENV, GRACE_ENV, RMS_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("raw", ["inf", "-inf", "nan", "1e309", "-1e309", "1e12"])
@pytest.mark.parametrize(
    ("name", "attr", "default"),
    [(SEC_ENV, "soft_segment_sec", 60.0), (GRACE_ENV, "soft_segment_grace_sec", 30.0)],
)
def test_non_finite_or_absurd_seconds_env_falls_back_and_session_still_builds(
    monkeypatch, caplog, name, attr, default, raw,
):
    _clear_soft_env(monkeypatch)
    monkeypatch.setenv(name, raw)
    with caplog.at_level(logging.WARNING):
        config = TranscriptionTurnSessionConfig.from_env()
    assert getattr(config, attr) == default
    assert any(name in r.getMessage() and raw in r.getMessage() for r in caplog.records)
    session = TranscriptionTurnSession(FakeWebSocket(), config)   # must not raise
    assert session._segment_quiet_tracker is not None              # soft is on


@pytest.mark.parametrize("raw", ["inf", "nan", "1e309", "1e200", "32769"])
def test_non_finite_or_overflowing_rms_env_falls_back_and_session_still_builds(
    monkeypatch, caplog, raw,
):
    _clear_soft_env(monkeypatch)
    monkeypatch.setenv(RMS_ENV, raw)
    with caplog.at_level(logging.WARNING):
        config = TranscriptionTurnSessionConfig.from_env()
    assert config.soft_segment_silence_rms == 150.0
    assert any(RMS_ENV in r.getMessage() for r in caplog.records)
    session = TranscriptionTurnSession(FakeWebSocket(), config)
    assert session._segment_quiet_tracker is not None
    session._segment_quiet_tracker.feed(loud(1) + quiet(1))        # no overflow


def test_disabled_soft_segment_ignores_unusable_grace_and_rms(monkeypatch):
    """SEC=0 turns the feature off before the other two values are used."""
    _clear_soft_env(monkeypatch)
    monkeypatch.setenv(SEC_ENV, "0")
    monkeypatch.setenv(GRACE_ENV, "1e309")
    monkeypatch.setenv(RMS_ENV, "1e200")
    session = TranscriptionTurnSession(
        FakeWebSocket(), TranscriptionTurnSessionConfig.from_env()
    )
    assert session._soft_thresholds is None
    assert session._segment_quiet_tracker is None
    assert session._tail_quiet_tracker is None

    # The same holds for a config object built by hand with unusable numbers.
    session = TranscriptionTurnSession(
        FakeWebSocket(),
        make_config(
            soft_segment_sec=0.0,
            soft_segment_grace_sec=float("inf"),
            soft_segment_silence_rms=1e200,
        ),
    )
    assert session._soft_thresholds is None


@pytest.mark.parametrize("bad_sec", [float("inf"), float("nan"), 1e308])
def test_hand_built_config_with_unusable_target_disables_soft_instead_of_raising(bad_sec):
    session = TranscriptionTurnSession(
        FakeWebSocket(), make_config(soft_segment_sec=bad_sec)
    )
    assert session._soft_thresholds is None
    assert session._segment_quiet_tracker is None


def test_hand_built_config_with_unusable_grace_and_rms_uses_defaults():
    session = TranscriptionTurnSession(
        FakeWebSocket(),
        make_config(
            soft_segment_sec=60.0,
            soft_segment_grace_sec=float("inf"),
            soft_segment_silence_rms=1e200,
        ),
    )
    thresholds = session._soft_thresholds
    assert thresholds.soft_bytes == 60 * SAMPLE_RATE * 2
    assert thresholds.force_bytes == 90 * SAMPLE_RATE * 2          # default grace
    assert thresholds.rms_threshold == 150.0


@pytest.mark.parametrize("bad", [1e200, float("inf"), float("nan"), 0.0, -3.0])
def test_quiet_run_tracker_rejects_unusable_thresholds(bad):
    with pytest.raises(ValueError):
        QuietRunTracker(sample_rate=SAMPLE_RATE, rms_threshold=bad)


# ── a boundary never commits less than the provider minimum ──────────────────


@pytest.mark.asyncio
async def test_small_positive_target_waits_for_minimum_commit_size_before_rollover():
    """soft=20 ms, grace=0 used to cut (and end the whole turn) on frame one."""
    s = make_session(soft_segment_sec=0.02, soft_segment_grace_sec=0.0)
    providers, committed = install_providers(s, 2)
    frames = [loud(i) for i in range(8)]
    await feed(s, frames[:4])                    # 80 ms: below the 100 ms minimum
    assert not committed[0].is_set()
    assert s._segment_rollover_pending is False
    assert s._finalized is False
    assert s._is_recording is True
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    assert providers[0].sent_audio == frames[:4]

    await feed(s, frames[4:5])                   # 100 ms: now committable
    await asyncio.wait_for(committed[0].wait(), 1)
    assert providers[0].sent_audio == frames[:5]
    assert s._finalized is False                 # the turn is still open
    await feed(s, frames[5:])                    # audio keeps flowing (to the tail)
    assert s._rollover_tail_chunks == frames[5:]


@pytest.mark.asyncio
async def test_small_positive_target_with_silence_also_respects_the_minimum():
    s = make_session(soft_segment_sec=0.001, soft_segment_grace_sec=60.0)
    providers, committed = install_providers(s, 2)
    await feed(s, [quiet(i) for i in range(14)])  # 280 ms of pause, >= 100 ms
    assert not committed[0].is_set()              # silence needs 300 ms
    await feed(s, [quiet(14)])
    await asyncio.wait_for(committed[0].wait(), 1)
    assert len(providers[0].sent_audio) == 15


def test_resolver_clamps_small_target_and_skips_work_when_disabled():
    from micwave_core.whisper_adapter import resolve_soft_segment_thresholds

    common = dict(
        grace_sec=0.0, rms_threshold=150.0, bytes_per_sec=SAMPLE_RATE * 2,
        min_commit_bytes=4_800, hard_cap_bytes=86_400_000,
    )
    thresholds = resolve_soft_segment_thresholds(soft_sec=0.02, **common)
    assert thresholds.soft_bytes == thresholds.force_bytes == 4_800
    assert resolve_soft_segment_thresholds(soft_sec=0.0, **common) is None
    assert resolve_soft_segment_thresholds(soft_sec=-1.0, **common) is None
    # disabled before the other arguments are even looked at
    assert resolve_soft_segment_thresholds(
        soft_sec=0.0, **{**common, "grace_sec": float("inf"), "rms_threshold": 1e200}
    ) is None


# ── stop waiter: the final segment owns its own timeout window (N2) ──────────
#
# The stop waiter used to compare only the per-segment attempt integer, which
# restarts at 0 on every promotion. Segment A on attempt 2 at stop and final
# segment B on attempt 2 later looked like "nothing happened" and B was sealed
# on A's deadline; B on attempt 1 next to A on attempt 1 got a premature retry.

STOP_WINDOW = 0.5            # transcription_finalize_timeout_sec for these tests
LATE = 0.15                  # how long after the old (A-based) deadline B answers


async def open_two_segments_at_stop(s, providers, committed, *, a_attempts):
    """A committed (terminal withheld) with a 20-frame tail B queued, then stop.

    Returns the stop task and the loop time at which its waiter started.
    `a_attempts` is the attempt A is on when the user stops (1 or 2).
    """
    await open_slow_prefix(s, committed)
    await feed(s, [loud(100 + i) for i in range(20)])
    if a_attempts == 2:
        assert await s._retry_finalize_result("test: first attempt lost")
        assert committed[1].is_set()
    assert s._finalize_commit_attempts == a_attempts
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await settle()
    return stop, asyncio.get_running_loop().time()


async def sleep_until(deadline):
    await asyncio.sleep(max(0.0, deadline - asyncio.get_running_loop().time()))


@pytest.mark.asyncio
async def test_final_segment_recommit_on_the_same_attempt_number_keeps_its_own_window():
    s = soft_session(transcription_finalize_timeout_sec=STOP_WINDOW)
    providers, committed = install_providers(s, 5)
    stop, stopped_at = await open_two_segments_at_stop(
        s, providers, committed, a_attempts=2
    )
    await complete(providers[1], "n2-a", "甲")             # A lands on its 2nd attempt
    await land_rollover(s)
    await asyncio.wait_for(committed[2].wait(), 1)         # B committed, attempt 1
    assert s._finalize_commit_attempts == 1
    await s._on_provider_disconnect()                      # B's provider drops
    assert committed[3].is_set()                           # B recommitted, attempt 2
    assert s._finalize_commit_attempts == 2                # same number A had at stop

    await sleep_until(stopped_at + STOP_WINDOW + LATE)     # past the A-based deadline
    assert not stop.done(), "B was sealed on the deadline A's attempt started"
    assert _client_text_view(s) == "甲"
    await complete(providers[3], "n2-b", "乙")             # inside B's own window
    await asyncio.wait_for(stop, 2)

    assert _client_text_view(s) == "甲乙"
    assert len(_idles(s)) == 1
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.asyncio
async def test_final_segment_is_not_retried_early_when_both_segments_are_on_attempt_one():
    s = soft_session(transcription_finalize_timeout_sec=STOP_WINDOW)
    providers, committed = install_providers(s, 5)
    stop, stopped_at = await open_two_segments_at_stop(
        s, providers, committed, a_attempts=1
    )
    await complete(providers[0], "n2-a", "甲")
    await land_rollover(s)
    await asyncio.wait_for(committed[1].wait(), 1)         # B committed, attempt 1
    assert s._finalize_commit_attempts == 1                # equals A's attempt at stop

    await sleep_until(stopped_at + STOP_WINDOW + LATE)     # past the A-based deadline
    assert not stop.done()
    assert not committed[2].is_set(), "B was retried before its own window ran out"
    assert s._finalize_commit_attempts == 1
    await complete(providers[1], "n2-b", "乙")
    await asyncio.wait_for(stop, 2)

    assert _client_text_view(s) == "甲乙"
    assert not committed[2].is_set()                       # no provider was ever rebuilt for B
    assert len(_idles(s)) == 1
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.asyncio
async def test_final_segment_that_never_answers_still_gets_one_retry_then_a_bounded_finish():
    """The new window must not make the wait unbounded or drop B's retry budget."""
    s = soft_session(transcription_finalize_timeout_sec=0.2)
    providers, committed = install_providers(s, 5)
    stop, _ = await open_two_segments_at_stop(s, providers, committed, a_attempts=1)
    await complete(providers[0], "n2-a", "甲")
    await land_rollover(s)
    await asyncio.wait_for(committed[1].wait(), 1)         # B committed, attempt 1
    started = asyncio.get_running_loop().time()

    await asyncio.wait_for(stop, 3)                        # B never gets a terminal

    elapsed = asyncio.get_running_loop().time() - started
    assert committed[2].is_set()                           # B's own retry happened
    assert s._finalize_commit_attempts == 2
    assert not committed[3].is_set()                       # and only one
    assert elapsed >= 0.2 * 2.5                            # a fresh window, the retry, another window
    assert s._finalized is True
    assert _client_text_view(s) == "甲"                    # A's text is kept
    assert len(_idles(s)) == 1
    _assert_s1(s)


@pytest.mark.asyncio
async def test_stop_waiter_still_restarts_its_window_when_the_attempt_advances_in_one_segment():
    """The attempt half of (epoch, attempt): same segment, new attempt, new window."""
    s = soft_session(transcription_finalize_timeout_sec=STOP_WINDOW)
    providers, committed = install_providers(s, 3)
    await feed(s, [loud(i) for i in range(10)])            # below the soft target
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(committed[0].wait(), 1)         # the only segment, attempt 1
    await settle()
    stopped_at = asyncio.get_running_loop().time()
    await sleep_until(stopped_at + STOP_WINDOW * 0.4)
    assert await s._retry_finalize_result("test: another owner retried")
    assert committed[1].is_set()
    assert s._finalize_commit_attempts == 2                # same epoch, next attempt

    await sleep_until(stopped_at + STOP_WINDOW + LATE)     # past the first attempt's window
    assert not stop.done(), "the retried attempt was sealed on the old attempt's deadline"
    await complete(providers[1], "n2-single", "单段")
    await asyncio.wait_for(stop, 2)

    assert _client_text_view(s) == "单段"
    assert len(_idles(s)) == 1
    _assert_s1(s)
