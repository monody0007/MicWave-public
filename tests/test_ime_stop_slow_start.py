"""IME stop-on-slow-start regression (task 0436 R3-M1 / FIX-2).

`_async_stop` used to `wait_for(self._session_task, 1.5)`, which CANCELS a
legitimately slow session start (the server confirmation contract is 5s + one 5s
resend). A ~3s start that would have succeeded was killed and the whole turn's
already-recorded PCM was archived instead of uploaded.

The fix waits on the matching-turn `_server_connected` Event (SSOT) up to a
budget-aligned deadline and never touches the concurrently running session task.
These tests drive the real `_async_stop` on a lightweight core (built with
`__new__` so no CoreAudio/pyaudio is needed) across four tiers:

  * within budget (three tiers standing in for 0.8s / 1.2s / 5s slow starts):
    the session task is NOT cancelled, full PCM is delivered, and exactly one
    stop_recording frame is sent.
  * over budget: archive give-up, no PCM upload, no stop frame — and the stop
    path still does not cancel the session task.
"""
import asyncio
import json
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ime_menubar
from ime_menubar import BrainwaveIMECore, IMEState


class _FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(data)


def _make_stop_core(deadline_sec, pcm_chunks):
    """A core with only the attributes/methods `_async_stop` touches."""
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core._server_connected_event = asyncio.Event()
    core._server_connected = False
    core._start_requested = True
    core._stop_session_connect_deadline_sec = deadline_sec
    core._active_turn_id = 7
    core._audio_consumer_paused = False
    core._audio_queue = None  # skip the drain block (covered elsewhere)
    core._audio_abort_task = None
    core._audio_drop_count = 0
    core._audio_ingestion_failed_reason = None
    core._failed_turn_ids = set()
    core.state = IMEState.PROCESSING
    core.ws_connected = True
    core.ws = _FakeWS()
    core.audio_buffer = list(pcm_chunks)
    core._audio_buffer_samples = sum(len(c) // 2 for c in pcm_chunks)

    core._archive_calls = []
    core._sounds = []
    core._states = []

    core._compute_stop_tail_wait_sec = lambda: 0.0

    async def _noop_async(*a, **k):
        return None

    core._close_audio_stream = _noop_async

    def _archive_failed(reason):
        core._archive_calls.append(reason)
        return f"/tmp/{reason}.wav"

    core._archive_failed_turn_audio = _archive_failed
    core._archive_recent_turn_audio = lambda reason: core._archive_calls.append(("recent", reason))
    core._play_sound = lambda name: core._sounds.append(name)

    def _set_state(s):
        core.state = s
        core._states.append(s)

    core._set_state = _set_state
    return core


def _sent_pcm(core):
    return b"".join(f for f in core.ws.sent if isinstance(f, (bytes, bytearray)))


def _stop_frames(core):
    frames = []
    for f in core.ws.sent:
        if isinstance(f, str):
            try:
                msg = json.loads(f)
            except ValueError:
                continue
            if msg.get("type") == "stop_recording":
                frames.append(msg)
    return frames


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connect_after",
    [
        0.05,  # stands in for a 0.8s start (well within budget)
        0.10,  # stands in for a 1.2s start
        0.25,  # stands in for a ~5s slow start the old 1.5s wait_for would kill
    ],
)
async def test_async_stop_slow_start_within_budget_uploads_all_pcm_once(connect_after):
    pcm = [b"\x01\x00" * 100, b"\x02\x00" * 100]
    core = _make_stop_core(deadline_sec=0.6, pcm_chunks=pcm)

    async def session_coro():
        # A legitimately slow start that reaches the matching-turn `connected`
        # ack only after `connect_after` — while the stop path is waiting.
        await asyncio.sleep(connect_after)
        core._server_connected = True

    core._session_task = asyncio.create_task(session_coro())

    await asyncio.wait_for(core._async_stop(), timeout=2.0)

    # The slow start was NOT cancelled by the stop path (the whole point).
    assert not core._session_task.cancelled()
    assert core._session_task.done()
    # All buffered PCM delivered exactly once, in order (no loss from a cancel).
    assert _sent_pcm(core) == b"".join(pcm)
    # Exactly one stop_recording frame.
    assert len(_stop_frames(core)) == 1
    assert _stop_frames(core)[0]["turn_id"] == 7
    # No archive give-up on a successful (if slow) upload.
    assert core._archive_calls == []


@pytest.mark.asyncio
async def test_async_stop_over_budget_archives_without_cancelling_session_task():
    pcm = [b"\x03\x00" * 100]
    core = _make_stop_core(deadline_sec=0.2, pcm_chunks=pcm)

    async def session_coro():
        # Never reaches `connected` within the budget.
        await asyncio.sleep(5.0)
        core._server_connected = True

    core._session_task = asyncio.create_task(session_coro())

    await asyncio.wait_for(core._async_stop(), timeout=2.0)

    # Over budget -> archive give-up, no PCM upload, no stop frame.
    assert "upload_unavailable" in core._archive_calls
    assert _sent_pcm(core) == b""
    assert _stop_frames(core) == []
    assert IMEState.IDLE in core._states
    # Even on give-up the stop path must not cancel the still-running start.
    assert not core._session_task.cancelled()

    core._session_task.cancel()
    await asyncio.gather(core._session_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_async_stop_already_connected_is_zero_wait():
    pcm = [b"\x04\x00" * 50]
    core = _make_stop_core(deadline_sec=11.0, pcm_chunks=pcm)
    core._server_connected = True  # already connected -> must not wait

    async def session_coro():
        return None

    core._session_task = asyncio.create_task(session_coro())

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await asyncio.wait_for(core._async_stop(), timeout=1.0)
    elapsed = loop.time() - t0

    assert elapsed < 0.5  # did not burn the 11s deadline
    assert _sent_pcm(core) == b"".join(pcm)
    assert len(_stop_frames(core)) == 1
    assert core._archive_calls == []
