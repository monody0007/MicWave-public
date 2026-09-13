"""Transcription server event-flow tests (task 0436 Phase 2, WhisperWave).

Drives TranscriptionTurnSession's real handlers with fake transcription events
(no live provider) to assert the transcription-form invariants AND the R2
correctness fixes (H1/H2/H3):

  * delta streaming     — raw deltas reach the client immediately and unchanged
  * item identity (H1)   — tentative (pre-ACK) → confirmed (commit ACK) → retired
                          state machine; only a confirmed item may finalize; a
                          foreign / stale / unconfirmed / missing-id terminal is
                          fail-closed
  * single barrier (H2)  — every exit path (completed/failed/timeout/commit
                          error) sends idle once; provider text after finalize is
                          dropped (no idle→text, no idle→text→idle)
  * finalize reconciliation — partial/completed are compared after trimming and
                          the longer nonblank view wins (ties retain partial)
  * S1 invariant         — any replacement lands strictly before a single idle
  * 60-min rebuild        — the session rotates once past the max-age guard

The four `test_negprobe_*` cases replay the reviewer's deterministic negative
probe event sequences and assert the fixed (no-longer-reproducing) behaviour.
"""
import asyncio
import logging
import pathlib
import sys
import time

import pytest
from starlette.websockets import WebSocketState

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_backend.config import WHISPER_TRANSCRIBE_MODEL
from whisper_backend import realtime_server as realtime_server_module
from whisper_backend.realtime_server import (
    TranscriptionTurnSession,
    TranscriptionTurnSessionConfig,
    TransportState,
)


class FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.client_state = WebSocketState.CONNECTED

    async def send_text(self, txt):
        import json
        self.sent.append(json.loads(txt))

    async def close(self):
        self.client_state = WebSocketState.DISCONNECTED


class FakeClient:
    """Minimal provider client stand-in for reuse/rotation tests."""

    def __init__(self):
        self.refresh_called = False
        self.closed = False
        self.connected = False
        self._open = True
        self.handlers = {}
        self.sent_audio = []

    def register_handler(self, t, h):
        self.handlers[t] = h

    def set_on_disconnect(self, cb):
        self._on_disconnect = cb

    def _is_ws_open(self):
        return self._open

    async def connect(self):
        self.connected = True

    async def refresh_session(self):
        self.refresh_called = True

    async def close(self):
        self.closed = True
        self._open = False

    async def clear_audio_buffer(self):
        pass

    async def commit_audio(self):
        pass

    async def send_audio(self, chunk):
        # New send-side contract (task 0446): True iff the frame was sent. A
        # closed WS (self._open = False) reports False so the server buffers it.
        if self._open:
            self.sent_audio.append(chunk)
            return True
        return False


class ExplodingCommitClient(FakeClient):
    """Provider client whose commit raises — drives the H2 commit-error path."""

    async def commit_audio(self):
        raise RuntimeError("commit boom")


class BlockingProviderWebSocket:
    """Provider socket whose close stays in flight until the test releases it."""

    def __init__(self):
        self.close_started = asyncio.Event()
        self.release_close = asyncio.Event()

    async def close(self):
        self.close_started.set()
        await self.release_close.wait()


def make_config(**ov):
    base = dict(
        keep_provider_session=True,
        provider_session_max_turns=8,
        provider_session_max_age_sec=3300,
        provider_init_max_attempts=1,
        provider_init_retry_delay_sec=0.0,
        transcription_finalize_timeout_sec=5.0,
        transcription_ack_grace_sec=2.0,
        transcription_failure_rotate_threshold=2,
        default_source_sample_rate=24000,
        max_turn_audio_bytes=14_400_000,
        force_reconnect_after_ms=0,
    )
    base.update(ov)
    return TranscriptionTurnSessionConfig(**base)


def make_session(**ov):
    s = TranscriptionTurnSession(FakeWebSocket(), make_config(**ov))
    s._active_turn_id = 900
    s._is_recording = True
    s._transport_state = TransportState.RECORDING
    s._openai_ready.set()
    return s


def bind_confirmed(session, item_id):
    """Simulate a committed turn: item bound AND confirmed by the commit ACK."""
    session._current_item_id = item_id
    session._item_confirmed = True


def mark_commit_sent(session, *, attempts=1):
    session._commit_key = session._current_commit_key()
    session._transport_state = TransportState.COMMITTING
    session._finalize_commit_attempts = attempts


async def drain_finalize(session, timeout=2.0):
    if session._finalize_task is not None:
        await asyncio.wait_for(asyncio.shield(session._finalize_task), timeout)


def _texts(session):
    return [p for p in session._websocket.sent if p.get("type") == "text"]


def _client_text_view(session):
    """Reconstruct the IME view using its append/replacement wire semantics."""

    view = ""
    for payload in _texts(session):
        if payload.get("isNewResponse"):
            view = payload.get("content", "")
        else:
            view += payload.get("content", "")
    return view


def _idles(session):
    return [p for p in session._websocket.sent if p.get("status") == "idle"]


def _event_kinds(session):
    """Reviewer-style compact event trace: text(len)/status:idle, in order."""
    kinds = []
    for p in session._websocket.sent:
        if p.get("type") == "text":
            if p.get("content"):
                kinds.append(f"text(len={len(p['content'])},new={p.get('isNewResponse')})")
        elif p.get("type") == "status":
            kinds.append(f"status:{p.get('status')}")
    return kinds


def _assert_s1(session):
    """No text after the (single) idle; exactly one idle."""
    sent = session._websocket.sent
    idle_idxs = [i for i, p in enumerate(sent) if p.get("status") == "idle"]
    assert len(idle_idxs) == 1, f"expected exactly one idle, got {len(idle_idxs)}: {sent}"
    last_idle = idle_idxs[0]
    after = [p for p in sent[last_idle + 1:] if p.get("type") == "text" and p.get("content")]
    assert after == [], f"text after idle violates S1: {after}"


def test_websocket_wire_loggers_stay_suppressed_under_app_debug():
    assert logging.getLogger("websockets").getEffectiveLevel() >= logging.WARNING
    assert logging.getLogger("websockets.client").getEffectiveLevel() >= logging.WARNING


def test_turn_audio_segment_limit_defaults_to_300s_and_keeps_env_override(monkeypatch):
    monkeypatch.delenv("WHISPERWAVE_MAX_TURN_AUDIO_BYTES", raising=False)
    assert TranscriptionTurnSessionConfig.from_env().max_turn_audio_bytes == 86_400_000

    monkeypatch.setenv("WHISPERWAVE_MAX_TURN_AUDIO_BYTES", "48000")
    assert TranscriptionTurnSessionConfig.from_env().max_turn_audio_bytes == 48_000


def test_server_audio_cache_cap_tracks_two_audio_segments(monkeypatch):
    captured = {}

    class StubAudioCache:
        @classmethod
        def from_env(cls, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(realtime_server_module, "TurnAudioCache", StubAudioCache)
    TranscriptionTurnSession(
        FakeWebSocket(),
        make_config(max_turn_audio_bytes=48_000),
    )

    assert captured == {"max_buffer_bytes": 96_000}


@pytest.mark.asyncio
async def test_conversation_item_done_is_registered_as_metadata_event():
    session = make_session()
    client = FakeClient()
    session._register_client_handlers(client)
    assert "conversation.item.done" in client.handlers
    await client.handlers["conversation.item.done"]({
        "type": "conversation.item.done",
        "item": {"id": "synthetic-item", "content": []},
    })


@pytest.mark.asyncio
async def test_forced_close_exposes_recovering_window_to_concurrent_stop():
    """The isolated-live fault hook marks recovery before provider close awaits,
    so stop intent can latch without an unlocked cache/send mutator or a later
    failed audio frame."""

    session = make_session(force_reconnect_after_ms=0)
    client = FakeClient()
    provider_ws = BlockingProviderWebSocket()
    client.ws = provider_ws
    session._client = client

    force_task = asyncio.create_task(
        session._force_reconnect_after_delay(session._active_turn_id)
    )
    await asyncio.wait_for(provider_ws.close_started.wait(), 1.0)
    assert session._transport_state == TransportState.RECOVERING
    assert not session._openai_ready.is_set()
    assert any(
        payload.get("type") == "recovery_started"
        for payload in session._websocket.sent
    )

    stop_task = asyncio.create_task(
        session._handle_stop_recording({"turn_id": session._active_turn_id})
    )
    for _ in range(20):
        if session._transport_state == TransportState.STOP_REQUESTED:
            break
        await asyncio.sleep(0)
    assert session._transport_state == TransportState.STOP_REQUESTED
    assert not stop_task.done()  # waiting on the force hook's transport owner

    provider_ws.release_close.set()
    await asyncio.wait_for(force_task, 1.0)
    await asyncio.wait_for(stop_task, 1.0)
    assert session._finalized is True


# ── delta streaming (L2: exact content) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_delta_streams_exact_input_immediately():
    session = make_session()
    long_delta = "这是一段足够长用于触发流式发送的实时转写文本内容片段再补一些字"
    await session._on_transcription_delta({"item_id": "item_1", "delta": long_delta})
    texts = _texts(session)
    assert texts, "a provider delta should stream immediately"
    emitted = "".join(p["content"] for p in texts)
    assert emitted == long_delta
    assert all(p["isNewResponse"] is False for p in texts)
    assert texts[0]["turn_id"] == 900
    # first delta tentatively binds the item, unconfirmed until the commit ACK
    assert session._current_item_id == "item_1"
    assert session._item_confirmed is False


@pytest.mark.asyncio
async def test_repeated_latin_single_character_deltas_reach_display_losslessly():
    session = make_session()
    for delta in ("S", "S", "O", "T"):
        await session._on_transcription_delta(
            {"item_id": "item_1", "delta": delta}
        )

    assert [payload["content"] for payload in _texts(session)] == ["S", "S", "O", "T"]
    assert session._transcript_text == "SSOT"
    assert session._emitted_text == "SSOT"


@pytest.mark.parametrize(
    "deltas",
    [
        ("SSOT", "SSOT"),
        ("你好啊", "你好啊"),
        ("abc", "abc"),
        ("S", "S", "O", "T"),
    ],
)
@pytest.mark.asyncio
async def test_r3_m4_repeated_provider_chunks_append_end_to_end(deltas):
    session = make_session()
    bind_confirmed(session, "item_1")

    for delta in deltas:
        await session._on_transcription_delta(
            {"item_id": "item_1", "delta": delta}
        )
    await session._on_transcription_completed(
        {"item_id": "item_1", "transcript": ""}
    )
    await drain_finalize(session)

    expected = "".join(deltas)
    assert [payload["content"] for payload in _texts(session)] == list(deltas)
    assert session._transcript_text == expected
    assert session._normalized_final_text == expected
    assert session._emitted_text == expected
    _assert_s1(session)


# ── completed replacement semantics (H3 authority) ───────────────────────────


@pytest.mark.asyncio
async def test_completed_diverges_replaces_before_idle():
    session = make_session()
    bind_confirmed(session, "item_1")
    # A short raw delta is displayed immediately.
    await session._on_transcription_delta({"item_id": "item_1", "delta": "半句"})
    # The authoritative completed transcript differs materially.
    final = "这是完整且和流式内容不同的最终转写结果文本"
    await session._on_transcription_completed({"item_id": "item_1", "transcript": final})
    await drain_finalize(session)

    replacements = [p for p in _texts(session) if p.get("isNewResponse") is True]
    assert len(replacements) == 1, "divergent completed must replace via isNewResponse=True"
    assert replacements[0]["content"] == final
    _assert_s1(session)
    # replacement strictly before idle
    sent = session._websocket.sent
    repl_idx = next(i for i, p in enumerate(sent) if p.get("isNewResponse") is True)
    idle_idx = next(i for i, p in enumerate(sent) if p.get("status") == "idle")
    assert repl_idx < idle_idx


@pytest.mark.asyncio
async def test_completed_matches_streamed_flushes_without_replacement():
    session = make_session()
    bind_confirmed(session, "item_1")
    short = "几点了"
    await session._on_transcription_delta({"item_id": "item_1", "delta": short})
    await session._on_transcription_completed({"item_id": "item_1", "transcript": short})
    await drain_finalize(session)

    texts = _texts(session)
    # No isNewResponse replacement — completed equals streamed text.
    assert all(p.get("isNewResponse") is False for p in texts)
    # The raw partial was already sent immediately.
    assert any(p["content"] == short for p in texts)
    _assert_s1(session)


@pytest.mark.asyncio
async def test_semantic_policy_is_final_only_and_replaces_before_idle(caplog):
    session = make_session()
    bind_confirmed(session, "item_1")
    provider_raw = "这里有四个字母，缩写是 SOT，Google Cloud Code 和功利主义"
    normalized = "这里有4个字母，缩写是 SSOT，Google Cloud Code 和功利主义"

    with caplog.at_level(logging.INFO):
        await session._on_transcription_delta(
            {"item_id": "item_1", "delta": provider_raw}
        )
        partials = _texts(session)
        assert partials[-1]["content"] == provider_raw
        assert partials[-1]["isNewResponse"] is False
        assert session._emitted_text == provider_raw

        await session._on_transcription_completed(
            {"item_id": "item_1", "transcript": provider_raw}
        )
        await drain_finalize(session)

    replacements = [p for p in _texts(session) if p.get("isNewResponse") is True]
    assert len(replacements) == 1
    assert replacements[0]["content"] == normalized
    assert session._provider_completed_text == provider_raw
    assert session._normalized_final_text == normalized
    assert "number.count" in caplog.text
    assert "domain.acronym.ssot" in caplog.text
    assert provider_raw not in caplog.text
    _assert_s1(session)


@pytest.mark.parametrize(
    ("streamed", "completed"),
    [
        ("alpha ", "alpha"),
        ("alpha", "alpha "),
    ],
)
@pytest.mark.asyncio
async def test_finalize_resolver_uses_partial_on_trimmed_length_tie(
    streamed, completed
):
    session = make_session()
    bind_confirmed(session, "item_1")
    await session._on_transcription_delta(
        {"item_id": "item_1", "delta": streamed}
    )
    await session._on_transcription_completed(
        {"item_id": "item_1", "transcript": completed}
    )
    await drain_finalize(session)

    replacements = [p for p in _texts(session) if p.get("isNewResponse") is True]
    assert replacements == []
    assert session._provider_completed_text == completed
    assert session._normalized_final_text == streamed
    assert session._emitted_text == streamed
    _assert_s1(session)


@pytest.mark.asyncio
async def test_completed_only_no_deltas_emits_full_transcript_then_idle():
    session = make_session()
    bind_confirmed(session, "item_1")
    final = "问句原文完整转写没有任何流式增量"
    await session._on_transcription_completed({"item_id": "item_1", "transcript": final})
    await drain_finalize(session)
    replacements = [p for p in _texts(session) if p.get("isNewResponse") is True]
    assert len(replacements) == 1 and replacements[0]["content"] == final
    _assert_s1(session)


@pytest.mark.asyncio
async def test_completed_longer_than_stream_replaces_wholesale():
    session = make_session()
    bind_confirmed(session, "item_1")
    await session._on_transcription_delta({"item_id": "item_1", "delta": "短的流式内容片段仅此而已"})
    longer = "这是明显更长的权威最终转写结果文本内容比流式版本多出许多字符"
    await session._on_transcription_completed({"item_id": "item_1", "transcript": longer})
    await drain_finalize(session)
    replacements = [p for p in _texts(session) if p.get("isNewResponse") is True]
    assert len(replacements) == 1
    assert replacements[0]["content"] == longer
    assert session._emitted_text == longer
    _assert_s1(session)


@pytest.mark.asyncio
async def test_completed_shorter_than_stream_keeps_longer_partial():
    """Grapeot defense: a shorter completed cannot truncate the longer partial."""
    session = make_session()
    bind_confirmed(session, "item_1")
    long_partial = "这是一段很长的流式转写文本内容一直在增长而且相当冗长啰嗦重复"
    await session._on_transcription_delta({"item_id": "item_1", "delta": long_partial})
    streamed = session._current_emitted_view()
    short_final = "简短的最终结果"
    assert len(short_final) < len(streamed)
    await session._on_transcription_completed({"item_id": "item_1", "transcript": short_final})
    await drain_finalize(session)

    replacements = [p for p in _texts(session) if p.get("isNewResponse") is True]
    assert replacements == []
    assert session._emitted_text == long_partial
    _assert_s1(session)


@pytest.mark.parametrize("completed", ["", " ", "\n\t"])
@pytest.mark.asyncio
async def test_semantically_blank_completed_uses_raw_partial_and_applies_policy_once(
    completed, monkeypatch
):
    session = make_session()
    bind_confirmed(session, "item_1")
    partial = "缩写是 SOT"
    policy_inputs = []
    real_normalize = realtime_server_module.normalize_final_transcript

    def track_policy_call(text):
        policy_inputs.append(text)
        return real_normalize(text)

    monkeypatch.setattr(
        realtime_server_module, "normalize_final_transcript", track_policy_call
    )

    await session._on_transcription_delta({"item_id": "item_1", "delta": partial})
    assert [payload["content"] for payload in _texts(session)] == [partial]

    await session._on_transcription_completed(
        {"item_id": "item_1", "transcript": completed}
    )
    await drain_finalize(session)

    replacements = [p for p in _texts(session) if p.get("isNewResponse") is True]
    assert [payload["content"] for payload in replacements] == ["缩写是 SSOT"]
    assert policy_inputs == [partial]
    assert session._provider_completed_text == completed
    assert session._transcript_text == partial
    assert session._normalized_final_text == "缩写是 SSOT"
    assert session._emitted_text == "缩写是 SSOT"
    _assert_s1(session)


@pytest.mark.parametrize("completed", ["", " ", "\n\t"])
@pytest.mark.asyncio
async def test_semantically_blank_completed_without_partial_stays_empty(
    completed, monkeypatch
):
    session = make_session()
    bind_confirmed(session, "item_1")
    policy_inputs = []
    real_normalize = realtime_server_module.normalize_final_transcript

    def track_policy_call(text):
        policy_inputs.append(text)
        return real_normalize(text)

    monkeypatch.setattr(
        realtime_server_module, "normalize_final_transcript", track_policy_call
    )

    await session._on_transcription_completed(
        {"item_id": "item_1", "transcript": completed}
    )
    await drain_finalize(session)

    assert [p for p in _texts(session) if p.get("content")] == []  # nothing emitted
    assert policy_inputs == [""]
    assert session._provider_completed_text == completed
    assert session._normalized_final_text == ""
    assert session._emitted_text == ""
    assert session._finalized is True
    _assert_s1(session)


@pytest.mark.asyncio
async def test_r3_m5_authoritative_empty_final_clears_streamed_whitespace_before_idle(
    monkeypatch,
):
    session = make_session()
    bind_confirmed(session, "item_1")
    policy_inputs = []
    real_normalize = realtime_server_module.normalize_final_transcript

    def track_policy_call(text):
        policy_inputs.append(text)
        return real_normalize(text)

    monkeypatch.setattr(
        realtime_server_module, "normalize_final_transcript", track_policy_call
    )

    await session._on_transcription_delta(
        {"item_id": "item_1", "delta": " \t"}
    )
    await session._on_transcription_completed(
        {"item_id": "item_1", "transcript": ""}
    )
    await drain_finalize(session)

    text_payloads = _texts(session)
    assert [payload["content"] for payload in text_payloads] == [" \t", ""]
    assert [payload["isNewResponse"] for payload in text_payloads] == [False, True]
    replacement_index = session._websocket.sent.index(text_payloads[-1])
    idle_index = next(
        index
        for index, payload in enumerate(session._websocket.sent)
        if payload.get("status") == "idle"
    )
    assert replacement_index < idle_index
    assert policy_inputs == [""]
    assert session._provider_completed_text == ""
    assert session._transcript_text == " \t"
    assert session._normalized_final_text == ""
    assert session._emitted_text == ""
    _assert_s1(session)


@pytest.mark.asyncio
async def test_failed_without_completed_falls_back_to_partial():
    """Only failed/timeout with no completed may fall back to the streamed
    partial (H3)."""
    session = make_session()
    bind_confirmed(session, "item_1")
    partial = "失败前已经流式出来的部分内容文字片段信息补充"
    await session._on_transcription_delta({"item_id": "item_1", "delta": partial})
    await session._on_transcription_failed({"item_id": "item_1", "error": {"type": "server_error"}})
    await drain_finalize(session)
    # No replacement (no completed); the already-streamed raw partial is retained.
    assert all(p.get("isNewResponse") is False for p in _texts(session))
    assert session._emitted_text == partial
    _assert_s1(session)


# ── item identity state machine (H1) ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_foreign_item_id_deltas_are_dropped():
    session = make_session()
    # First delta tentatively binds item_1.
    await session._on_transcription_delta({"item_id": "item_1", "delta": "aaaa"})
    assert session._current_item_id == "item_1"
    before = session._transcript_text
    # A foreign delta (different item_id) must be ignored.
    await session._on_transcription_delta({"item_id": "item_2", "delta": "ZZZZ"})
    assert session._transcript_text == before
    # Same-item delta is processed.
    await session._on_transcription_delta({"item_id": "item_1", "delta": "bbbb"})
    assert session._transcript_text != before


@pytest.mark.asyncio
async def test_item_id_confirmed_from_input_buffer_committed():
    session = make_session()
    await session._on_input_buffer_committed({"item_id": "item_42"})
    assert session._current_item_id == "item_42"
    assert session._item_confirmed is True
    # A delta from a different item is now foreign and dropped.
    await session._on_transcription_delta({"item_id": "item_99", "delta": "nope"})
    assert session._transcript_text == ""


@pytest.mark.asyncio
async def test_completed_on_unconfirmed_item_is_fail_closed():
    """A completed can never self-bind an unbound turn or finalize without a
    commit ACK (core of H1). With the grace net disabled it is HELD PENDING (not
    lost), but still grants no terminal authority until the ACK confirms it."""
    session = make_session(transcription_ack_grace_sec=0)  # grace disabled: no timer
    mark_commit_sent(session)
    await session._on_transcription_completed({"item_id": "solo", "transcript": "无 ACK 的完成事件"})
    assert session._current_item_id is None  # did NOT self-bind
    assert not session._item_confirmed
    assert session._finalize_task is None
    assert not session._finalized
    assert [p for p in _texts(session) if p.get("content")] == []  # no text emitted
    # Held pending its own commit ACK rather than dropped (H1 residual).
    assert "solo" in session._pending_terminals


@pytest.mark.asyncio
async def test_precommit_terminal_cannot_arm_grace_or_finalize_active_recording():
    session = make_session(transcription_ack_grace_sec=0.01)
    session._client = FakeClient()
    frames = [_distinct_frame(seed) for seed in range(93, 98)]
    for frame in frames:
        await session._handle_audio_bytes(frame)

    await session._on_transcription_completed(
        {"item_id": "precommit-ghost", "transcript": "stale"}
    )
    await asyncio.sleep(0.03)

    assert session._commit_key is None
    assert session._pending_terminals == {}
    assert session._ack_grace_task is None
    assert session._transport_state == TransportState.RECORDING
    assert session._finalized is False
    assert list(session._turn_audio_chunks) == frames
    assert _idles(session) == []


@pytest.mark.asyncio
async def test_terminal_missing_item_id_is_fail_closed():
    session = make_session()
    bind_confirmed(session, "item_1")
    await session._on_transcription_delta({"item_id": "item_1", "delta": "半句流式"})
    # A completed with NO item_id must not finalize (fail-closed).
    await session._on_transcription_completed({"transcript": "缺 item_id 的完成"})
    assert session._finalize_task is None
    assert not session._finalized


@pytest.mark.asyncio
async def test_commit_ack_overrides_stale_tentative_binding():
    session = make_session()
    # A stale delta tentatively binds the wrong item and streams content.
    await session._on_transcription_delta({"item_id": "stale", "delta": "上一轮泄漏进来的流式内容片段较长一点"})
    assert session._current_item_id == "stale"
    assert not session._item_confirmed
    # The commit ACK for the REAL item overrides the tentative binding.
    await session._on_input_buffer_committed({"item_id": "real"})
    assert session._current_item_id == "real"
    assert session._item_confirmed is True
    assert "stale" in session._retired_item_ids
    assert session._transcript_text == ""  # tentative content discarded
    # A display reset (isNewResponse empty) is sent to clear the wrong text.
    assert any(
        p.get("type") == "text" and p.get("isNewResponse") is True and p.get("content") == ""
        for p in session._websocket.sent
    )
    # A late delta from the now-retired stale item is dropped.
    await session._on_transcription_delta({"item_id": "stale", "delta": "更多泄漏"})
    assert session._transcript_text == ""
    # The real item streams normally.
    await session._on_transcription_delta({"item_id": "real", "delta": "真正内容"})
    assert session._transcript_text == "真正内容"


# ── R4 N1: commit ACK authority is fail-closed & rebind is atomic ─────────────


@pytest.mark.asyncio
async def test_confirmed_current_survives_mismatched_late_ack():
    """N1: once the current item is confirmed, a late mismatched ACK must be
    DROPPED — it may neither retire the real current item nor confirm a different
    id (reviewer's retired_ack_after_current_ack: current stayed item_A while
    both A and B ended retired and the real completed no longer finalized)."""
    session = make_session()
    await session._on_input_buffer_committed({"item_id": "item_A"})
    assert session._current_item_id == "item_A" and session._item_confirmed
    # A late, divergent ACK arrives (a stale committed from another item).
    await session._on_input_buffer_committed({"item_id": "item_B"})
    assert session._current_item_id == "item_A"       # real current untouched
    assert session._item_confirmed is True
    assert "item_A" not in session._retired_item_ids  # NOT retired by the stray ACK
    # item_A still has terminal authority: its completed finalizes normally.
    await session._on_transcription_completed({"item_id": "item_A", "transcript": "真正结果"})
    await drain_finalize(session)
    assert session._finalized
    _assert_s1(session)


@pytest.mark.asyncio
async def test_retired_item_ack_is_fail_closed():
    """N1: an ACK whose item_id is already retired (a past turn) must be dropped,
    never re-confirming a retired id."""
    session = make_session()
    await session._on_input_buffer_committed({"item_id": "item_A"})
    await session._on_transcription_completed({"item_id": "item_A", "transcript": "第一轮"})
    await drain_finalize(session)
    async with session._transport_lock:
        session._reset_turn_state(901)
    session._is_recording = True
    session._openai_ready.set()
    assert "item_A" in session._retired_item_ids
    # A stray ACK for the retired item_A must not confirm anything.
    await session._on_input_buffer_committed({"item_id": "item_A"})
    assert session._current_item_id is None
    assert not session._item_confirmed
    # The real turn-2 item still confirms cleanly.
    await session._on_input_buffer_committed({"item_id": "item_B"})
    assert session._current_item_id == "item_B" and session._item_confirmed


@pytest.mark.asyncio
async def test_missing_id_ack_does_not_promote_tentative():
    """N1: an ACK with no item_id is malformed and must NOT promote a tentative
    binding to confirmed (reviewer's missing_id_ack_promotes_tentative, which
    finalized a stale item)."""
    session = make_session(transcription_ack_grace_sec=0)
    await session._on_transcription_delta({"item_id": "stale", "delta": "上一轮泄漏内容片段"})
    assert session._current_item_id == "stale" and not session._item_confirmed
    # Malformed ACK (no item_id): fail-closed, tentative stays UNCONFIRMED.
    await session._on_input_buffer_committed({"transcript": "no id"})
    assert not session._item_confirmed
    # A completed for the still-unconfirmed stale item cannot finalize.
    await session._on_transcription_completed({"item_id": "stale", "transcript": "无授权完成"})
    assert not session._finalized
    assert session._finalize_task is None
    assert len(_idles(session)) == 0


@pytest.mark.asyncio
async def test_clear_failure_during_rebind_keeps_atomic_identity():
    """N1: if the best-effort display clear raises during a mismatch rebind, the
    internal identity transition must already be COMMITTED — current==real,
    confirmed, stale retired (reviewer's clear_failure_during_rebind left
    current='stale'/confirmed=False)."""
    session = make_session()
    await session._on_transcription_delta({"item_id": "stale", "delta": "错误暂绑的流式内容较长一点"})
    assert session._current_item_id == "stale"

    async def boom():
        raise RuntimeError("clear boom")

    session._reset_client_display = boom  # display clear fails mid-rebind
    await session._on_input_buffer_committed({"item_id": "real"})  # must NOT raise
    # Internal identity is atomic despite the clear failure.
    assert session._current_item_id == "real"
    assert session._item_confirmed is True
    assert "stale" in session._retired_item_ids
    assert session._transcript_text == ""  # tentative content discarded
    # The rebound real item finalizes normally.
    await session._on_transcription_completed({"item_id": "real", "transcript": "真正的最终结果"})
    await drain_finalize(session)
    assert session._finalized
    _assert_s1(session)


# ── R4 FIX-2 (H1 residual): terminal before its own ACK — bounded recovery ────


@pytest.mark.asyncio
async def test_completed_before_own_ack_is_replayed_on_ack():
    """The item's own completed arrives BEFORE its commit ACK. It is stashed (not
    dropped); the ACK confirms the item and replays it → normal finalize, no
    120s hang. Mirrors the normal live order producing the same end state."""
    session = make_session()
    mark_commit_sent(session)
    final = "自身结果早于ACK到达的最终转写文本内容"
    await session._on_transcription_completed({"item_id": "X", "transcript": final})
    assert not session._finalized
    assert session._finalize_task is None
    assert "X" in session._pending_terminals  # stashed, not lost
    # The commit ACK confirms X and replays the stashed completed.
    await session._on_input_buffer_committed({"item_id": "X"})
    assert session._current_item_id == "X" and session._item_confirmed
    assert "X" not in session._pending_terminals  # consumed on confirm
    await drain_finalize(session)
    assert session._finalized
    replacements = [p for p in _texts(session) if p.get("isNewResponse") is True]
    assert replacements and replacements[-1]["content"] == final
    _assert_s1(session)


@pytest.mark.asyncio
async def test_normal_ack_then_completed_order_is_unchanged():
    """Zero behavior change for the live-normal order (ACK first, completed
    ~1.1s later): no terminal is ever stashed and no grace timer is armed."""
    session = make_session()
    await session._on_input_buffer_committed({"item_id": "X"})
    assert session._ack_grace_task is None
    assert not session._pending_terminals
    await session._on_transcription_completed({"item_id": "X", "transcript": "正常顺序最终结果"})
    assert not session._pending_terminals  # never stashed on the happy path
    await drain_finalize(session)
    assert session._finalized
    _assert_s1(session)


@pytest.mark.asyncio
async def test_completed_before_ack_grace_finalizes_without_120s_hang():
    """If the commit ACK NEVER arrives after an early completed, the bounded
    ACK-grace window finalizes (failure reason + provider rotate) — total time
    far below the 120s stop timeout."""
    session = make_session(transcription_ack_grace_sec=0.1, transcription_finalize_timeout_sec=120.0)
    session._client = FakeClient()
    mark_commit_sent(session, attempts=2)  # retry budget already consumed
    start = time.monotonic()
    await session._on_transcription_completed({"item_id": "orphan", "transcript": "永远等不到ACK"})
    assert "orphan" in session._pending_terminals
    await asyncio.sleep(0.3)  # wait out the grace window (+ margin)
    elapsed = time.monotonic() - start
    assert session._finalized is True
    assert session._turn_done.is_set()
    assert elapsed < 5.0                       # far below the 120s stop timeout
    assert session._client is None             # provider rotated on failure finalize
    assert len(_idles(session)) == 1
    _assert_s1(session)


@pytest.mark.asyncio
async def test_ack_grace_rechecks_confirmation_after_waiting_for_transport_owner():
    """An ACK that wins while the expired grace timer is queued on the
    transport owner must remain authoritative; the timer may not finalize the
    turn as an ACK timeout from its stale pre-lock observation."""

    session = make_session(transcription_ack_grace_sec=0.01)
    mark_commit_sent(session)
    await session._on_transcription_completed(
        {"item_id": "X", "transcript": "ACK 竞态后的最终结果"}
    )
    grace_task = session._ack_grace_task
    assert grace_task is not None

    # Keep the timer alive after the ACK so it must execute its under-lock
    # recheck rather than relying on cancellation for correctness.
    cancel_grace = session._cancel_ack_grace_timer
    session._cancel_ack_grace_timer = lambda: None
    try:
        async with session._transport_lock:
            await asyncio.sleep(0.03)  # timer expires and queues on this owner
            await session._on_input_buffer_committed({"item_id": "X"})
            assert session._item_confirmed is True
        await asyncio.wait_for(grace_task, 1.0)
        await drain_finalize(session)
    finally:
        session._cancel_ack_grace_timer = cancel_grace

    assert session._finalized is True
    assert session._provider_completed_text == "ACK 竞态后的最终结果"
    assert _texts(session)[-1]["content"] == "ACK 竞态后的最终结果"
    _assert_s1(session)


# ── R4 N2: finalize barrier terminates even when replacement send fails ───────


class ExplodingSendWebSocket(FakeWebSocket):
    """WebSocket whose send_text always raises — simulates a broken client socket
    so the completed replacement send fails inside the finalize barrier."""

    async def send_text(self, txt):
        raise RuntimeError("socket send boom")


class BlockingDeltaWebSocket(FakeWebSocket):
    """Hold one ordinary delta inside send_text to exercise the S1 race."""

    def __init__(self, blocked_content):
        super().__init__()
        self.blocked_content = blocked_content
        self.delta_in_flight = asyncio.Event()
        self.release_delta = asyncio.Event()

    async def send_text(self, txt):
        import json

        payload = json.loads(txt)
        if (
            payload.get("type") == "text"
            and payload.get("isNewResponse") is False
            and payload.get("content") == self.blocked_content
        ):
            self.delta_in_flight.set()
            await self.release_delta.wait()
        self.sent.append(payload)


@pytest.mark.asyncio
async def test_replacement_send_failure_finalizes_turn():
    """N2: when the completed replacement send raises inside the finalize barrier,
    the turn must still TERMINATE — finalized/turn_done set, idle best-effort
    (<=1), provider rotated, later provider text dropped — never wedged at
    _finalizing=True/_finalized=False until the 120s stop timeout."""
    session = make_session()
    session._websocket = ExplodingSendWebSocket()
    session._client = FakeClient()
    bind_confirmed(session, "item_1")
    # A divergent completed forces _apply_completed_replacement to send (and raise).
    await session._on_transcription_completed({"item_id": "item_1", "transcript": "权威最终结果与流式不同"})
    await drain_finalize(session)
    assert session._finalized is True
    assert session._turn_done.is_set()
    assert len(session._websocket.sent) == 0   # broken socket: idle send also failed (<=1)
    assert session._client is None             # provider rotated on failure finalize
    # Post-finalize provider text is dropped.
    await session._on_transcription_delta({"item_id": "item_1", "delta": "迟到"})
    assert len(session._websocket.sent) == 0


@pytest.mark.asyncio
async def test_inflight_delta_drains_before_final_replacement_and_idle():
    """S1: a delta already inside a backpressured client write must finish
    before finalize performs its replacement+idle transition. A queued/late
    delta may never resume after idle."""

    inflight = "在途流式片段"
    websocket = BlockingDeltaWebSocket(inflight)
    session = make_session()
    session._websocket = websocket
    bind_confirmed(session, "item_1")

    delta_task = asyncio.create_task(
        session._on_transcription_delta(
            {"item_id": "item_1", "delta": inflight}
        )
    )
    await asyncio.wait_for(websocket.delta_in_flight.wait(), 1.0)

    final = "这是更长的权威最终转写结果，用于强制最终替换"
    await session._on_transcription_completed(
        {"item_id": "item_1", "transcript": final}
    )
    for _ in range(20):
        if session._finalizing:
            break
        await asyncio.sleep(0)
    assert session._finalizing is True
    assert _idles(session) == []  # finalize is blocked behind the in-flight delta

    websocket.release_delta.set()
    await asyncio.wait_for(delta_task, 1.0)
    await drain_finalize(session)

    kinds = _event_kinds(session)
    assert kinds == [
        f"text(len={len(inflight)},new=False)",
        f"text(len={len(final)},new=True)",
        "status:idle",
    ]
    assert _texts(session)[-1]["content"] == final
    _assert_s1(session)


# ── reviewer deterministic negative probes (must be green after fix) ──────────


@pytest.mark.asyncio
async def test_negprobe_foreign_completed_before_committed_binding():
    """Replay: a stale/foreign completed arrives before the commit ACK binds the
    real item. It must NOT claim the turn, NOT finalize, and the real commit must
    still bind + finalize. (bound_item_is_stale / prematurely_finalized = false)"""
    session = make_session()
    # Stale completed from a prior/foreign speech turn, still in flight.
    await session._on_transcription_completed({"item_id": "old_item", "transcript": "旧一轮的残留结果文本"})
    assert session._current_item_id is None, "stale completed must not bind the turn"
    assert not session._finalized
    assert session._finalize_task is None
    assert [p for p in _texts(session) if p.get("content")] == []
    # The real commit ACK binds + confirms; the real completed then finalizes.
    await session._on_input_buffer_committed({"item_id": "cur_item"})
    assert session._current_item_id == "cur_item" and session._item_confirmed
    final = "本轮真正的转写结果文本内容"
    await session._on_transcription_completed({"item_id": "cur_item", "transcript": final})
    await drain_finalize(session)
    replacements = [p for p in _texts(session) if p.get("isNewResponse") is True]
    assert replacements and replacements[-1]["content"] == final
    _assert_s1(session)


@pytest.mark.asyncio
async def test_negprobe_retired_completed_after_turn_boundary():
    """Replay regression `old completed → current committed → current completed`
    across a turn boundary: the previous turn's item is retired, so its late
    completed cannot rebind/finalize the new turn."""
    session = make_session()
    await session._on_input_buffer_committed({"item_id": "item_A"})
    await session._on_transcription_completed({"item_id": "item_A", "transcript": "第一轮结果"})
    await drain_finalize(session)
    # Turn 2 begins.
    async with session._transport_lock:
        session._reset_turn_state(901)
    session._is_recording = True
    session._openai_ready.set()
    assert "item_A" in session._retired_item_ids
    sent_before = len(session._websocket.sent)
    # A late completed for the retired item_A is dropped.
    await session._on_transcription_completed({"item_id": "item_A", "transcript": "迟到的旧结果"})
    assert session._finalize_task is None and not session._finalized
    assert len(session._websocket.sent) == sent_before
    # The real turn-2 item confirms and finalizes.
    await session._on_input_buffer_committed({"item_id": "item_B"})
    await session._on_transcription_completed({"item_id": "item_B", "transcript": "第二轮结果"})
    await drain_finalize(session)
    assert session._current_item_id == "item_B"
    assert session._finalized


@pytest.mark.asyncio
async def test_negprobe_commit_exception_then_late_completed():
    """Replay: commit raises → single idle via finalize barrier (provider
    rotated). A late completed after finalize is fully dropped — never the buggy
    idle → text → idle."""
    session = make_session()
    bind_confirmed(session, "item_1")
    session._client = ExplodingCommitClient()
    attempts = 0

    async def _fail_reconnect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return False

    session._init_or_reuse_client = _fail_reconnect
    session._RECONNECT_BACKOFF_BASE_SEC = 0
    for seed in range(81, 86):
        await session._handle_audio_bytes(_distinct_frame(seed))
    await session._handle_stop_recording({"turn_id": 900})
    assert attempts == 5
    assert [
        p["type"] for p in session._websocket.sent
        if p.get("type") in {"recovery_started", "recovery_failed"}
    ] == ["recovery_started", "recovery_failed"]
    assert len(_idles(session)) == 1
    assert session._finalized is True
    assert session._client is None, "commit error must rotate the provider"
    # Late completed after finalize: dropped, no second idle, no text.
    await session._on_transcription_completed({"item_id": "item_1", "transcript": "迟到结果长度二十个字左右测试"})
    assert len(_idles(session)) == 1
    assert _event_kinds(session) == ["status:idle"]
    _assert_s1(session)


@pytest.mark.asyncio
async def test_negprobe_late_delta_after_timeout_idle():
    """Replay: timeout finalizes (idle), then a late delta for the same item must
    be dropped — never the buggy idle → text."""
    session = make_session()
    bind_confirmed(session, "item_1")
    await session._on_transcription_delta({"item_id": "item_1", "delta": "半"})
    # Simulate the stop-recording timeout finalization.
    await session._finalize_turn("timeout")
    assert len(_idles(session)) == 1
    n_text = len([p for p in _texts(session) if p.get("content")])
    await session._on_transcription_delta(
        {"item_id": "item_1", "delta": "这是超时之后迟到的一大段流式增量文本本不该上屏"}
    )
    assert len([p for p in _texts(session) if p.get("content")]) == n_text  # no new text
    _assert_s1(session)


@pytest.mark.asyncio
async def test_negprobe_shorter_completed_authority():
    """Replay: streamed long, completed short → longer partial prevents truncation."""
    session = make_session()
    bind_confirmed(session, "item_1")
    await session._on_transcription_delta(
        {"item_id": "item_1", "delta": "流式阶段产生了很长的文本内容一直在累积增长七十字左右非常冗长"}
    )
    completed = "简短二十字最终结果文本"
    actual_before = session._current_emitted_view()
    assert len(completed) < len(actual_before)
    await session._on_transcription_completed({"item_id": "item_1", "transcript": completed})
    await drain_finalize(session)
    assert session._emitted_text == actual_before
    _assert_s1(session)


# ── post-finalize late events (M5) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_finalize_events_all_dropped():
    session = make_session()
    bind_confirmed(session, "item_1")
    await session._on_transcription_completed({"item_id": "item_1", "transcript": "正常完成结果"})
    await drain_finalize(session)
    assert session._finalized
    idle_before = len(_idles(session))
    text_before = len([p for p in _texts(session) if p.get("content")])
    # A whole barrage of post-finalize events must be no-ops.
    await session._on_transcription_delta({"item_id": "item_1", "delta": "迟到 delta"})
    await session._on_transcription_completed({"item_id": "item_1", "transcript": "迟到 completed"})
    await session._on_transcription_failed({"item_id": "item_1", "error": {"type": "x"}})
    await session._on_input_buffer_committed({"item_id": "item_1"})
    sent_before_error = len(session._websocket.sent)
    await session._on_error({
        "error": {"type": "server_error", "code": "late", "message": "late"}
    })
    assert len(session._websocket.sent) == sent_before_error
    assert len(_idles(session)) == idle_before
    assert len([p for p in _texts(session) if p.get("content")]) == text_before
    _assert_s1(session)


# ── item_id alignment (kept from R1, now confirmed-aware) ─────────────────────


@pytest.mark.asyncio
async def test_foreign_completed_does_not_finalize_confirmed_turn():
    session = make_session()
    bind_confirmed(session, "item_1")
    await session._on_transcription_delta({"item_id": "item_1", "delta": "aaaa"})
    # A completed for a foreign item must not finalize this (confirmed) turn.
    await session._on_transcription_completed({"item_id": "item_2", "transcript": "ZZZZ done"})
    assert session._finalize_task is None
    assert not session._finalized


# ── 60-minute preventive rebuild ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fresh_session_is_reused_not_rebuilt():
    session = make_session()
    fake = FakeClient()
    session._client = fake
    session._active_model = WHISPER_TRANSCRIBE_MODEL
    session._provider_session_started_at = time.time()  # fresh
    session._provider_session_turns = 0
    created = []

    async def fake_create():
        c = FakeClient()
        created.append(c)
        return c

    session._create_client = fake_create
    async with session._transport_lock:
        ok = await session._init_or_reuse_client(turn_id=1)
    assert ok is True
    # Prompt is configured once when the provider session is created. A
    # per-turn session.update would either resend or clear that prompt.
    assert fake.refresh_called is False
    assert created == []  # reused, no new client built
    assert session._client is fake


@pytest.mark.asyncio
async def test_session_past_max_age_is_rotated():
    session = make_session(provider_session_max_age_sec=3300)
    old = FakeClient()
    session._client = old
    session._active_model = WHISPER_TRANSCRIBE_MODEL
    session._provider_session_started_at = time.time() - 4000  # > 55 min
    session._provider_session_turns = 0
    created = []

    async def fake_create():
        c = FakeClient()
        created.append(c)
        return c

    session._create_client = fake_create
    async with session._transport_lock:
        ok = await session._init_or_reuse_client(turn_id=1)
    assert ok is True
    assert len(created) == 1, "past max-age session must be rotated (rebuilt)"
    assert old.closed is True
    assert created[0].connected is True
    assert session._client is created[0]


@pytest.mark.asyncio
async def test_session_past_max_turns_is_rotated():
    session = make_session(provider_session_max_turns=8)
    old = FakeClient()
    session._client = old
    session._active_model = WHISPER_TRANSCRIBE_MODEL
    session._provider_session_started_at = time.time()
    session._provider_session_turns = 8  # at cap
    created = []

    async def fake_create():
        c = FakeClient()
        created.append(c)
        return c

    session._create_client = fake_create
    async with session._transport_lock:
        ok = await session._init_or_reuse_client(turn_id=1)
    assert ok is True
    assert len(created) == 1
    assert old.closed is True


# ── audio-ingest conservation (task 0446: occasional first-utterance loss) ────
# The menubar buffers audio until the server acks `connected` (0436 S2); the
# server must give the same buffer-never-drop guarantee on the provider leg.
# These drive _handle_audio_bytes directly (previously untested) across every
# not-forwardable state and assert the frame is CONSERVED, not silently dropped.

_FRAME = b"\x11\x22" * 480  # 960B == one ~20ms int16 frame @24kHz (rate passthrough)


def _distinct_frame(seed: int) -> bytes:
    return bytes([seed]) * len(_FRAME)


def _stub_reconnect_to(session, new_client):
    async def _init(turn_id=None, **kwargs):
        session._client = new_client
        session._provider_generation += 1
        session._provider_audio_bytes = 0
        session._begin_provider_epoch_locked()
        session._register_client_handlers(new_client)
        return True

    session._init_or_reuse_client = _init
    session._RECONNECT_BACKOFF_BASE_SEC = 0


@pytest.mark.asyncio
async def test_audio_before_provider_ready_is_buffered_not_dropped():
    """N frames received before the provider session is ready are all conserved
    (the 'frames before connection ready' regression)."""
    s = make_session()
    s._is_recording = True
    s._openai_ready.clear()      # session not established yet
    s._client = None
    for _ in range(5):
        await s._handle_audio_bytes(_FRAME)
    assert len(s._turn_audio_chunks) == 5, "early frames must be cached, not dropped"
    assert s._turn_audio_bytes == 5 * len(_FRAME)
    assert s._provider_audio_bytes == 0


@pytest.mark.asyncio
async def test_audio_ready_and_recording_is_cached_then_forwarded():
    s = make_session()
    fc = FakeClient()
    sent = []

    async def _send(chunk):
        sent.append(chunk)
        return True

    fc.send_audio = _send
    s._client = fc
    s._is_recording = True
    s._openai_ready.set()
    await s._handle_audio_bytes(_FRAME)
    assert len(sent) == 1
    assert list(s._turn_audio_chunks) == [_FRAME]
    assert s._provider_audio_bytes == len(_FRAME)


@pytest.mark.asyncio
async def test_audio_when_provider_ws_closed_is_buffered_not_dropped():
    """openai_ready but the provider WS closed mid-turn: send_audio reports False,
    so the server conserves it and immediately replays through a replacement."""
    s = make_session()
    fc = FakeClient()
    fc._open = False             # WS not open -> send_audio returns False
    s._client = fc
    s._is_recording = True
    s._openai_ready.set()
    replacement = FakeClient()
    _stub_reconnect_to(s, replacement)

    await s._handle_audio_bytes(_FRAME)

    assert list(s._turn_audio_chunks) == [_FRAME]
    assert replacement.sent_audio == [_FRAME]
    assert s._provider_audio_bytes == len(_FRAME)
    assert s._transport_state == TransportState.RECORDING
    assert [
        p["type"] for p in s._websocket.sent
        if p.get("type") in {"recovery_started", "recovery_failed"}
    ] == ["recovery_started"]
    assert [
        p["status"] for p in s._websocket.sent
        if p.get("type") == "status" and p.get("status") == "connected"
    ] == ["connected"]


@pytest.mark.asyncio
async def test_audio_after_recording_window_is_not_added_to_turn_cache():
    """A post-stop/inter-turn frame must not mutate the finalized turn cache."""
    s = make_session()
    s._client = FakeClient()
    s._is_recording = False
    s._openai_ready.set()
    await s._handle_audio_bytes(_FRAME)
    assert s._turn_audio_chunks == []
    assert s._turn_audio_bytes == 0


@pytest.mark.asyncio
async def test_audio_send_exception_buffers_for_replay():
    """A send that raises (connection drop mid-send) buffers the frame so
    immediate auto-recovery replays it, rather than waiting for a callback."""
    s = make_session()
    fc = FakeClient()

    async def _boom(chunk):
        raise RuntimeError("send boom")

    fc.send_audio = _boom
    s._client = fc
    s._is_recording = True
    s._openai_ready.set()
    replacement = FakeClient()
    _stub_reconnect_to(s, replacement)

    await s._handle_audio_bytes(_FRAME)

    assert list(s._turn_audio_chunks) == [_FRAME]
    assert replacement.sent_audio == [_FRAME]
    assert s._provider_audio_bytes == len(_FRAME)
    assert s._transport_state == TransportState.RECORDING


@pytest.mark.asyncio
async def test_buffered_early_frames_flush_to_provider_on_same_turn_restart():
    """End-to-end conservation: frames buffered before the session was ready are
    actually replayed to the provider when the turn (re)starts, so the onset
    reaches OpenAI instead of being stranded."""
    s = make_session()  # _active_turn_id=900, not finalized
    s._openai_ready.clear()
    s._client = None
    for _ in range(3):
        await s._handle_audio_bytes(_FRAME)
    assert len(s._turn_audio_chunks) == 3

    fake = FakeClient()
    flushed = []

    async def _send(chunk):
        flushed.append(chunk)
        return True

    fake.send_audio = _send
    s._client = fake
    s._active_model = WHISPER_TRANSCRIBE_MODEL
    s._provider_session_started_at = time.time()
    s._openai_ready.set()  # reuse-path precondition

    # Same-turn restart (turn_id matches, not finalized) preserves + flushes.
    await s._handle_start_recording({"turn_id": 900, "input_sample_rate": 24000})
    assert len(flushed) == 3, "conserved onset frames must be delivered to the provider"
    assert len(s._turn_audio_chunks) == 3, "turn cache remains replayable until finalize"
    assert s._provider_audio_bytes == s._turn_audio_bytes


# ── provider reconnect replay (W10: one complete turn cache) ──────────────────


@pytest.mark.asyncio
async def test_reconnect_replays_cached_unconfirmed_audio_in_order():
    """Audio that send_audio accepted before a mid-turn provider disconnect is
    replayed into the new provider session before any later pending chunks."""
    s = make_session()
    fc1 = FakeClient()
    s._client = fc1
    chunks = [_distinct_frame(1), _distinct_frame(2), _distinct_frame(3)]

    for chunk in chunks:
        await s._handle_audio_bytes(chunk)

    assert fc1.sent_audio == chunks
    assert list(s._turn_audio_chunks) == chunks

    fc2 = FakeClient()
    _stub_reconnect_to(s, fc2)
    await s._on_provider_disconnect()

    assert fc2.sent_audio == chunks
    assert list(s._turn_audio_chunks) == chunks
    assert s._provider_audio_bytes == s._turn_audio_bytes


@pytest.mark.asyncio
async def test_replay_turn_cache_rejects_unlocked_call():
    s = make_session()
    s._client = FakeClient()
    with pytest.raises(RuntimeError, match="_transport_lock"):
        await s._replay_turn_audio_locked()


@pytest.mark.asyncio
async def test_segment_cache_boundary_does_not_evict_replayable_prefix(caplog):
    bound = len(_FRAME) * 2 - 1
    s = make_session(max_turn_audio_bytes=bound)
    frames = [_distinct_frame(7), _distinct_frame(8), _distinct_frame(9)]

    caplog.set_level(logging.INFO, logger=realtime_server_module.__name__)
    async with s._transport_lock:
        assert s._append_turn_audio_locked(frames[0]) is True
        assert s._append_turn_audio_locked(frames[1]) is False
    assert list(s._turn_audio_chunks) == [frames[0]]
    assert s._turn_audio_bytes == len(frames[0])
    assert "Segment audio cache limit reached" in caplog.text


@pytest.mark.asyncio
async def test_reset_turn_state_clears_turn_audio_cache():
    s = make_session()
    await s._handle_audio_bytes(_distinct_frame(10))
    assert s._turn_audio_chunks

    async with s._transport_lock:
        s._reset_turn_state(901)

    assert s._turn_audio_chunks == []
    assert s._turn_audio_bytes == 0
    assert s._provider_audio_bytes == 0


def test_reset_turn_state_rejects_unlocked_runtime_mutation():
    s = make_session()
    with pytest.raises(RuntimeError, match="_transport_lock"):
        s._reset_turn_state(901)


@pytest.mark.asyncio
async def test_finalized_turn_audio_does_not_replay_after_next_turn_reconnect():
    s = make_session()
    s._client = FakeClient()
    old_frame = _distinct_frame(11)
    await s._handle_audio_bytes(old_frame)
    assert list(s._turn_audio_chunks) == [old_frame]

    await s._finalize_turn("transcription.completed")
    assert s._turn_audio_chunks == []
    assert s._turn_audio_bytes == 0

    async with s._transport_lock:
        s._reset_turn_state(901)
    s._is_recording = True
    s._openai_ready.set()
    s._client = FakeClient()
    fc2 = FakeClient()
    _stub_reconnect_to(s, fc2)

    await s._on_provider_disconnect()

    assert fc2.sent_audio == []
    assert s._turn_audio_chunks == []


# ── W2c review-R1 deterministic counterexamples (RED before serialization) ──


@pytest.mark.asyncio
async def test_recovery_barrier_prevents_new_frame_before_replay():
    """A new frame waits behind replay and cannot mutate the cache lock-free."""
    s = make_session()
    old_frame = _distinct_frame(21)
    new_frame = _distinct_frame(22)
    s._client = FakeClient()
    await s._handle_audio_bytes(old_frame)

    replacement = FakeClient()
    transport_created = asyncio.Event()
    allow_recovery = asyncio.Event()

    async def _init(turn_id=None, **kwargs):
        s._client = replacement
        s._openai_ready.set()  # old implementation publishes too early
        transport_created.set()
        await allow_recovery.wait()
        return True

    s._init_or_reuse_client = _init
    s._RECONNECT_BACKOFF_BASE_SEC = 0
    recovery = asyncio.create_task(s._on_provider_disconnect())
    await asyncio.wait_for(transport_created.wait(), 1)

    new_audio = asyncio.create_task(s._handle_audio_bytes(new_frame))
    await asyncio.sleep(0)
    assert not new_audio.done()
    assert list(s._turn_audio_chunks) == [old_frame]
    allow_recovery.set()
    await asyncio.wait_for(asyncio.gather(recovery, new_audio), 1)

    assert replacement.sent_audio == [old_frame, new_frame]
    assert list(s._turn_audio_chunks) == [old_frame, new_frame]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["false", "exception"])
async def test_replay_failure_keeps_full_cache_and_next_provider_restarts_at_zero(failure):
    s = make_session()
    frames = [_distinct_frame(31), _distinct_frame(32), _distinct_frame(33)]
    async with s._transport_lock:
        for frame in frames:
            assert s._append_turn_audio_locked(frame)
    client = FakeClient()
    calls = 0

    async def _send(chunk):
        nonlocal calls
        calls += 1
        if calls == 2:
            if failure == "false":
                return False
            raise RuntimeError("injected replay failure")
        client.sent_audio.append(chunk)
        return True

    client.send_audio = _send
    s._client = client

    async with s._transport_lock:
        assert await s._replay_turn_audio_locked() is False

    assert calls == 2
    assert list(s._turn_audio_chunks) == frames
    assert not s._openai_ready.is_set()

    replacement = FakeClient()
    s._client = replacement
    async with s._transport_lock:
        assert await s._replay_turn_audio_locked() is True
    assert replacement.sent_audio == frames
    assert s._provider_audio_bytes == s._turn_audio_bytes


@pytest.mark.asyncio
async def test_finalize_waits_for_replay_owner_before_clearing_turn_cache():
    """R2 F1 counterexample: finalize cannot clear a cache under active replay."""
    s = make_session()
    frames = [_distinct_frame(51), _distinct_frame(52)]
    async with s._transport_lock:
        for frame in frames:
            assert s._append_turn_audio_locked(frame)

    client = FakeClient()
    send_entered = asyncio.Event()
    allow_send = asyncio.Event()

    async def _send(chunk):
        send_entered.set()
        await allow_send.wait()
        client.sent_audio.append(chunk)
        return True

    client.send_audio = _send
    s._client = client

    async def _replay():
        async with s._transport_lock:
            return await s._replay_turn_audio_locked()

    replay = asyncio.create_task(_replay())
    await asyncio.wait_for(send_entered.wait(), 1)
    finalize = asyncio.create_task(s._finalize_turn("error"))
    await asyncio.sleep(0)
    assert not finalize.done()
    assert list(s._turn_audio_chunks) == frames

    allow_send.set()
    assert await asyncio.wait_for(replay, 1) is True
    await asyncio.wait_for(finalize, 1)
    assert client.sent_audio == frames
    assert s._turn_audio_chunks == []


@pytest.mark.asyncio
async def test_transport_owner_guard_rejects_lockless_mutator_from_other_task():
    """R2 F1: globally locked is insufficient; the caller must own the lock."""
    s = make_session()
    first = _distinct_frame(53)
    intruder = _distinct_frame(54)
    async with s._transport_lock:
        assert s._append_turn_audio_locked(first)

    client = FakeClient()
    send_entered = asyncio.Event()
    allow_send = asyncio.Event()

    async def _send(chunk):
        send_entered.set()
        await allow_send.wait()
        client.sent_audio.append(chunk)
        return True

    client.send_audio = _send
    s._client = client

    async def _replay_as_owner():
        async with s._transport_lock:
            return await s._replay_turn_audio_locked()

    replay = asyncio.create_task(_replay_as_owner())
    await asyncio.wait_for(send_entered.wait(), 1)
    with pytest.raises(RuntimeError, match="_transport_lock"):
        s._append_turn_audio_locked(intruder)
    assert list(s._turn_audio_chunks) == [first]

    allow_send.set()
    assert await asyncio.wait_for(replay, 1) is True
    assert client.sent_audio == [first]


@pytest.mark.asyncio
async def test_stop_during_replay_commits_only_after_recovery_drain():
    """Replay and commit share one serial owner; no append may follow commit."""
    s = make_session()
    frames = [_distinct_frame(seed) for seed in range(41, 46)]
    async with s._transport_lock:
        for frame in frames:
            assert s._append_turn_audio_locked(frame)

    replacement = FakeClient()
    first_send_entered = asyncio.Event()
    allow_send = asyncio.Event()
    events = []

    async def _send(chunk):
        events.append(("send_enter", chunk[0]))
        if not first_send_entered.is_set():
            first_send_entered.set()
            await allow_send.wait()
        replacement.sent_audio.append(chunk)
        events.append(("send_done", chunk[0]))
        return True

    async def _commit():
        events.append(("commit", None))
        s._turn_done.set()

    replacement.send_audio = _send
    replacement.commit_audio = _commit
    _stub_reconnect_to(s, replacement)

    recovery = asyncio.create_task(s._on_provider_disconnect())
    await asyncio.wait_for(first_send_entered.wait(), 1)
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.sleep(0)
    assert ("commit", None) not in events

    allow_send.set()
    await asyncio.wait_for(asyncio.gather(recovery, stop), 1)
    assert [event for event, _ in events] == ["send_enter", "send_done"] * 5 + ["commit"]
    assert replacement.sent_audio == frames


@pytest.mark.asyncio
async def test_stale_disconnect_rechecks_generation_after_waiting_for_owner():
    """A gen-N callback queued behind recovery cannot tear down committed gen N+1."""
    s = make_session()
    original = FakeClient()
    s._client = original
    s._provider_generation = 7
    frames = [_distinct_frame(seed) for seed in range(55, 60)]
    for frame in frames:
        await s._handle_audio_bytes(frame)

    replacement = FakeClient()
    commits = 0

    async def _commit():
        nonlocal commits
        commits += 1

    replacement.commit_audio = _commit
    _stub_reconnect_to(s, replacement)

    async with s._transport_lock:
        stale_callback = asyncio.create_task(s._on_provider_disconnect(7))
        await asyncio.sleep(0)
        assert not stale_callback.done()
        s._transport_state = TransportState.STOP_REQUESTED
        s._is_recording = False
        assert await s._recover_transport_locked(s._turn_generation) is True

    await asyncio.wait_for(stale_callback, 1)
    assert s._provider_generation == 8
    assert s._client is replacement
    assert replacement.closed is False
    assert replacement.sent_audio == frames
    assert commits == 1
    assert s._commit_key == s._current_commit_key()
    assert s._transport_state == TransportState.COMMITTING


@pytest.mark.asyncio
@pytest.mark.parametrize("acknowledged_before_disconnect", [False, True])
async def test_disconnect_during_finalize_replays_and_recommits_new_provider_epoch(
    acknowledged_before_disconnect,
):
    """Commit send/ACK is not terminal; disconnect preserves retry state."""
    s = make_session()
    original = FakeClient()
    original_commits = 0

    async def _original_commit():
        nonlocal original_commits
        original_commits += 1

    original.commit_audio = _original_commit
    s._client = original
    s._provider_generation = 10
    frames = [_distinct_frame(seed) for seed in range(60, 65)]
    for frame in frames:
        await s._handle_audio_bytes(frame)

    async with s._transport_lock:
        assert await s._commit_transport_locked() is True
    assert original_commits == 1
    if acknowledged_before_disconnect:
        await s._on_input_buffer_committed({"item_id": "old-item"})
        assert s._transport_state == TransportState.COMMIT_ACKED

    replacement = FakeClient()
    replacement_commits = 0

    async def _replacement_commit():
        nonlocal replacement_commits
        replacement_commits += 1

    replacement.commit_audio = _replacement_commit
    _stub_reconnect_to(s, replacement)
    await s._on_provider_disconnect(10)

    assert s._provider_generation == 11
    assert replacement.sent_audio == frames
    assert replacement_commits == 1
    assert s._commit_key == s._current_commit_key()
    assert s._transport_state == TransportState.COMMITTING
    if acknowledged_before_disconnect:
        assert "old-item" in s._retired_item_ids

    await replacement.handlers["input_audio_buffer.committed"](
        {"item_id": "replacement-item"}
    )
    await replacement.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "replacement-item", "transcript": "完整结果"})
    await drain_finalize(s)
    assert s._finalized is True
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_commit_send_failure_preserves_cache_recovers_and_retries_finalize():
    s = make_session()
    original = ExplodingCommitClient()
    s._client = original
    s._provider_generation = 30
    frames = [_distinct_frame(seed) for seed in range(66, 71)]
    for frame in frames:
        await s._handle_audio_bytes(frame)

    replacement = FakeClient()
    replacement_commit = asyncio.Event()

    async def _commit():
        replacement_commit.set()

    replacement.commit_audio = _commit
    _stub_reconnect_to(s, replacement)

    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(replacement_commit.wait(), 1)
    assert list(s._turn_audio_chunks) == frames
    assert replacement.sent_audio == frames
    assert s._commit_key == s._current_commit_key()

    await replacement.handlers["input_audio_buffer.committed"]({"item_id": "retry-item"})
    await replacement.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "retry-item", "transcript": "重试后的完整结果"})
    await drain_finalize(s)
    await asyncio.wait_for(stop, 1)
    assert s._finalized is True
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_finalize_timeout_preserves_cache_replays_and_retries_once():
    s = make_session(transcription_finalize_timeout_sec=0.02)
    original = FakeClient()
    original_commits = 0

    async def _original_commit():
        nonlocal original_commits
        original_commits += 1

    original.commit_audio = _original_commit
    s._client = original
    s._provider_generation = 35
    frames = [_distinct_frame(seed) for seed in range(77, 82)]
    for frame in frames:
        await s._handle_audio_bytes(frame)

    replacement = FakeClient()
    replacement_commits = 0

    async def _replacement_commit():
        nonlocal replacement_commits
        replacement_commits += 1

        async def _finish():
            await replacement.handlers["input_audio_buffer.committed"](
                {"item_id": "timeout-retry-item"}
            )
            await replacement.handlers[
                "conversation.item.input_audio_transcription.completed"
            ]({"item_id": "timeout-retry-item", "transcript": "第二次完成"})

        asyncio.create_task(_finish())

    replacement.commit_audio = _replacement_commit
    _stub_reconnect_to(s, replacement)

    await asyncio.wait_for(
        s._handle_stop_recording({"turn_id": 900}), timeout=1
    )

    assert original_commits == 1
    assert replacement_commits == 1
    assert replacement.sent_audio == frames
    assert s._finalize_commit_attempts == 2
    assert s._finalized is True
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_terminal_before_commit_returns_retries_without_losing_final_signal():
    s = make_session(transcription_finalize_timeout_sec=0.2)
    original = FakeClient()
    s._client = original
    s._provider_generation = 36
    frames = [_distinct_frame(seed) for seed in range(82, 87)]
    for frame in frames:
        await s._handle_audio_bytes(frame)

    async def _terminal_during_commit():
        await s._on_input_buffer_committed({"item_id": "first-attempt-item"})
        await s._on_transcription_failed({
            "item_id": "first-attempt-item",
            "error": {"code": "synthetic"},
        })

    original.commit_audio = _terminal_during_commit
    replacement = FakeClient()

    async def _replacement_commit():
        async def _finish():
            await replacement.handlers["input_audio_buffer.committed"](
                {"item_id": "second-attempt-item"}
            )
            await replacement.handlers[
                "conversation.item.input_audio_transcription.completed"
            ]({"item_id": "second-attempt-item", "transcript": "恢复后的最终结果"})

        asyncio.create_task(_finish())

    replacement.commit_audio = _replacement_commit
    _stub_reconnect_to(s, replacement)

    await asyncio.wait_for(
        s._handle_stop_recording({"turn_id": 900}), timeout=1
    )

    assert replacement.sent_audio == frames
    assert s._finalize_commit_attempts == 2
    assert s._transcript_completed is True
    assert s._finalized is True
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_empty_finalize_result_retries_once_then_finalizes_empty():
    s = make_session(transcription_finalize_timeout_sec=0.2)
    original = FakeClient()
    s._client = original
    s._provider_generation = 44
    frames = [_distinct_frame(seed) for seed in range(98, 103)]
    for frame in frames:
        await s._handle_audio_bytes(frame)

    async def _first_empty():
        await s._on_input_buffer_committed({"item_id": "first-empty"})
        await s._on_transcription_completed(
            {"item_id": "first-empty", "transcript": ""}
        )

    original.commit_audio = _first_empty
    replacement = FakeClient()

    async def _second_empty():
        async def _finish():
            await replacement.handlers["input_audio_buffer.committed"](
                {"item_id": "second-empty"}
            )
            await replacement.handlers[
                "conversation.item.input_audio_transcription.completed"
            ]({"item_id": "second-empty", "transcript": ""})

        asyncio.create_task(_finish())

    replacement.commit_audio = _second_empty
    _stub_reconnect_to(s, replacement)
    await asyncio.wait_for(
        s._handle_stop_recording({"turn_id": 900}), timeout=1
    )

    assert replacement.sent_audio == frames
    assert s._finalize_commit_attempts == 2
    assert s._normalized_final_text == ""
    assert s._finalized is True
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_provider_error_during_finalize_retries_without_exposing_error():
    s = make_session()
    old = FakeClient()
    s._client = old
    s._provider_generation = 45
    s._register_client_handlers(old)
    frames = [_distinct_frame(seed) for seed in range(103, 108)]
    for frame in frames:
        await s._handle_audio_bytes(frame)
    async with s._transport_lock:
        assert await s._commit_transport_locked() is True

    replacement = FakeClient()
    _stub_reconnect_to(s, replacement)
    await old.handlers["error"]({
        "error": {"type": "server_error", "code": "synthetic", "message": "retry"}
    })

    assert replacement.sent_audio == frames
    assert s._finalize_commit_attempts == 2
    assert s._finalized is False
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []

    await replacement.handlers["input_audio_buffer.committed"](
        {"item_id": "error-retry-item"}
    )
    await replacement.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "error-retry-item", "transcript": "错误后恢复"})
    await drain_finalize(s)
    assert s._finalized is True
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_stale_finalize_retry_task_does_not_kill_advanced_attempt():
    s = make_session()
    mark_commit_sent(s, attempts=1)

    async with s._transport_lock:
        s._schedule_finalize_retry("synthetic-first-attempt-failure")
        retry_task = s._finalize_retry_task
        await asyncio.sleep(0)
        assert retry_task is not None and not retry_task.done()
        s._finalize_commit_attempts = 2  # another owner advanced the attempt

    await asyncio.wait_for(retry_task, 1)
    assert s._finalized is False
    assert s._transport_state == TransportState.COMMITTING
    assert _idles(s) == []


@pytest.mark.asyncio
async def test_disconnect_after_second_finalize_attempt_cannot_commit_third_time():
    s = make_session()
    client = FakeClient()
    s._client = client
    s._provider_generation = 37
    frames = [_distinct_frame(seed) for seed in range(87, 92)]
    for frame in frames:
        await s._handle_audio_bytes(frame)
    mark_commit_sent(s, attempts=2)
    reconnect_calls = 0

    async def _unexpected_reconnect(*args, **kwargs):
        nonlocal reconnect_calls
        reconnect_calls += 1
        return False

    s._init_or_reuse_client = _unexpected_reconnect
    await s._on_provider_disconnect(37)

    assert reconnect_calls == 0
    assert s._finalize_commit_attempts == 2
    assert s._finalized is True
    assert s._turn_audio_chunks == []
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_completed_then_immediate_disconnect_defers_to_completed_barrier():
    s = make_session()
    old = FakeClient()
    s._client = old
    s._provider_generation = 38
    s._register_client_handlers(old)
    await old.handlers["conversation.item.input_audio_transcription.delta"](
        {"item_id": "old-item", "delta": "epoch-one "}
    )

    replacement = FakeClient()
    async with s._transport_lock:
        s._client = replacement
        s._provider_generation = 39
        s._begin_provider_epoch_locked()
        s._register_client_handlers(replacement)
    full = "epoch-one complete"
    await replacement.handlers["conversation.item.input_audio_transcription.delta"](
        {"item_id": "replacement-item", "delta": full}
    )
    mark_commit_sent(s, attempts=2)
    await replacement.handlers["input_audio_buffer.committed"](
        {"item_id": "replacement-item"}
    )
    await replacement.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "replacement-item", "transcript": full})

    await s._on_provider_disconnect(39)
    assert s._finalized is False
    await drain_finalize(s)

    assert s._normalized_final_text == full
    assert s._emitted_text == full
    replacements = [p for p in _texts(s) if p.get("isNewResponse") is True]
    assert replacements[-1]["content"] == full
    _assert_s1(s)


@pytest.mark.asyncio
async def test_full_replay_provider_epoch_does_not_duplicate_partial_candidate():
    s = make_session()
    old = FakeClient()
    s._client = old
    s._provider_generation = 40
    s._register_client_handlers(old)
    await old.handlers["conversation.item.input_audio_transcription.delta"](
        {"item_id": "old-item", "delta": "epoch-one "}
    )

    replacement = FakeClient()
    async with s._transport_lock:
        s._client = replacement
        s._provider_generation = 41
        s._begin_provider_epoch_locked()
        s._register_client_handlers(replacement)

    full = "epoch-one complete"
    await replacement.handlers["conversation.item.input_audio_transcription.delta"](
        {"item_id": "replacement-item", "delta": full}
    )
    await replacement.handlers["input_audio_buffer.committed"](
        {"item_id": "replacement-item"}
    )
    await replacement.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "replacement-item", "transcript": full})
    await drain_finalize(s)

    assert s._transcript_text == full
    assert s._normalized_final_text == full
    assert s._emitted_text == full
    replacements = [p for p in _texts(s) if p.get("isNewResponse") is True]
    assert replacements[-1]["content"] == full
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_failed_replacement_epoch_collapses_streamed_duplicate_before_idle():
    s = make_session()
    old = FakeClient()
    s._client = old
    s._provider_generation = 42
    s._register_client_handlers(old)
    await old.handlers["conversation.item.input_audio_transcription.delta"](
        {"item_id": "old-item", "delta": "epoch-one "}
    )

    replacement = FakeClient()
    async with s._transport_lock:
        s._client = replacement
        s._provider_generation = 43
        s._begin_provider_epoch_locked()
        s._register_client_handlers(replacement)

    full = "epoch-one complete"
    await replacement.handlers["conversation.item.input_audio_transcription.delta"](
        {"item_id": "replacement-item", "delta": full}
    )
    await replacement.handlers["input_audio_buffer.committed"](
        {"item_id": "replacement-item"}
    )
    await replacement.handlers[
        "conversation.item.input_audio_transcription.failed"
    ]({"item_id": "replacement-item", "error": {"code": "synthetic"}})
    await drain_finalize(s)

    assert s._transcript_text == full
    assert s._emitted_text == full
    replacements = [p for p in _texts(s) if p.get("isNewResponse") is True]
    assert replacements[-1]["content"] == full
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_replacement_epoch_tentative_rebind_keeps_prior_partial_fallback():
    s = make_session()
    s._client = FakeClient()
    await s._on_transcription_delta({"item_id": "old-item", "delta": "prior fallback"})
    async with s._transport_lock:
        s._begin_provider_epoch_locked()

    await s._on_transcription_delta({"item_id": "tentative", "delta": "wrong"})
    await s._on_input_buffer_committed({"item_id": "real-item"})

    assert s._best_partial_text == "prior fallback"
    assert s._provider_epoch_text == ""
    assert s._transcript_text == "prior fallback"


@pytest.mark.asyncio
async def test_old_provider_error_waiting_for_owner_cannot_finalize_replacement():
    s = make_session()
    old = FakeClient()
    s._client = old
    s._provider_generation = 50
    s._register_client_handlers(old)
    frames = [_distinct_frame(seed) for seed in range(72, 77)]
    for frame in frames:
        await s._handle_audio_bytes(frame)

    replacement = FakeClient()
    _stub_reconnect_to(s, replacement)
    async with s._transport_lock:
        stale_error = asyncio.create_task(old.handlers["error"]({
            "error": {"type": "server_error", "code": "old", "message": "old"}
        }))
        await asyncio.sleep(0)
        assert not stale_error.done()
        assert await s._recover_transport_locked(s._turn_generation) is True

    await asyncio.wait_for(stale_error, 1)
    assert s._provider_generation == 51
    assert s._client is replacement
    assert replacement.closed is False
    assert replacement.sent_audio == frames
    assert list(s._turn_audio_chunks) == frames
    assert s._finalized is False
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []


@pytest.mark.asyncio
async def test_stale_provider_handlers_cannot_bind_or_finalize_replacement_epoch():
    s = make_session()
    old = FakeClient()
    s._client = old
    s._provider_generation = 20
    s._register_client_handlers(old)
    bind_confirmed(s, "old-item")

    replacement = FakeClient()
    async with s._transport_lock:
        s._client = replacement
        s._provider_generation = 21
        s._begin_provider_epoch_locked()
        s._register_client_handlers(replacement)

    await old.handlers["conversation.item.input_audio_transcription.delta"](
        {"item_id": "old-item", "delta": "stale"}
    )
    await old.handlers["input_audio_buffer.committed"]({"item_id": "old-item"})
    await old.handlers["conversation.item.input_audio_transcription.completed"](
        {"item_id": "old-item", "transcript": "stale-final"}
    )

    assert s._current_item_id is None
    assert s._item_confirmed is False
    assert s._finalized is False
    assert _texts(s) == []


@pytest.mark.asyncio
async def test_stop_with_no_client_rebuilds_replays_and_commits():
    s = make_session()
    frames = [_distinct_frame(seed) for seed in range(45, 50)]
    async with s._transport_lock:
        for frame in frames:
            assert s._append_turn_audio_locked(frame)
    s._client = None
    s._openai_ready.clear()
    replacement = FakeClient()
    commits = 0

    async def _create():
        return replacement

    async def _commit():
        nonlocal commits
        commits += 1
        s._turn_done.set()

    replacement.commit_audio = _commit
    s._create_client = _create
    await s._handle_stop_recording({"turn_id": 900})

    assert replacement.sent_audio == frames
    assert commits == 1


@pytest.mark.asyncio
async def test_duplicate_stop_commits_once_per_turn_generation():
    s = make_session()
    client = FakeClient()
    commits = 0

    async def _commit():
        nonlocal commits
        commits += 1
        s._turn_done.set()

    client.commit_audio = _commit
    s._client = client
    for seed in range(71, 76):
        await s._handle_audio_bytes(_distinct_frame(seed))
    await asyncio.gather(
        s._handle_stop_recording({"turn_id": 900}),
        s._handle_stop_recording({"turn_id": 900}),
    )

    assert commits == 1


@pytest.mark.asyncio
async def test_same_turn_retry_reuses_provider_without_clearing_ledger():
    s = make_session()
    client = FakeClient()
    s._client = client
    s._active_model = WHISPER_TRANSCRIBE_MODEL
    s._provider_session_started_at = time.time()
    frame = _distinct_frame(51)
    await s._handle_audio_bytes(frame)

    await s._handle_start_recording({"turn_id": 900, "input_sample_rate": 24000})

    assert list(s._turn_audio_chunks) == [frame]
    assert s._provider_audio_bytes == len(frame)
    assert s._client is client


@pytest.mark.asyncio
async def test_same_turn_retry_rebuilds_provider_and_replays_ledger_before_ready():
    s = make_session()
    old = FakeClient()
    old._open = False
    s._client = old
    s._active_model = WHISPER_TRANSCRIBE_MODEL
    s._provider_session_started_at = time.time()
    s._transport_state = TransportState.CONNECTING
    s._openai_ready.clear()
    frame = _distinct_frame(52)
    await s._handle_audio_bytes(frame)
    replacement = FakeClient()

    async def _create():
        return replacement

    s._create_client = _create
    await s._handle_start_recording({"turn_id": 900, "input_sample_rate": 24000})

    assert replacement.sent_audio == [frame]
    connected = [
        payload for payload in s._websocket.sent
        if payload.get("type") == "status" and payload.get("status") == "connected"
    ]
    assert len(connected) == 1


# ── W10 grapeot transport contract (RED before single-cache refactor) ─────────


@pytest.mark.asyncio
async def test_w10_turn_cache_is_appended_before_provider_send():
    s = make_session()
    client = FakeClient()
    send_entered = asyncio.Event()
    allow_send = asyncio.Event()

    async def _send(chunk):
        assert s._transport_lock.locked()
        assert list(s._turn_audio_chunks) == [chunk]
        assert s._turn_audio_bytes == len(chunk)
        send_entered.set()
        await allow_send.wait()
        client.sent_audio.append(chunk)
        return True

    client.send_audio = _send
    s._client = client
    audio_task = asyncio.create_task(s._handle_audio_bytes(_distinct_frame(61)))
    await asyncio.wait_for(send_entered.wait(), 1)
    allow_send.set()
    await asyncio.wait_for(audio_task, 1)


@pytest.mark.asyncio
async def test_w10_reconnect_replays_entire_turn_cache_from_offset_zero():
    s = make_session()
    original = FakeClient()
    s._client = original
    frames = [_distinct_frame(62), _distinct_frame(63), _distinct_frame(64)]
    for frame in frames:
        await s._handle_audio_bytes(frame)

    replacement = FakeClient()
    _stub_reconnect_to(s, replacement)
    await s._on_provider_disconnect()

    assert original.sent_audio == frames
    assert replacement.sent_audio == frames
    assert list(s._turn_audio_chunks) == frames
    assert s._turn_audio_bytes == sum(map(len, frames))
    assert s._provider_audio_bytes == s._turn_audio_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_audio_bytes", [0, 4_800])
async def test_w10_finalize_resyncs_full_cache_before_commit_when_provider_under_synced(
    provider_audio_bytes,
):
    s = make_session()
    old_client = FakeClient()
    s._client = old_client
    frames = [_distinct_frame(seed) for seed in range(65, 71)]  # 5760B
    for frame in frames:
        await s._handle_audio_bytes(frame)

    # Model a replacement/current session whose local successful-send count says
    # the turn cache has not yet been synchronized. An open socket alone must not
    # authorize commit.
    s._provider_audio_bytes = provider_audio_bytes
    replacement = FakeClient()
    events = []

    async def _create():
        return replacement

    async def _send(chunk):
        events.append(("send", chunk[0]))
        replacement.sent_audio.append(chunk)
        return True

    async def _commit():
        events.append(("commit", None))
        s._turn_done.set()

    replacement.send_audio = _send
    replacement.commit_audio = _commit
    s._create_client = _create
    s._RECONNECT_BACKOFF_BASE_SEC = 0

    await s._handle_stop_recording({"turn_id": 900})

    assert old_client.closed is True
    assert replacement.sent_audio == frames
    assert events == [("send", frame[0]) for frame in frames] + [("commit", None)]


@pytest.mark.asyncio
async def test_w11_resync_then_completed_only_finalizes_and_keeps_w6_fallback():
    """gpt-4o-transcribe may finalize from completed without any delta."""

    s = make_session()
    old_client = FakeClient()
    s._client = old_client
    frames = [_distinct_frame(seed) for seed in range(93, 99)]
    for frame in frames:
        await s._handle_audio_bytes(frame)
    s._provider_audio_bytes = 0

    replacement = FakeClient()
    commit_sent = asyncio.Event()

    async def _create():
        return replacement

    async def _commit():
        commit_sent.set()

    replacement.commit_audio = _commit
    s._create_client = _create
    s._RECONNECT_BACKOFF_BASE_SEC = 0

    stop_task = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(commit_sent.wait(), 1)
    item_id = "gpt4o-completed-only"
    await s._on_input_buffer_committed({"item_id": item_id})
    await s._on_transcription_completed(
        {"item_id": item_id, "transcript": "缩写是 SOT"}
    )
    await asyncio.wait_for(stop_task, 2)

    assert replacement.sent_audio == frames
    texts = _texts(s)
    assert [payload["content"] for payload in texts] == ["缩写是 SSOT"]
    assert texts[0]["isNewResponse"] is True
    assert len(_idles(s)) == 1
    _assert_s1(s)


@pytest.mark.asyncio
async def test_w11_digital_silence_finalizes_without_provider_commit():
    s = make_session()
    client = FakeClient()
    commits = 0

    async def _commit():
        nonlocal commits
        commits += 1

    client.commit_audio = _commit
    s._client = client
    for _ in range(6):
        await s._handle_audio_bytes(b"\0" * len(_FRAME))

    await s._handle_stop_recording({"turn_id": 900})

    assert commits == 0
    assert s._finalized is True
    assert _texts(s) == []
    assert len(_idles(s)) == 1
    assert client.closed is False
    _assert_s1(s)


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_bytes, expected_commits", [(4_798, 0), (4_800, 1)])
async def test_w10_commit_has_local_100ms_minimum_guard(audio_bytes, expected_commits):
    s = make_session()
    client = FakeClient()
    commits = 0

    async def _commit():
        nonlocal commits
        commits += 1

    client.commit_audio = _commit
    s._client = client
    s._turn_audio_chunks = [b"\1" * audio_bytes]
    s._turn_audio_bytes = audio_bytes
    s._provider_audio_bytes = audio_bytes

    async with s._transport_lock:
        result = await s._commit_transport_locked()

    assert s._MIN_COMMIT_AUDIO_BYTES == 4_800
    assert commits == expected_commits
    assert result is (expected_commits == 1)


@pytest.mark.asyncio
async def test_w10_non_finalize_buffer_too_small_is_swallowed():
    s = make_session()
    await s._on_error({"error": {"message": "Input Audio BUFFER TOO SMALL to commit"}})

    assert s._finalized is False
    assert s._transport_state == TransportState.RECORDING
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    assert _idles(s) == []


@pytest.mark.asyncio
async def test_w10_buffer_too_small_during_finalize_remains_terminal():
    s = make_session()
    s._transport_state = TransportState.STOP_REQUESTED
    await s._on_error({"error": {"message": "input audio buffer too small"}})

    assert s._finalized is True
    assert len([p for p in s._websocket.sent if p.get("type") == "error"]) == 1
    assert len(_idles(s)) == 1


@pytest.mark.asyncio
async def test_w10_recovery_uses_five_attempts_and_grapeot_backoff(monkeypatch):
    s = make_session()
    s._client = None
    attempts = 0
    delays = []

    async def _init(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return False

    async def _sleep(delay):
        delays.append(delay)

    s._init_or_reuse_client = _init
    monkeypatch.setattr(realtime_server_module.asyncio, "sleep", _sleep)
    async with s._transport_lock:
        recovered = await s._recover_transport_locked(s._turn_generation)

    assert recovered is False
    assert attempts == 5
    assert delays == [0.3, 0.6, 1.2, 2.4]
    recovery_events = [
        p for p in s._websocket.sent
        if p.get("type") in {"recovery_started", "recovery_failed"}
    ]
    assert [p["type"] for p in recovery_events] == [
        "recovery_started", "recovery_failed",
    ]
    assert recovery_events[-1]["attempts"] == 5


@pytest.mark.asyncio
async def test_turn_cache_limit_commits_then_rebuilds_without_losing_either_segment():
    """The production regression: a boundary commits the prefix, retains the
    crossing frame, and only the later user stop ends the surrounding turn."""

    frames = [_distinct_frame(seed) for seed in range(1, 101)]
    segment_bytes = 50 * len(_FRAME)  # 1s at PCM16/24k; valid env-scale minimum
    s = make_session(max_turn_audio_bytes=segment_bytes)
    first = FakeClient()
    second = FakeClient()
    first_committed = asyncio.Event()
    second_committed = asyncio.Event()

    async def _commit_first():
        first_committed.set()

    async def _commit_second():
        second_committed.set()

    first.commit_audio = _commit_first
    second.commit_audio = _commit_second
    s._client = first
    s._provider_generation = 70
    s._register_client_handlers(first)
    _stub_reconnect_to(s, second)

    for frame in frames[:50]:
        await s._handle_audio_bytes(frame)
    await s._handle_audio_bytes(frames[50])  # crossing frame belongs to segment 2
    await asyncio.wait_for(first_committed.wait(), 1)

    assert first.sent_audio == frames[:50]
    assert s._rollover_tail_chunks == [frames[50]]
    assert s._finalized is False
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    assert _idles(s) == []

    # Audio keeps arriving while segment A's terminal is outstanding. It must be
    # retained in order and replayed with the original crossing frame.
    for frame in frames[51:55]:
        await s._handle_audio_bytes(frame)
    await first.handlers["input_audio_buffer.committed"]({"item_id": "segment-A"})
    await first.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "segment-A", "transcript": "前段内容"})
    rollover_task = s._segment_rollover_task
    assert rollover_task is not None
    await asyncio.wait_for(asyncio.shield(rollover_task), 1)

    assert s._finalized is False
    assert s._transport_state == TransportState.RECORDING
    assert _client_text_view(s) == "前段内容"
    assert second.sent_audio == frames[50:55]
    assert _idles(s) == []
    assert [
        payload for payload in s._websocket.sent
        if payload.get("type") == "recovery_started"
        or (
            payload.get("type") == "status"
            and payload.get("status") == "connected"
        )
    ] == []

    for frame in frames[55:]:
        await s._handle_audio_bytes(frame)
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(second_committed.wait(), 1)
    await second.handlers["input_audio_buffer.committed"]({"item_id": "segment-B"})
    await second.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "segment-B", "transcript": "后段内容"})
    await asyncio.wait_for(stop, 1)

    assert second.sent_audio == frames[50:]
    assert _client_text_view(s) == "前段内容后段内容"
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.asyncio
async def test_stop_during_segment_completion_commits_buffered_tail_before_idle():
    """A user stop racing the intermediate completed event latches until the
    buffered tail has been rebuilt, committed, and transcribed."""

    prefix = [_distinct_frame(seed) for seed in range(101, 151)]
    tail = bytes([151]) * 4_800
    s = make_session(max_turn_audio_bytes=50 * len(_FRAME))
    first = FakeClient()
    second = FakeClient()
    first_committed = asyncio.Event()
    second_committed = asyncio.Event()

    async def _commit_first():
        first_committed.set()

    async def _commit_second():
        second_committed.set()

    first.commit_audio = _commit_first
    second.commit_audio = _commit_second
    s._client = first
    s._provider_generation = 80
    s._register_client_handlers(first)
    _stub_reconnect_to(s, second)

    for frame in prefix:
        await s._handle_audio_bytes(frame)
    await s._handle_audio_bytes(tail)
    await asyncio.wait_for(first_committed.wait(), 1)

    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.sleep(0)
    assert not stop.done()
    assert s._stop_requested is True
    assert _idles(s) == []

    await first.handlers["input_audio_buffer.committed"]({"item_id": "segment-C"})
    await first.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "segment-C", "transcript": "边界前"})
    await asyncio.wait_for(second_committed.wait(), 1)
    await second.handlers["input_audio_buffer.committed"]({"item_id": "segment-D"})
    await second.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "segment-D", "transcript": "边界后"})
    await asyncio.wait_for(stop, 1)

    assert first.sent_audio == prefix
    assert second.sent_audio == [tail]
    assert _client_text_view(s) == "边界前边界后"
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.asyncio
async def test_rollover_disconnect_recommits_prefix_then_replays_tail_once():
    """A disconnect before the prefix terminal replays only that prefix; after
    its completed-before-ACK is confirmed, a planned rebuild replays only tail."""

    prefix = [_distinct_frame(seed) for seed in range(152, 202)]
    tail_frames = [_distinct_frame(seed) for seed in range(202, 207)]
    s = make_session(max_turn_audio_bytes=50 * len(_FRAME))
    original = FakeClient()
    prefix_retry = FakeClient()
    tail_provider = FakeClient()
    original_committed = asyncio.Event()
    retry_committed = asyncio.Event()
    tail_committed = asyncio.Event()
    replacements = iter([prefix_retry, tail_provider])

    async def _commit_original():
        original_committed.set()

    async def _commit_retry():
        retry_committed.set()

    async def _commit_tail():
        tail_committed.set()

    async def _init_replacement(turn_id=None, **kwargs):
        replacement = next(replacements)
        s._client = replacement
        s._provider_generation += 1
        s._provider_audio_bytes = 0
        s._begin_provider_epoch_locked()
        s._register_client_handlers(replacement)
        return True

    original.commit_audio = _commit_original
    prefix_retry.commit_audio = _commit_retry
    tail_provider.commit_audio = _commit_tail
    s._client = original
    s._provider_generation = 90
    s._register_client_handlers(original)
    s._init_or_reuse_client = _init_replacement
    s._RECONNECT_BACKOFF_BASE_SEC = 0

    for frame in prefix:
        await s._handle_audio_bytes(frame)
    await s._handle_audio_bytes(tail_frames[0])
    await asyncio.wait_for(original_committed.wait(), 1)

    await s._on_provider_disconnect(90)
    await asyncio.wait_for(retry_committed.wait(), 1)
    assert original.sent_audio == prefix
    assert prefix_retry.sent_audio == prefix
    assert s._rollover_tail_chunks == [tail_frames[0]]

    # The existing item authority gate must still preserve an early terminal for
    # this intermediate (not whole-turn-final) commit.
    await prefix_retry.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "segment-retry", "transcript": "断线前"})
    assert "segment-retry" in s._pending_terminals
    await prefix_retry.handlers["input_audio_buffer.committed"](
        {"item_id": "segment-retry"}
    )
    rollover_task = s._segment_rollover_task
    assert rollover_task is not None
    await asyncio.wait_for(asyncio.shield(rollover_task), 1)

    assert tail_provider.sent_audio == [tail_frames[0]]
    assert _client_text_view(s) == "断线前"
    assert _idles(s) == []

    for frame in tail_frames[1:]:
        await s._handle_audio_bytes(frame)
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(tail_committed.wait(), 1)
    await tail_provider.handlers["input_audio_buffer.committed"](
        {"item_id": "segment-tail"}
    )
    await tail_provider.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "segment-tail", "transcript": "断线后"})
    await asyncio.wait_for(stop, 1)

    assert tail_provider.sent_audio == tail_frames
    assert _client_text_view(s) == "断线前断线后"
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.asyncio
async def test_two_consecutive_rollovers_preserve_three_segment_turn():
    """Rollover state is reusable, not a one-shot 300s extension."""

    frames = [_distinct_frame(seed) for seed in range(1, 151)]
    segment_bytes = 50 * len(_FRAME)
    s = make_session(max_turn_audio_bytes=segment_bytes)
    providers = [FakeClient(), FakeClient(), FakeClient()]
    commit_events = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
    replacements = iter(providers[1:])

    for provider, committed in zip(providers, commit_events):
        async def _commit(event=committed):
            event.set()

        provider.commit_audio = _commit

    async def _init_replacement(turn_id=None, **kwargs):
        replacement = next(replacements)
        s._client = replacement
        s._provider_generation += 1
        s._provider_audio_bytes = 0
        s._begin_provider_epoch_locked()
        s._register_client_handlers(replacement)
        return True

    s._client = providers[0]
    s._provider_generation = 100
    s._register_client_handlers(providers[0])
    s._init_or_reuse_client = _init_replacement
    s._RECONNECT_BACKOFF_BASE_SEC = 0

    for frame in frames[:51]:
        await s._handle_audio_bytes(frame)
    await asyncio.wait_for(commit_events[0].wait(), 1)
    await providers[0].handlers["input_audio_buffer.committed"](
        {"item_id": "three-A"}
    )
    await providers[0].handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "three-A", "transcript": "甲"})
    first_rollover = s._segment_rollover_task
    assert first_rollover is not None
    await asyncio.wait_for(asyncio.shield(first_rollover), 1)

    for frame in frames[51:101]:
        await s._handle_audio_bytes(frame)
    await asyncio.wait_for(commit_events[1].wait(), 1)
    await providers[1].handlers["input_audio_buffer.committed"](
        {"item_id": "three-B"}
    )
    await providers[1].handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "three-B", "transcript": "乙"})
    second_rollover = s._segment_rollover_task
    assert second_rollover is not None
    await asyncio.wait_for(asyncio.shield(second_rollover), 1)

    for frame in frames[101:]:
        await s._handle_audio_bytes(frame)
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(commit_events[2].wait(), 1)
    await providers[2].handlers["input_audio_buffer.committed"](
        {"item_id": "three-C"}
    )
    await providers[2].handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "three-C", "transcript": "丙"})
    await asyncio.wait_for(stop, 1)

    assert providers[0].sent_audio == frames[:50]
    assert providers[1].sent_audio == frames[50:100]
    assert providers[2].sent_audio == frames[100:]
    assert _client_text_view(s) == "甲乙丙"
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.parametrize(
    ("logical_seconds", "expected_segments"),
    [
        (600, 2),
        (1000, 4),
    ],
)
@pytest.mark.asyncio
async def test_scaled_600_and_1000_second_paths_preserve_every_segment_exactly_once(
    logical_seconds, expected_segments
):
    """Accelerated 10s/frame proof of repeated 300s rollover semantics.

    The production 14,400,000-byte value is a reusable 300-second segment
    boundary, not a whole-turn cap.  This keeps real ordering/commit/rebuild
    code while scaling each 300-second segment to 30 deterministic frames.
    """

    scaled_seconds_per_frame = 10
    segment_frames = 300 // scaled_seconds_per_frame
    total_frames = logical_seconds // scaled_seconds_per_frame
    assert total_frames == logical_seconds / scaled_seconds_per_frame
    assert (total_frames + segment_frames - 1) // segment_frames == expected_segments

    frames = [_distinct_frame(seed) for seed in range(1, total_frames + 1)]
    session = make_session(max_turn_audio_bytes=segment_frames * len(_FRAME))
    providers = [FakeClient() for _ in range(expected_segments)]
    commits = [asyncio.Event() for _ in providers]
    replacements = iter(providers[1:])

    for provider, committed in zip(providers, commits):
        async def _commit(event=committed):
            event.set()

        provider.commit_audio = _commit

    async def _init_replacement(turn_id=None, **kwargs):
        replacement = next(replacements)
        session._client = replacement
        session._provider_generation += 1
        session._provider_audio_bytes = 0
        session._begin_provider_epoch_locked()
        session._register_client_handlers(replacement)
        return True

    session._client = providers[0]
    session._provider_generation = 200
    session._register_client_handlers(providers[0])
    session._init_or_reuse_client = _init_replacement
    session._RECONNECT_BACKOFF_BASE_SEC = 0

    cursor = 0
    expected_markers = []
    for segment_index in range(expected_segments - 1):
        crossing_index = (segment_index + 1) * segment_frames
        while cursor <= crossing_index:
            await session._handle_audio_bytes(frames[cursor])
            cursor += 1
        await asyncio.wait_for(commits[segment_index].wait(), 1)
        item_id = f"scaled-{logical_seconds}-{segment_index}"
        marker = f"<{logical_seconds}:{segment_index}>"
        expected_markers.append(marker)
        await providers[segment_index].handlers["input_audio_buffer.committed"](
            {"item_id": item_id}
        )
        await providers[segment_index].handlers[
            "conversation.item.input_audio_transcription.completed"
        ]({"item_id": item_id, "transcript": marker})
        rollover = session._segment_rollover_task
        assert rollover is not None
        await asyncio.wait_for(asyncio.shield(rollover), 1)

    while cursor < total_frames:
        await session._handle_audio_bytes(frames[cursor])
        cursor += 1
    stop = asyncio.create_task(
        session._handle_stop_recording({"turn_id": session._active_turn_id})
    )
    final_index = expected_segments - 1
    await asyncio.wait_for(commits[final_index].wait(), 1)
    final_item = f"scaled-{logical_seconds}-{final_index}"
    final_marker = f"<{logical_seconds}:{final_index}>"
    expected_markers.append(final_marker)
    await providers[final_index].handlers["input_audio_buffer.committed"](
        {"item_id": final_item}
    )
    await providers[final_index].handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": final_item, "transcript": final_marker})
    await asyncio.wait_for(stop, 1)

    assert [chunk for provider in providers for chunk in provider.sent_audio] == frames
    full_view = _client_text_view(session)
    assert full_view == "".join(expected_markers)
    assert all(full_view.count(marker) == 1 for marker in expected_markers)
    assert len(_idles(session)) == 1
    assert [p for p in session._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(session)


@pytest.mark.asyncio
async def test_digital_silence_prefix_promotes_crossing_speech_instead_of_ending_turn():
    silence = [b"\0" * len(_FRAME) for _ in range(50)]
    speech = bytes([211]) * 4_800
    s = make_session(max_turn_audio_bytes=50 * len(_FRAME))
    first = FakeClient()
    second = FakeClient()
    first_commits = 0
    second_committed = asyncio.Event()

    async def _unexpected_first_commit():
        nonlocal first_commits
        first_commits += 1

    async def _commit_second():
        second_committed.set()

    first.commit_audio = _unexpected_first_commit
    second.commit_audio = _commit_second
    s._client = first
    s._provider_generation = 110
    s._register_client_handlers(first)
    _stub_reconnect_to(s, second)

    for frame in silence:
        await s._handle_audio_bytes(frame)
    await s._handle_audio_bytes(speech)

    assert first.sent_audio == silence
    assert first_commits == 0
    assert second.sent_audio == [speech]
    assert s._finalized is False
    assert s._transport_state == TransportState.RECORDING
    assert _idles(s) == []

    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(second_committed.wait(), 1)
    await second.handlers["input_audio_buffer.committed"](
        {"item_id": "after-silence"}
    )
    await second.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "after-silence", "transcript": "终于开口"})
    await asyncio.wait_for(stop, 1)

    assert _client_text_view(s) == "终于开口"
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.asyncio
async def test_rollover_completed_authority_ignores_immediate_provider_error():
    prefix = [_distinct_frame(seed) for seed in range(1, 51)]
    tail = bytes([212]) * 4_800
    s = make_session(max_turn_audio_bytes=50 * len(_FRAME))
    first = FakeClient()
    second = FakeClient()
    first_committed = asyncio.Event()
    second_committed = asyncio.Event()

    async def _commit_first():
        first_committed.set()

    async def _commit_second():
        second_committed.set()

    first.commit_audio = _commit_first
    second.commit_audio = _commit_second
    s._client = first
    s._provider_generation = 120
    s._register_client_handlers(first)
    _stub_reconnect_to(s, second)

    for frame in prefix:
        await s._handle_audio_bytes(frame)
    await s._handle_audio_bytes(tail)
    await asyncio.wait_for(first_committed.wait(), 1)
    await first.handlers["input_audio_buffer.committed"](
        {"item_id": "completed-before-error"}
    )
    await first.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "completed-before-error", "transcript": "权威前段"})
    await first.handlers["error"]({
        "error": {
            "type": "server_error",
            "code": "after_completed",
            "message": "must not revoke completed",
        }
    })

    rollover_task = s._segment_rollover_task
    assert rollover_task is not None
    await asyncio.wait_for(asyncio.shield(rollover_task), 1)
    assert second.sent_audio == [tail]
    assert _client_text_view(s) == "权威前段"
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []

    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(second_committed.wait(), 1)
    await second.handlers["input_audio_buffer.committed"](
        {"item_id": "completed-after-error"}
    )
    await second.handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "completed-after-error", "transcript": "权威后段"})
    await asyncio.wait_for(stop, 1)

    assert _client_text_view(s) == "权威前段权威后段"
    _assert_s1(s)


@pytest.mark.asyncio
async def test_rollover_recovery_exhaustion_sends_error_before_idle():
    s = make_session(max_turn_audio_bytes=5 * len(_FRAME))
    client = ExplodingCommitClient()
    s._client = client
    s._provider_generation = 130
    s._RECONNECT_BACKOFF_BASE_SEC = 0

    async def _cannot_reconnect(*args, **kwargs):
        return False

    s._init_or_reuse_client = _cannot_reconnect
    for seed in range(1, 7):
        await s._handle_audio_bytes(_distinct_frame(seed))

    sent = s._websocket.sent
    error_idxs = [i for i, p in enumerate(sent) if p.get("type") == "error"]
    idle_idxs = [i for i, p in enumerate(sent) if p.get("status") == "idle"]
    assert len(error_idxs) == 1
    assert len(idle_idxs) == 1
    assert error_idxs[0] < idle_idxs[0]
    assert sent[error_idxs[0]]["turn_id"] == 900
    assert s._finalized is True


@pytest.mark.asyncio
async def test_full_rollover_tail_backpressures_then_starts_next_segment():
    frames = [_distinct_frame(seed) for seed in range(1, 16)]
    segment_bytes = 5 * len(_FRAME)
    s = make_session(max_turn_audio_bytes=segment_bytes)
    providers = [FakeClient(), FakeClient(), FakeClient()]
    commit_events = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
    replacements = iter(providers[1:])

    for provider, committed in zip(providers, commit_events):
        async def _commit(event=committed):
            event.set()

        provider.commit_audio = _commit

    async def _init_replacement(turn_id=None, **kwargs):
        replacement = next(replacements)
        s._client = replacement
        s._provider_generation += 1
        s._provider_audio_bytes = 0
        s._begin_provider_epoch_locked()
        s._register_client_handlers(replacement)
        return True

    s._client = providers[0]
    s._provider_generation = 140
    s._register_client_handlers(providers[0])
    s._init_or_reuse_client = _init_replacement
    s._RECONNECT_BACKOFF_BASE_SEC = 0

    for frame in frames[:10]:
        await s._handle_audio_bytes(frame)
    await asyncio.wait_for(commit_events[0].wait(), 1)
    assert s._rollover_tail_bytes == segment_bytes

    blocked_audio = asyncio.create_task(s._handle_audio_bytes(frames[10]))
    await asyncio.sleep(0)
    assert not blocked_audio.done()
    assert s._rollover_tail_bytes == segment_bytes

    await providers[0].handlers["input_audio_buffer.committed"](
        {"item_id": "backpressure-A"}
    )
    await providers[0].handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "backpressure-A", "transcript": "一"})
    first_rollover = s._segment_rollover_task
    assert first_rollover is not None
    await asyncio.wait_for(asyncio.shield(first_rollover), 1)

    # Once segment A frees the tail fence, the blocked frame crosses segment B's
    # now-full boundary and starts its commit instead of being dropped.
    await asyncio.wait_for(commit_events[1].wait(), 1)
    await asyncio.wait_for(blocked_audio, 1)
    assert providers[1].sent_audio == frames[5:10]
    assert s._rollover_tail_chunks == [frames[10]]
    assert s._rollover_tail_bytes == len(_FRAME)

    await providers[1].handlers["input_audio_buffer.committed"](
        {"item_id": "backpressure-B"}
    )
    await providers[1].handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "backpressure-B", "transcript": "二"})
    second_rollover = s._segment_rollover_task
    assert second_rollover is not None
    await asyncio.wait_for(asyncio.shield(second_rollover), 1)
    assert providers[2].sent_audio == [frames[10]]

    for frame in frames[11:]:
        await s._handle_audio_bytes(frame)
    stop = asyncio.create_task(s._handle_stop_recording({"turn_id": 900}))
    await asyncio.wait_for(commit_events[2].wait(), 1)
    await providers[2].handlers["input_audio_buffer.committed"](
        {"item_id": "backpressure-C"}
    )
    await providers[2].handlers[
        "conversation.item.input_audio_transcription.completed"
    ]({"item_id": "backpressure-C", "transcript": "三"})
    await asyncio.wait_for(stop, 1)

    assert providers[2].sent_audio == frames[10:]
    assert _client_text_view(s) == "一二三"
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    _assert_s1(s)


@pytest.mark.asyncio
async def test_segment_rollover_watchdog_exception_uses_failure_barrier_and_sets_done():
    s = make_session(transcription_finalize_timeout_sec=0.01)
    s._segment_rollover_pending = True
    s._segment_rollover_done.clear()
    s._finalize_commit_attempts = 1

    async def _explode(_reason):
        raise RuntimeError("watchdog retry boom")

    s._retry_finalize_result_locked = _explode
    s._arm_segment_rollover_timeout()
    watchdog = s._segment_rollover_timeout_task
    assert watchdog is not None
    await asyncio.wait_for(asyncio.shield(watchdog), 1)

    sent = s._websocket.sent
    error_idxs = [i for i, p in enumerate(sent) if p.get("type") == "error"]
    idle_idxs = [i for i, p in enumerate(sent) if p.get("status") == "idle"]
    assert s._finalized is True
    assert s._turn_done.is_set()
    assert s._segment_rollover_done.is_set()
    assert len(error_idxs) == 1
    assert len(idle_idxs) == 1
    assert error_idxs[0] < idle_idxs[0]


@pytest.mark.asyncio
async def test_recovery_state_and_failure_notifications_are_best_effort():
    s = make_session()
    s._websocket = ExplodingSendWebSocket()
    s._transport_state = TransportState.RECOVERING
    s._is_recording = False
    s._openai_ready.clear()

    await s._send_recovery_event("recovery_started")
    await s._send_recovery_event("recovery_failed", attempts=5)
    await s._publish_transport_ready()

    assert s._transport_state == TransportState.RECORDING
    assert s._is_recording is True
    assert s._openai_ready.is_set()


@pytest.mark.asyncio
async def test_rollover_watchdog_client_send_failure_releases_backpressure():
    segment_bytes = 5 * len(_FRAME)
    s = make_session(
        max_turn_audio_bytes=segment_bytes,
        transcription_finalize_timeout_sec=0.01,
    )
    first = FakeClient()
    committed = asyncio.Event()

    async def _commit():
        committed.set()

    first.commit_audio = _commit
    s._client = first
    s._provider_generation = 150
    s._register_client_handlers(first)
    _stub_reconnect_to(s, FakeClient())
    s._RECONNECT_BACKOFF_BASE_SEC = 0
    # Isolate the watchdog path from the independent backpressure deadline.
    s._segment_rollover_timeout_budget_sec = lambda: 1.0

    for seed in range(1, 11):
        await s._handle_audio_bytes(_distinct_frame(seed))
    await asyncio.wait_for(committed.wait(), 1)
    assert s._rollover_tail_bytes == segment_bytes

    blocked = asyncio.create_task(s._handle_audio_bytes(_distinct_frame(11)))
    await asyncio.sleep(0)
    assert not blocked.done()
    s._websocket = ExplodingSendWebSocket()

    await asyncio.wait_for(blocked, 2)

    assert s._finalized is True
    assert s._turn_done.is_set()
    assert s._segment_rollover_done.is_set()
    assert blocked.done()


@pytest.mark.asyncio
async def test_full_rollover_tail_backpressure_timeout_uses_failure_barrier():
    segment_bytes = 5 * len(_FRAME)
    s = make_session(
        max_turn_audio_bytes=segment_bytes,
        transcription_finalize_timeout_sec=0.01,
    )
    first = FakeClient()
    committed = asyncio.Event()

    async def _commit():
        committed.set()

    first.commit_audio = _commit
    s._client = first
    s._provider_generation = 160
    s._register_client_handlers(first)
    # Prove the receive-side deadline independently of the watchdog.
    s._arm_segment_rollover_timeout = lambda: None
    s._RECONNECT_BACKOFF_BASE_SEC = 0

    for seed in range(1, 11):
        await s._handle_audio_bytes(_distinct_frame(seed))
    await asyncio.wait_for(committed.wait(), 1)
    assert s._rollover_tail_bytes == segment_bytes

    blocked = asyncio.create_task(s._handle_audio_bytes(_distinct_frame(11)))
    await asyncio.wait_for(blocked, 1)

    sent = s._websocket.sent
    error_idxs = [i for i, p in enumerate(sent) if p.get("type") == "error"]
    idle_idxs = [i for i, p in enumerate(sent) if p.get("status") == "idle"]
    assert s._finalized is True
    assert s._turn_done.is_set()
    assert s._segment_rollover_done.is_set()
    assert s._turn_audio_chunks == []
    assert s._rollover_tail_chunks == []
    assert len(error_idxs) == 1
    assert len(idle_idxs) == 1
    assert error_idxs[0] < idle_idxs[0]


@pytest.mark.parametrize(
    "completed, partial, expected",
    [
        ("short", "a materially longer streamed partial", "a materially longer streamed partial"),
        ("a materially longer completed result", "short", "a materially longer completed result"),
        ("  same  ", "same", "same"),
        ("", "partial", "partial"),
        ("completed", "", "completed"),
    ],
)
def test_w10_finalize_transcript_resolver_prefers_longer_nonblank_text(
    completed, partial, expected
):
    assert TranscriptionTurnSession._select_completed_authority(completed, partial) == expected
