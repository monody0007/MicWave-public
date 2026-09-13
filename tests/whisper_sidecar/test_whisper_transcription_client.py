"""Tests for the WhisperWave transcription client.

Covers the transcription session.update payload shape and the fail-closed
session.updated / connect-first-frame semantics carried over from MicWave.
"""
import asyncio
import json
import logging
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_backend import whisper_realtime_client as wc
from whisper_backend.whisper_realtime_client import WhisperRealtimeTranscriptionClient


class _SendWS:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send(self, msg):
        self.sent.append(msg)

    async def close(self):
        self.closed = True


class _RecvWS:
    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []
        self.closed = False

    async def recv(self):
        if self._frames:
            return self._frames.pop(0)
        await asyncio.sleep(3600)

    async def send(self, msg):
        self.sent.append(msg)

    async def close(self):
        self.closed = True


def _make_client(**kw):
    return WhisperRealtimeTranscriptionClient(api_key="test-key", **kw)


def test_provider_event_metrics_are_opt_in_and_content_redacted(monkeypatch, caplog):
    monkeypatch.setenv("WHISPERWAVE_PROVIDER_EVENT_METRICS", "1")
    client = _make_client()
    caplog.set_level(logging.INFO, logger=wc.__name__)

    client._log_provider_event_metric({
        "type": "conversation.item.input_audio_transcription.delta",
        "item_id": "item-private-identifier",
        "delta": "private synthetic transcript body",
    })

    message = caplog.messages[-1]
    assert "seq=1" in message
    assert "type=conversation.item.input_audio_transcription.delta" in message
    assert "delta_len=33" in message
    assert "transcript_len=0" in message
    assert "item-private-identifier" not in message
    assert "private synthetic transcript body" not in message


def test_provider_event_metrics_default_off(monkeypatch, caplog):
    monkeypatch.delenv("WHISPERWAVE_PROVIDER_EVENT_METRICS", raising=False)
    client = _make_client()
    caplog.set_level(logging.INFO, logger=wc.__name__)
    client._log_provider_event_metric({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item-private-identifier",
        "transcript": "private synthetic transcript body",
    })
    assert caplog.messages == []


# ── session config payload shape ─────────────────────────────────────────────


def test_session_config_is_transcription_with_null_turn_detection():
    # Explicit model keeps this payload test independent of the developer's
    # ignored local .env override; config default behavior is tested separately.
    client = _make_client(model="gpt-4o-transcribe", delay="low")
    cfg = client._build_session_config()
    assert cfg["type"] == "transcription"
    audio_in = cfg["audio"]["input"]
    assert audio_in["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_in["transcription"]["model"] == "gpt-4o-transcribe"
    assert "delay" not in audio_in["transcription"]
    assert audio_in["transcription"]["prompt"]
    # WhisperWave uses explicit stop/commit rather than server VAD.
    assert audio_in["turn_detection"] is None
    # No instructions/tools/response face in a transcription session.
    assert "instructions" not in cfg and "tools" not in cfg


def test_session_config_keeps_delay_only_for_realtime_whisper():
    realtime = _make_client(
        model="gpt-realtime-whisper", delay="high", prompt="domain terms"
    )._build_session_config()
    transcription = realtime["audio"]["input"]["transcription"]
    assert transcription == {
        "model": "gpt-realtime-whisper",
        "delay": "high",
    }


def test_session_config_prompt_nonempty_and_empty_branches():
    enabled = _make_client(
        model="gpt-4o-transcribe", prompt="  domain terms  "
    )._build_session_config()
    assert enabled["audio"]["input"]["transcription"]["prompt"] == "domain terms"

    disabled = _make_client(
        model="gpt-4o-transcribe", prompt="  "
    )._build_session_config()
    assert "prompt" not in disabled["audio"]["input"]["transcription"]


def test_session_config_model_remains_constructor_configurable():
    cfg = _make_client(model="custom-transcription-model", prompt="bias")._build_session_config()
    transcription = cfg["audio"]["input"]["transcription"]
    assert transcription == {"model": "custom-transcription-model"}


def test_session_config_language_and_noise_are_conditional():
    bare = _make_client(language="", noise_reduction="")._build_session_config()
    assert "language" not in bare["audio"]["input"]["transcription"]
    assert "noise_reduction" not in bare["audio"]["input"]

    rich = _make_client(language="zh", noise_reduction="near_field")._build_session_config()
    assert rich["audio"]["input"]["transcription"]["language"] == "zh"
    assert rich["audio"]["input"]["noise_reduction"] == {"type": "near_field"}


# ── session.updated fail-closed ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refresh_session_resend_success_continues():
    client = _make_client()
    client.ws = _SendWS()
    calls = []

    async def fake_wait(payload, timeout):
        calls.append(payload)
        return len(calls) >= 2  # first fails, resend confirms

    client._send_session_update_and_wait = fake_wait
    await client.refresh_session()  # must NOT raise
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_refresh_session_resend_failure_raises_for_rebuild():
    client = _make_client()
    client.ws = _SendWS()
    calls = []

    async def fake_wait(payload, timeout):
        calls.append(payload)
        return False  # never confirms

    client._send_session_update_and_wait = fake_wait
    with pytest.raises(RuntimeError):
        await client.refresh_session()
    assert len(calls) == 2  # initial + one resend, then fail-closed


@pytest.mark.asyncio
async def test_recv_session_updated_ignores_interleaved_frames():
    client = _make_client()
    client.ws = _RecvWS([
        json.dumps({"type": "rate_limits.updated"}),
        json.dumps({"type": "session.updated"}),
    ])
    assert await client._recv_session_updated(timeout=1.0) is True


@pytest.mark.asyncio
async def test_recv_session_updated_error_frame_returns_false():
    client = _make_client()
    client.ws = _RecvWS([json.dumps({"type": "error", "error": {"message": "boom"}})])
    assert await client._recv_session_updated(timeout=1.0) is False


@pytest.mark.asyncio
async def test_recv_session_updated_timeout_returns_false():
    client = _make_client()
    client.ws = _RecvWS([])
    assert await client._recv_session_updated(timeout=0.05) is False


# ── connect() first-frame fail-closed ────────────────────────────────────────


class _FirstFrameWS:
    def __init__(self, first_frame):
        self._frames = [first_frame]
        self.sent = []
        self.closed = False

    async def recv(self):
        if self._frames:
            return self._frames.pop(0)
        await asyncio.sleep(3600)

    async def send(self, msg):
        self.sent.append(msg)

    async def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "first_frame",
    [
        json.dumps({"type": "error", "error": {"type": "invalid_request_error", "code": "bad"}}),
        json.dumps({"type": "rate_limits.updated"}),
        json.dumps({"type": "some.unexpected.frame"}),
    ],
)
@pytest.mark.asyncio
async def test_connect_non_session_created_first_frame_fails_closed(monkeypatch, first_frame):
    client = _make_client()
    ws = _FirstFrameWS(first_frame)

    async def fake_connect(url, **kwargs):
        return ws

    monkeypatch.setattr(wc.websockets, "connect", fake_connect)

    with pytest.raises(RuntimeError):
        await client.connect()

    assert client.session_id is None
    assert client.ws is None
    assert client.receive_task is None
    assert ws.closed is True
    assert not any("session.update" in s for s in ws.sent)


# ── message shapes ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_commit_and_clear_and_append_message_shapes():
    client = _make_client()
    client.ws = _SendWS()
    sent = await client.send_audio(b"\x01\x00" * 10)
    assert sent is True
    await client.commit_audio()
    await client.clear_audio_buffer()
    types = [json.loads(m)["type"] for m in client.ws.sent]
    assert types == [
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.clear",
    ]
    # No response.create anywhere — transcription session has no response face.
    assert not any("response.create" in m for m in client.ws.sent)


@pytest.mark.asyncio
async def test_send_audio_returns_false_without_open_provider_ws():
    client = _make_client()
    assert await client.send_audio(b"\x01\x00" * 10) is False

    closed = _SendWS()
    closed.closed = True
    client.ws = closed
    assert await client.send_audio(b"\x01\x00" * 10) is False
    assert closed.sent == []


# ── wire-level session.update resend / close (M5) ─────────────────────────────


@pytest.mark.asyncio
async def test_send_session_update_and_wait_writes_wire_frame_and_confirms():
    """The real _send_session_update_and_wait writes a session.update to the wire
    and returns True only when session.updated actually fires."""
    client = _make_client()
    client.ws = _SendWS()

    async def confirm_soon():
        await asyncio.sleep(0.01)
        handler = client.handlers.get("session.updated")
        assert handler is not None, "temp session.updated handler must be installed"
        await handler({"type": "session.updated"})

    task = asyncio.create_task(confirm_soon())
    ok = await client._send_session_update_and_wait(client._build_session_config(), timeout=1.0)
    await task
    assert ok is True
    update_frames = [json.loads(m) for m in client.ws.sent if json.loads(m).get("type") == "session.update"]
    assert len(update_frames) == 1
    # M5: assert the EXACT outer payload, not merely the frame type/count — a
    # frame carrying an empty/wrong session must fail this.
    assert update_frames[0] == {"type": "session.update", "session": client._build_session_config()}
    # The temp handler is removed (no leak) after confirmation.
    assert client.handlers.get("session.updated") is None


@pytest.mark.asyncio
async def test_send_session_update_and_wait_times_out_and_restores_handler():
    client = _make_client()
    client.ws = _SendWS()
    sentinel_calls = []

    async def original(data):
        sentinel_calls.append(data)

    client.handlers["session.updated"] = original
    ok = await client._send_session_update_and_wait(client._build_session_config(), timeout=0.05)
    assert ok is False
    update_frames = [json.loads(m) for m in client.ws.sent if json.loads(m).get("type") == "session.update"]
    assert len(update_frames) == 1  # frame really hit the wire
    # M5: the frame carried the exact expected outer payload.
    assert update_frames[0] == {"type": "session.update", "session": client._build_session_config()}
    # The original handler is restored on timeout (no leaked temp handler).
    assert client.handlers.get("session.updated") is original


class _ConnectWS:
    """Fake provider WS for connect() wire tests: yields queued recv frames then
    blocks; records every wire send; tracks close."""

    def __init__(self, recv_frames):
        self._frames = list(recv_frames)
        self.sent = []
        self.closed = False

    async def recv(self):
        if self._frames:
            return self._frames.pop(0)
        await asyncio.sleep(3600)

    async def send(self, msg):
        self.sent.append(msg)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_connect_resends_session_update_once_then_closes(monkeypatch):
    """connect() must resend session.update exactly once when session.updated is
    never confirmed, then close the socket and fail-closed (wire-level)."""
    client = _make_client()
    ws = _ConnectWS([json.dumps({"type": "session.created", "session": {"id": "s1"}})])

    async def fake_connect(url, **kwargs):
        return ws

    monkeypatch.setattr(wc.websockets, "connect", fake_connect)

    # Force confirmation to always fail — keeps the test fast (no 5s recv waits)
    # while still exercising the real resend + close + raise wiring.
    async def never_confirm(timeout):
        return False

    monkeypatch.setattr(client, "_recv_session_updated", never_confirm)

    with pytest.raises(RuntimeError):
        await client.connect()

    update_frames = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "session.update"]
    assert len(update_frames) == 2, "initial send + exactly one resend before fail-closed"
    # M5: BOTH the initial send and the resend carry the exact expected outer
    # payload, and the two frames are byte-for-byte identical.
    expected = {"type": "session.update", "session": client._build_session_config()}
    assert update_frames[0] == expected
    assert update_frames[1] == expected
    assert update_frames[0] == update_frames[1]
    assert ws.closed is True
    assert client.ws is None
    assert client.receive_task is None
