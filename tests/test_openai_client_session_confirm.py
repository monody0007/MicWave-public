"""Tests for OpenAI realtime session.updated confirmation (task 0436 R5/F8).

Covers the fail-closed behaviour: on session.updated timeout, resend once; if it
still does not confirm, raise so the caller rebuilds the session rather than
recording with default (answer) behaviour.
"""
import asyncio
import json
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import openai_realtime_client as oai
from openai_realtime_client import OpenAIRealtimeAudioTextClient


class _SendWS:
    """Minimal fake WebSocket that records sends and looks open."""

    def __init__(self):
        self.sent = []
        self.closed = False

    async def send(self, msg):
        self.sent.append(msg)

    async def close(self):
        self.closed = True


class _RecvWS:
    """Fake WebSocket that yields queued frames then blocks."""

    def __init__(self, frames):
        self._frames = list(frames)

    async def recv(self):
        if self._frames:
            return self._frames.pop(0)
        await asyncio.sleep(3600)  # block until cancelled/timed out


def _make_client():
    return OpenAIRealtimeAudioTextClient(api_key="test-key", model="test-model")


@pytest.mark.asyncio
async def test_refresh_session_resend_success_continues():
    client = _make_client()
    client.ws = _SendWS()
    calls = []

    async def fake_wait(payload, timeout):
        calls.append(payload)
        return len(calls) >= 2  # first attempt fails, resend confirms

    client._send_session_update_and_wait = fake_wait
    # Should NOT raise — the resend confirmed.
    await client.refresh_session()
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
    assert len(calls) == 2  # initial send + one resend, then fail-closed


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
    client.ws = _RecvWS([
        json.dumps({"type": "error", "error": {"message": "boom"}}),
    ])
    assert await client._recv_session_updated(timeout=1.0) is False


@pytest.mark.asyncio
async def test_recv_session_updated_timeout_returns_false():
    client = _make_client()
    client.ws = _RecvWS([])  # nothing to yield -> blocks -> timeout
    assert await client._recv_session_updated(timeout=0.05) is False


@pytest.mark.asyncio
async def test_send_session_update_restores_handler_on_timeout():
    client = _make_client()
    client.ws = _SendWS()
    sentinel = object()
    client.handlers["session.updated"] = sentinel
    # Never fire the temp handler -> should time out and restore the original.
    ok = await client._send_session_update_and_wait({"type": "realtime"}, timeout=0.05)
    assert ok is False
    assert client.handlers["session.updated"] is sentinel


# ── M1: connect() first-frame fail-closed ───────────────────────────────────


class _FirstFrameWS:
    """Fake ws whose recv yields one queued first frame then blocks."""

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

    monkeypatch.setattr(oai.websockets, "connect", fake_connect)

    with pytest.raises(RuntimeError):
        await client.connect()

    # No session config sent, session not marked ready, ws closed, no receive loop.
    assert client.session_id is None
    assert client.ws is None
    assert client.receive_task is None
    assert ws.closed is True
    assert not any("session.update" in s for s in ws.sent)


# ── M7: max_output_tokens only sent when set + validated; no temperature ─────


@pytest.mark.asyncio
async def test_start_response_omits_output_cap_when_unset(monkeypatch):
    client = _make_client()
    client.ws = _SendWS()
    monkeypatch.setattr(oai, "BRAINWAVE_MAX_OUTPUT_TOKENS", None)
    await client.start_response("please transcribe")
    sent = json.loads(client.ws.sent[-1])
    assert sent["type"] == "response.create"
    assert "max_output_tokens" not in sent["response"]
    assert "temperature" not in sent["response"]


@pytest.mark.asyncio
async def test_start_response_includes_valid_output_cap(monkeypatch):
    client = _make_client()
    client.ws = _SendWS()
    monkeypatch.setattr(oai, "BRAINWAVE_MAX_OUTPUT_TOKENS", 512)
    await client.start_response("please transcribe")
    sent = json.loads(client.ws.sent[-1])
    assert sent["response"]["max_output_tokens"] == 512
    assert "temperature" not in sent["response"]


def test_config_optional_int_in_range_validates_bounds(monkeypatch):
    from config import _optional_int_in_range

    monkeypatch.setenv("MW_TEST_TOK", "512")
    assert _optional_int_in_range("MW_TEST_TOK", 1, 4096) == 512
    monkeypatch.setenv("MW_TEST_TOK", "1")
    assert _optional_int_in_range("MW_TEST_TOK", 1, 4096) == 1
    monkeypatch.setenv("MW_TEST_TOK", "4096")
    assert _optional_int_in_range("MW_TEST_TOK", 1, 4096) == 4096
    monkeypatch.setenv("MW_TEST_TOK", "0")      # below range
    assert _optional_int_in_range("MW_TEST_TOK", 1, 4096) is None
    monkeypatch.setenv("MW_TEST_TOK", "99999")  # above range
    assert _optional_int_in_range("MW_TEST_TOK", 1, 4096) is None
    monkeypatch.setenv("MW_TEST_TOK", "abc")    # non-integer
    assert _optional_int_in_range("MW_TEST_TOK", 1, 4096) is None
    monkeypatch.delenv("MW_TEST_TOK", raising=False)  # unset
    assert _optional_int_in_range("MW_TEST_TOK", 1, 4096) is None


def test_no_temperature_pipeline_remains():
    # GA Realtime has no temperature field (task 0436 M7): the whole pipeline is
    # gone from config and the client module.
    import config
    assert not hasattr(config, "BRAINWAVE_TEMPERATURE")
    assert not hasattr(oai, "BRAINWAVE_TEMPERATURE")
