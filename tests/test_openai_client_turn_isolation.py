"""Turn isolation on a reused OpenAI Realtime session (task 1082).

A pooled session keeps every committed user audio item in the provider's
default conversation. A plain ``response.create`` reads that whole history, so
turn N re-transcribed turns N-1, N-2 (measured on micwave_phone, 2026-11-07).

The client must make each transcription response see only the current turn's
audio item by protocol construction (out-of-band response whose ``input`` is a
single ``item_reference``), and clean up the item afterwards without letting a
cleanup failure reach the server's turn handlers.
"""
import asyncio
import json
import pathlib
import sys

import pytest


from micwave_core import openai_realtime_client as oai
from micwave_core.openai_realtime_client import CommitNotAcknowledgedError, OpenAIRealtimeAudioTextClient


class _FakeRealtimeServer:
    """In-process stand-in for the provider WebSocket.

    Replies to client events the way the GA Realtime API does for the subset
    the client uses: commit -> committed (+ input transcription completed),
    response.create -> created / delta / done, item.delete -> item.deleted.
    Knobs let a test make delete or commit fail.
    """

    def __init__(self):
        self.sent = []
        self.closed = False
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._item_seq = 0
        self._resp_seq = 0
        self.fail_delete = False
        self.fail_commit = False
        self.transcribe_on_commit = True
        self.ack_commits = True  # False: commit gets no reply (ack lost/late)

    # websockets-compatible surface used by the client
    async def send(self, raw):
        msg = json.loads(raw)
        self.sent.append(msg)
        kind = msg.get("type")
        if kind == "input_audio_buffer.commit":
            if self.fail_commit:
                self._push({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "input_audio_buffer_commit_empty",
                        "message": "buffer too small",
                        "event_id": msg.get("event_id"),
                    },
                })
                return
            self._item_seq += 1
            item_id = f"item_{self._item_seq}"
            if not self.ack_commits:
                return
            self._push({"type": "input_audio_buffer.committed", "item_id": item_id})
            if self.transcribe_on_commit:
                self._push({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "transcript": f"turn {self._item_seq}",
                })
        elif kind == "response.create":
            self._resp_seq += 1
            rid = f"resp_{self._resp_seq}"
            meta = msg.get("response", {}).get("metadata")
            self._push({"type": "response.created", "response": {"id": rid, "metadata": meta}})
            self._push({"type": "response.output_text.delta", "response_id": rid, "delta": "x"})
            self._push({"type": "response.done", "response": {"id": rid, "metadata": meta}})
        elif kind == "conversation.item.delete":
            if self.fail_delete:
                self._push({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "item_not_found",
                        "message": "no such item",
                        "event_id": msg.get("event_id"),
                    },
                })
            else:
                self._push({"type": "conversation.item.deleted", "item_id": msg["item_id"]})

    def push_committed(self, item_id):
        self._push({"type": "input_audio_buffer.committed", "item_id": item_id})

    def push_transcription_completed(self, item_id):
        self._push({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "transcript": "late",
        })

    def _push(self, frame):
        self._inbox.put_nowait(json.dumps(frame))

    async def close(self):
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._inbox.get()

    def sent_of(self, kind):
        return [m for m in self.sent if m.get("type") == kind]


async def _start_client(fake):
    client = OpenAIRealtimeAudioTextClient(api_key="test-key", model="test-model")
    client.ws = fake
    client.register_handler("default", client.default_handler)
    client.receive_task = asyncio.create_task(client.receive_messages())
    return client


async def _run_turn(client, server_handlers):
    done = asyncio.Event()

    async def on_done(_data):
        done.set()

    async def on_error(data):
        server_handlers["errors"].append(data)
        done.set()

    client.register_handler("response.done", on_done)
    client.register_handler("error", on_error)
    await client.send_audio(b"\x00\x00" * 240)
    await client.commit_audio()
    await client.start_response("transcribe")
    await asyncio.wait_for(done.wait(), timeout=2.0)


async def _drain(times=5):
    for _ in range(times):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_second_turn_response_references_only_second_item():
    fake = _FakeRealtimeServer()
    client = await _start_client(fake)
    seen = {"errors": []}
    try:
        await _run_turn(client, seen)
        await _run_turn(client, seen)
    finally:
        await client.close()

    creates = fake.sent_of("response.create")
    assert len(creates) == 2
    second = creates[1]["response"]
    # Out-of-band: the response neither reads nor writes the default conversation.
    assert second.get("conversation") == "none"
    assert second.get("input") == [{"type": "item_reference", "id": "item_2"}]
    first = creates[0]["response"]
    assert first.get("input") == [{"type": "item_reference", "id": "item_1"}]
    assert seen["errors"] == []


@pytest.mark.asyncio
async def test_finished_turn_item_is_deleted_after_response_and_transcript():
    fake = _FakeRealtimeServer()
    client = await _start_client(fake)
    seen = {"errors": []}
    try:
        await _run_turn(client, seen)
        await _drain()
        deletes = [m["item_id"] for m in fake.sent_of("conversation.item.delete")]
        assert deletes == ["item_1"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_item_not_deleted_until_input_transcript_arrives():
    fake = _FakeRealtimeServer()
    fake.transcribe_on_commit = False
    client = await _start_client(fake)
    seen = {"errors": []}
    try:
        await _run_turn(client, seen)
        await _drain()
        # Deleting now could cut off the faithful input transcript the servers
        # wait for after response.done.
        assert fake.sent_of("conversation.item.delete") == []
        fake.push_transcription_completed("item_1")
        await _drain()
        assert [m["item_id"] for m in fake.sent_of("conversation.item.delete")] == ["item_1"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_abandoned_item_is_swept_on_next_commit():
    fake = _FakeRealtimeServer()
    fake.transcribe_on_commit = False
    client = await _start_client(fake)
    seen = {"errors": []}
    try:
        await _run_turn(client, seen)  # transcript for item_1 never arrives
        await _run_turn(client, seen)
        await _drain()
        deleted = [m["item_id"] for m in fake.sent_of("conversation.item.delete")]
        assert deleted == ["item_1"]
        # Turn 2 still references only its own item.
        assert fake.sent_of("response.create")[1]["response"]["input"] == [
            {"type": "item_reference", "id": "item_2"}
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_delete_failure_does_not_reach_server_error_handler():
    fake = _FakeRealtimeServer()
    fake.fail_delete = True
    client = await _start_client(fake)
    seen = {"errors": []}
    try:
        await _run_turn(client, seen)
        await _drain()
        assert len(fake.sent_of("conversation.item.delete")) == 1
        await _run_turn(client, seen)
        await _drain()
    finally:
        await client.close()
    # Hygiene failures are swallowed by the client; the turn result is untouched.
    assert seen["errors"] == []
    assert fake.sent_of("response.create")[1]["response"]["input"] == [
        {"type": "item_reference", "id": "item_2"}
    ]


@pytest.mark.asyncio
async def test_commit_error_raises_and_sends_no_response():
    fake = _FakeRealtimeServer()
    client = await _start_client(fake)
    seen = {"errors": []}
    try:
        await _run_turn(client, seen)
        fake.fail_commit = True

        async def on_error(data):
            seen["errors"].append(data)

        client.register_handler("error", on_error)
        await client.send_audio(b"\x00\x00")
        with pytest.raises(RuntimeError):
            await client.commit_audio()
        # Without a committed item for this turn there is nothing to respond to;
        # start_response must not fall back to the shared conversation.
        with pytest.raises(RuntimeError):
            await client.start_response("transcribe")
    finally:
        await client.close()
    assert len(fake.sent_of("response.create")) == 1
    # The commit error is still delivered to the server, as before.
    assert len(seen["errors"]) == 1


@pytest.mark.asyncio
async def test_late_ack_after_commit_timeout_fails_closed(monkeypatch):
    # Review round 2, Major #1: turn 1's ack is late; it must never be bound to
    # turn 2. The client refuses further turns so the server rebuilds the session.
    monkeypatch.setattr(oai, "COMMIT_ACK_TIMEOUT_SEC", 0.05)
    fake = _FakeRealtimeServer()
    fake.ack_commits = False
    client = await _start_client(fake)
    try:
        await client.send_audio(b"\x00\x00")
        with pytest.raises(CommitNotAcknowledgedError):
            await client.commit_audio()

        fake.ack_commits = True
        turn2 = asyncio.create_task(client.commit_audio())
        await _drain()
        fake.push_committed("item_1")  # late ack for turn 1
        await _drain()
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(turn2, timeout=1.0)
        with pytest.raises(RuntimeError):
            await client.start_response("transcribe")
        await _drain()
    finally:
        await client.close()
    assert len(fake.sent_of("input_audio_buffer.commit")) == 1
    assert fake.sent_of("response.create") == []
    # The orphaned item is removed, never referenced.
    assert [m["item_id"] for m in fake.sent_of("conversation.item.delete")] == ["item_1"]


@pytest.mark.asyncio
async def test_commit_rejection_also_breaks_turn_isolation():
    fake = _FakeRealtimeServer()
    fake.fail_commit = True
    client = await _start_client(fake)
    try:
        with pytest.raises(RuntimeError):
            await client.commit_audio()
        fake.fail_commit = False
        with pytest.raises(RuntimeError):
            await client.commit_audio()
    finally:
        await client.close()
    assert len(fake.sent_of("input_audio_buffer.commit")) == 1


@pytest.mark.asyncio
async def test_committed_without_pending_commit_is_deleted_not_tracked():
    fake = _FakeRealtimeServer()
    client = await _start_client(fake)
    try:
        fake.push_committed("item_orphan")
        await _drain()
        assert [m["item_id"] for m in fake.sent_of("conversation.item.delete")] == ["item_orphan"]
        assert client._pending_item_signals == {}
        # A normal turn afterwards still works and references only its item.
        seen = {"errors": []}
        await _run_turn(client, seen)
    finally:
        await client.close()
    assert fake.sent_of("response.create")[0]["response"]["input"] == [
        {"type": "item_reference", "id": "item_1"}
    ]


@pytest.mark.asyncio
async def test_committed_item_id_is_exposed_at_ack_time():
    fake = _FakeRealtimeServer()
    client = await _start_client(fake)
    seen_at_transcript = []

    async def on_transcript(data):
        # Dispatched right after the ack frame, possibly before commit_audio()
        # resumes: the bound id must already be visible to handlers.
        seen_at_transcript.append(client.committed_item_id)

    client.register_handler(
        "conversation.item.input_audio_transcription.completed", on_transcript
    )
    try:
        await _run_turn(client, {"errors": []})
    finally:
        await client.close()
    assert seen_at_transcript == ["item_1"]
