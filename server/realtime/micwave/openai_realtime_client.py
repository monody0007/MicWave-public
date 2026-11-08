import websockets
import json
import base64
import logging
import time
from typing import Optional, Callable, Dict, List
import asyncio
import itertools
import os
from prompts import get_realtime_prompt
from config import (
    OPENAI_REALTIME_MODEL,
    BRAINWAVE_MAX_OUTPUT_TOKENS,
)
from realtime_client_base import RealtimeClientBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# How long commit_audio() waits for input_audio_buffer.committed. The ack only
# needs the provider to drain the already-sent appends, so it is far below any
# turn budget; missing it means the session is unusable for this turn.
COMMIT_ACK_TIMEOUT_SEC = 15.0

# response.metadata key that ties an out-of-band response back to the audio
# item it transcribes (task 1082), so response.done can be matched to its item.
TURN_ITEM_METADATA_KEY = "turn_item_id"

# Completion signals a committed turn item waits for before it is deleted from
# the provider conversation: our response finished, and the per-item input
# transcription finished (completed or failed). Deleting earlier could cut off
# the faithful input transcript the servers wait for after response.done.
_ITEM_DONE_SIGNALS = frozenset({"response", "transcript"})


class CommitNotAcknowledgedError(RuntimeError):
    """input_audio_buffer.committed did not arrive for this turn's commit."""


class TurnIsolationBrokenError(RuntimeError):
    """An earlier commit on this session failed; the session must be recreated.

    The provider's committed ack carries no reference to the client commit, so
    after a lost or rejected commit a late ack could be mistaken for the next
    turn's item. Rather than guess, the client refuses further turns.
    """


class OpenAIRealtimeAudioTextClient(RealtimeClientBase):
    def __init__(self, api_key: str, model: str = OPENAI_REALTIME_MODEL):
        super().__init__(api_key)
        self.model = model
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.last_audio_time = None
        self.auto_commit_interval = 5
        self.include_instructions_each_response = (
            os.getenv("BRAINWAVE_INCLUDE_INSTRUCTIONS_EACH_RESPONSE", "1") == "1"
        )
        self._last_send_error_log: float = 0
        # Turn isolation on reused sessions (task 1082). Every committed user
        # audio item stays in the provider's default conversation; a plain
        # response.create would read all of them. Each response is therefore
        # created out-of-band with input = [this turn's item] only, and the item
        # is deleted once the turn is done (hygiene, not correctness).
        self._event_ids = itertools.count(1)
        self._commit_waiter: Optional[tuple] = None  # (event_id, Future[item_id])
        self._turn_item_id: Optional[str] = None
        # Item bound by the current commit's ack; set inside the receive loop so
        # handlers dispatched right after the ack can match events by item_id.
        self.committed_item_id: Optional[str] = None
        self._turn_isolation_broken = False
        self._pending_item_signals: Dict[str, set] = {}
        self._delete_event_ids: Dict[str, str] = {}  # event_id -> item_id

    def _build_session_config(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
    ) -> dict:
        effective_modalities = modalities or ["text"]
        session: dict = {
            "type": "realtime",
            "output_modalities": effective_modalities,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {
                        "model": "gpt-4o-transcribe",
                    },
                    "turn_detection": None,
                },
            },
            "instructions": instructions or get_realtime_prompt(),
        }
        return session
        
    async def connect(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
    ):
        """Connect to OpenAI's realtime API and configure the session"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # Support both websockets param names across versions: extra_headers (older) and additional_headers (newer)
        try:
            self.ws = await websockets.connect(
                f"{self.base_url}?model={self.model}",
                extra_headers=headers,
            )
        except TypeError:
            # Fallback for newer versions where the kwarg is 'additional_headers'
            self.ws = await websockets.connect(
                f"{self.base_url}?model={self.model}",
                additional_headers=headers,
            )
        
        # Wait for session creation
        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for OpenAI session.created")
            try:
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                logger.warning("Timeout or error closing half-open OpenAI WebSocket after session.created timeout")
            self.ws = None
            raise
        response_data = json.loads(response)
        first_frame_type = response_data.get("type")
        if first_frame_type != "session.created":
            # Fail-closed on an unexpected first frame (task 0436 M1). Any frame
            # other than session.created means the session was not created; do
            # not send session config or start the receive loop — close and
            # raise so the caller rebuilds a clean session. For an error frame we
            # log only the redacted error type/code, never the transcript-free
            # but potentially sensitive full body.
            if first_frame_type == "error":
                err = response_data.get("error", {})
                logger.error(
                    "OpenAI connect: first frame is an error "
                    "(type=%s code=%s) — aborting connect",
                    err.get("type", "unknown"),
                    err.get("code", "unknown"),
                )
            else:
                logger.error(
                    "OpenAI connect: unexpected first frame type=%r (expected "
                    "session.created) — aborting connect",
                    first_frame_type,
                )
            try:
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self.ws = None
            raise RuntimeError(
                f"OpenAI first frame was {first_frame_type!r}, not session.created"
            )

        if first_frame_type == "session.created":
            self.session_id = response_data["session"]["id"]
            logger.info(f"Session created with ID: {self.session_id}")

            session_config_payload = self._build_session_config(
                modalities=modalities,
                instructions=instructions,
            )
            logger.info("Configuring session for conversation mode with transcription and no turn detection.")

            # Configure session
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": session_config_payload
            }, ensure_ascii=False))

            # Wait for session.updated confirmation before proceeding.
            # Without this, audio may arrive at the API before instructions are
            # applied, causing default Q&A behaviour on the first turn (task
            # 0436 R5). If confirmation never arrives, resend once; if it still
            # fails, abort the connect so the caller rebuilds a clean session
            # instead of silently recording with default (answer) behaviour.
            confirmed = await self._recv_session_updated(timeout=5.0)
            if not confirmed:
                logger.warning(
                    "session.updated not confirmed; resending session.update once"
                )
                await self.ws.send(json.dumps({
                    "type": "session.update",
                    "session": session_config_payload,
                }, ensure_ascii=False))
                confirmed = await self._recv_session_updated(timeout=5.0)
            if not confirmed:
                logger.error(
                    "session.updated not confirmed after resend; aborting connect "
                    "to force a clean session rebuild"
                )
                try:
                    await asyncio.wait_for(self.ws.close(), timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                self.ws = None
                raise RuntimeError("OpenAI session.updated not confirmed after resend")
            logger.info("OpenAI session.updated confirmed")

        # Register the default handler
        self.register_handler("default", self.default_handler)

        # Start the receiver coroutine
        self.receive_task = asyncio.create_task(self.receive_messages())

    async def _recv_session_updated(self, timeout: float) -> bool:
        """Read frames until session.updated arrives or timeout elapses.

        Used during connect() before the receive loop starts, so frames are read
        directly here. Interleaved non-session.updated frames are ignored; an
        error frame short-circuits to a negative result.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                return False
            try:
                frame = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            frame_type = frame.get("type")
            if frame_type == "session.updated":
                return True
            if frame_type == "error":
                err = frame.get("error", {}) if isinstance(frame, dict) else {}
                logger.error(
                    "Received error while awaiting session.updated "
                    "(type=%s code=%s)",
                    err.get("type", "unknown"),
                    err.get("code", "unknown"),
                )
                return False
            logger.debug("Ignoring %s while awaiting session.updated", frame_type)

    async def _send_session_update_and_wait(
        self,
        session_config_payload: dict,
        timeout: float,
    ) -> bool:
        """Send session.update and wait for session.updated via a temp handler.

        The receive loop is already consuming frames, so we hook session.updated
        with a temporary handler that sets an Event, then always restore the
        original handler. Returns True iff confirmation arrived within timeout.
        """
        session_updated_event = asyncio.Event()
        original_handler = self.handlers.get("session.updated")

        async def _on_session_updated(data):
            session_updated_event.set()
            if original_handler:
                self.handlers["session.updated"] = original_handler
                await original_handler(data)
            else:
                self.handlers.pop("session.updated", None)

        self.handlers["session.updated"] = _on_session_updated
        try:
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": session_config_payload,
            }, ensure_ascii=False))
            logger.info("Refreshed OpenAI session configuration via session.update")
            await asyncio.wait_for(session_updated_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            # If the temp handler is still installed (timeout path), restore it.
            if self.handlers.get("session.updated") is _on_session_updated:
                if original_handler:
                    self.handlers["session.updated"] = original_handler
                else:
                    self.handlers.pop("session.updated", None)

    async def refresh_session(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
    ):
        if not self._is_ws_open():
            raise RuntimeError("WebSocket is not open. Cannot refresh OpenAI session.")

        session_config_payload = self._build_session_config(
            modalities=modalities,
            instructions=instructions,
        )
        # Wait for session.updated so instructions are active before any audio is
        # forwarded. On timeout, resend once; if still unconfirmed, fail-closed
        # (task 0436 R5) so the caller rebuilds the session instead of recording
        # with default (answer) behaviour.
        confirmed = await self._send_session_update_and_wait(
            session_config_payload, timeout=5.0
        )
        if not confirmed:
            logger.warning("session.updated not confirmed on refresh; resending once")
            confirmed = await self._send_session_update_and_wait(
                session_config_payload, timeout=5.0
            )
        if not confirmed:
            logger.error(
                "session.updated not confirmed after refresh resend; forcing "
                "session rebuild"
            )
            raise RuntimeError("OpenAI session.updated not confirmed on refresh")
        logger.info("OpenAI session.updated confirmed (refresh)")


    async def send_instructions_audio(self):
        """Send the instructions.wav file as audio input to be appended to current buffer"""
        instructions_path = "instructions.wav"
        if not os.path.exists(instructions_path):
            logger.warning(f"Instructions audio file not found: {instructions_path}")
            return
            
        try:
            with open(instructions_path, "rb") as f:
                audio_data = f.read()
            
            # Send the instructions audio to the buffer (appends to existing user audio)
            await self.send_audio(audio_data)
            logger.info("Sent instructions audio to OpenAI buffer (appended to user audio)")
            
        except Exception as e:
            logger.error(f"Error sending instructions audio: {e}")
    
    async def receive_messages(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                message_type = data.get("type", "default")
                if self._consume_hygiene_event(message_type, data):
                    continue
                handler = self.handlers.get(message_type, self.handlers.get("default"))
                if handler:
                    await handler(data)
                else:
                    logger.warning(f"No handler for message type: {message_type}")
                # After dispatch, so server handlers see an event (e.g. the input
                # transcript) before the client acts on it (e.g. deletes the item).
                await self._track_turn_event(message_type, data)
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"OpenAI WebSocket connection closed: {e}")
        except asyncio.CancelledError:
            logger.info("OpenAI receive_messages task cancelled")
            self._fail_commit_waiter(ConnectionError("OpenAI receive loop cancelled"))
            return  # don't fire on_disconnect for intentional cancellation
        except Exception as e:
            logger.error(f"Error in receive_messages: {e}", exc_info=True)
        self._fail_commit_waiter(ConnectionError("OpenAI WebSocket connection closed"))
        # Connection dropped unexpectedly — notify the server
        await self._fire_on_disconnect()

    def _next_event_id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._event_ids)}"

    def _fail_commit_waiter(self, exc: BaseException):
        if self._commit_waiter is not None:
            _, future = self._commit_waiter
            if not future.done():
                future.set_exception(exc)

    def _consume_hygiene_event(self, message_type: str, data: dict) -> bool:
        """Swallow provider replies to the client's own item deletes.

        A failed delete must not reach the server's ``error`` handler: both
        servers treat any error as a failed turn, while a leftover item cannot
        affect results (responses only reference their own item).
        """
        if message_type == "error":
            err = data.get("error") or {}
            item_id = self._delete_event_ids.pop(err.get("event_id"), None)
            if item_id is None:
                return False
            logger.warning(
                "conversation.item.delete failed for %s (type=%s code=%s); "
                "ignored, responses never reference earlier items",
                item_id,
                err.get("type", "unknown"),
                err.get("code", "unknown"),
            )
            return True
        if message_type == "conversation.item.deleted":
            item_id = data.get("item_id")
            for event_id in [e for e, i in self._delete_event_ids.items() if i == item_id]:
                del self._delete_event_ids[event_id]
            logger.debug("conversation.item.deleted %s", item_id)
            return message_type not in self.handlers
        return False

    async def _track_turn_event(self, message_type: str, data: dict):
        """Bookkeeping for turn items: commit acks and item completion."""
        if message_type == "input_audio_buffer.committed":
            item_id = data.get("item_id")
            if not item_id:
                logger.error("input_audio_buffer.committed without item_id")
                self._fail_commit_waiter(
                    RuntimeError("input_audio_buffer.committed without item_id")
                )
                return
            if self._commit_waiter is None or self._commit_waiter[1].done():
                # Ack for a commit that already failed (late) or that we never
                # sent: it belongs to no turn. Never bind it; just remove it.
                logger.warning(
                    "input_audio_buffer.committed %s without a pending commit; deleting it",
                    item_id,
                )
                await self._delete_item(item_id)
                return
            self._pending_item_signals[item_id] = set(_ITEM_DONE_SIGNALS)
            self.committed_item_id = item_id
            self._commit_waiter[1].set_result(item_id)
        elif message_type == "error":
            err = data.get("error") or {}
            if self._commit_waiter is not None and err.get("event_id") == self._commit_waiter[0]:
                self._fail_commit_waiter(RuntimeError(
                    f"OpenAI rejected input_audio_buffer.commit "
                    f"(type={err.get('type', 'unknown')} code={err.get('code', 'unknown')})"
                ))
        elif message_type == "response.done":
            metadata = (data.get("response") or {}).get("metadata") or {}
            await self._mark_item_signal(metadata.get(TURN_ITEM_METADATA_KEY), "response")
        elif message_type in (
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.failed",
        ):
            await self._mark_item_signal(data.get("item_id"), "transcript")

    async def _mark_item_signal(self, item_id: Optional[str], signal: str):
        signals = self._pending_item_signals.get(item_id)
        if signals is None:
            return
        signals.discard(signal)
        if not signals:
            del self._pending_item_signals[item_id]
            await self._delete_item(item_id)

    async def _delete_item(self, item_id: str):
        """Best-effort removal of a finished turn item from the conversation.

        Bounds session size/cost only; correctness never depends on it.
        """
        event_id = self._next_event_id("delete")
        self._delete_event_ids[event_id] = item_id
        try:
            await self.ws.send(json.dumps({
                "type": "conversation.item.delete",
                "item_id": item_id,
                "event_id": event_id,
            }))
        except websockets.exceptions.ConnectionClosed as e:
            # Session is gone; its conversation goes with it.
            self._delete_event_ids.pop(event_id, None)
            logger.warning("Could not delete conversation item %s: %s", item_id, e)
    
    async def default_handler(self, data: dict):
        """Override default handler for OpenAI-specific logging"""
        message_type = data.get("type", "unknown")
        logger.warning(f"Unhandled message type received from OpenAI: {message_type}")
    
    async def send_audio(self, audio_data: bytes):
        if self._is_ws_open():
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode('utf-8')
            }))
        else:
            now = time.time()
            if now - self._last_send_error_log >= 5.0:
                logger.error("WebSocket is not open. Cannot send audio. (suppressing repeats for 5s)")
                self._last_send_error_log = now
    
    async def commit_audio(self) -> str:
        """Commit the audio buffer and return the committed user item id.

        Waits for input_audio_buffer.committed so start_response() can reference
        exactly this item. Raises if the provider rejects the commit or never
        acknowledges it; the turn then has no item to transcribe.

        Invariant: at most one commit is in flight per client, and a commit
        starts only after the previous turn ended (callers serialize turns:
        micwave SessionPool lock, desktop single stop flow). The committed ack
        is matched to the in-flight commit by arrival alone, and the stale-item
        sweep below assumes earlier turns are over.

        Any failure here (timeout, rejection, malformed ack, disconnect,
        cancellation) leaves an unmatched commit behind whose ack may still
        arrive, so the client then refuses all further commits with
        TurnIsolationBrokenError; callers must close it and build a new one.
        """
        self._require_ws_open()
        if self._turn_isolation_broken:
            raise TurnIsolationBrokenError(
                "an earlier commit failed on this session; recreate the client"
            )
        if self._commit_waiter is not None:
            raise RuntimeError("commit_audio() called while another commit is in flight")
        self._turn_item_id = None
        self.committed_item_id = None
        event_id = self._next_event_id("commit")
        future = asyncio.get_running_loop().create_future()
        self._commit_waiter = (event_id, future)
        try:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.commit",
                "event_id": event_id,
            }))
            logger.info("Sent input_audio_buffer.commit message to OpenAI")
            try:
                item_id = await asyncio.wait_for(future, timeout=COMMIT_ACK_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                raise CommitNotAcknowledgedError(
                    f"input_audio_buffer.committed not received within {COMMIT_ACK_TIMEOUT_SEC:.0f}s"
                ) from None
        except BaseException:
            self._turn_isolation_broken = True
            raise
        finally:
            self._commit_waiter = None
        # Items of earlier turns that never got both completion signals
        # (timeouts, errors) are removed now; their turns are over.
        for stale_item_id in [i for i in self._pending_item_signals if i != item_id]:
            del self._pending_item_signals[stale_item_id]
            await self._delete_item(stale_item_id)
        self._turn_item_id = item_id
        return item_id
    
    async def clear_audio_buffer(self):
        """Clear the audio buffer"""
        if self._is_ws_open():
            clear_message = json.dumps({"type": "input_audio_buffer.clear"})
            await self.ws.send(clear_message)
            logger.info("Sent input_audio_buffer.clear message to OpenAI")
        else:
            logger.error("WebSocket is not open. Cannot clear audio buffer.")
    
    async def start_response(self, instructions: str):
        """Start the response for the item committed by commit_audio().

        The response is out-of-band (conversation "none") and its input is only
        this turn's audio item, so earlier turns left in a reused session can
        never reach the model (task 1082).
        """
        self._require_ws_open()
        item_id = self._turn_item_id
        if item_id is None:
            raise RuntimeError("start_response() requires a successful commit_audio() for this turn")
        self._turn_item_id = None
        response_config = {
            "output_modalities": ["text"],
            "conversation": "none",
            "input": [{"type": "item_reference", "id": item_id}],
            "metadata": {TURN_ITEM_METADATA_KEY: item_id},
        }
        if self.include_instructions_each_response and instructions:
            response_config["instructions"] = instructions
        # Optional output cap (task 0436 A2/M7). Only sent when configured and
        # validated to [1, 4096]; unset preserves today's behaviour of relying
        # on the API defaults. GA Realtime has no per-response temperature.
        if BRAINWAVE_MAX_OUTPUT_TOKENS is not None:
            response_config["max_output_tokens"] = BRAINWAVE_MAX_OUTPUT_TOKENS

        await self.ws.send(json.dumps({
            "type": "response.create",
            "response": response_config
        }))
        logger.info(
            f"Started out-of-band response for {item_id} (text-only)"
            + (
                " (with per-response instructions)"
                if self.include_instructions_each_response
                else " (session instructions only)"
            )
        )
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await asyncio.wait_for(self.receive_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            self.receive_task = None
        if self.ws:
            try:
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self.ws = None
            logger.info("Closed OpenAI WebSocket connection")
