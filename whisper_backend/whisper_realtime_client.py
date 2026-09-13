"""OpenAI Realtime transcription-session client (WhisperWave).

Connects to the same GA Realtime WebSocket endpoint as MicWave but drives a
`type:"transcription"` session (gpt-4o-transcribe by default) instead of a conversation
session. Structural differences vs MicWave's OpenAIRealtimeAudioTextClient:

  * session.update payload is `{type:"transcription", audio:{input:{...}}}` with
    no instructions/tools/response face and `turn_detection: null` (WhisperWave
    uses explicit stop/commit).
  * compatible transcription models receive one vocabulary-bias prompt when a
    provider session is created; the prompt cannot create an answer surface.
  * there is no `start_response()`; committing the buffer triggers transcription
    directly and the server listens to
    conversation.item.input_audio_transcription.delta/.completed.

The session.created (connect) and session.updated (config-applied) confirmations
keep MicWave's fail-closed semantics: on a missing/unexpected frame we resend
once and then abort so the caller rebuilds a clean session rather than recording
against an unconfigured session.
"""
import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from typing import List, Optional

import websockets

from .config import (
    WHISPER_TRANSCRIBE_MODEL,
    WHISPERWAVE_DELAY,
    WHISPERWAVE_INPUT_SAMPLE_RATE,
    WHISPERWAVE_LANGUAGE,
    WHISPERWAVE_NOISE_REDUCTION,
    transcription_delay_for_model,
)
from .realtime_client_base import RealtimeClientBase
from .transcription_prompt import (
    TRANSCRIPTION_BIAS_PROMPT,
    transcription_prompt_for_model,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class WhisperRealtimeTranscriptionClient(RealtimeClientBase):
    def __init__(
        self,
        api_key: str,
        model: str = WHISPER_TRANSCRIBE_MODEL,
        delay: str = WHISPERWAVE_DELAY,
        prompt: str = TRANSCRIPTION_BIAS_PROMPT,
        language: str = WHISPERWAVE_LANGUAGE,
        noise_reduction: str = WHISPERWAVE_NOISE_REDUCTION,
        input_sample_rate: int = WHISPERWAVE_INPUT_SAMPLE_RATE,
    ):
        super().__init__(api_key)
        self.model = model
        self.delay = delay
        self.prompt = prompt
        self.language = language
        self.noise_reduction = noise_reduction
        self.input_sample_rate = input_sample_rate
        # GA transcription sessions connect with ?intent=transcription. The GA
        # endpoint REQUIRES a query param: a bare connect returns
        # invalid_request_error/missing_model, and ?model=gpt-realtime-whisper is
        # rejected as invalid_model (grw is not a conversation URL model).
        # ?intent=transcription yields session.created then, after our
        # session.update, session.updated. (Verified against the live API
        # 2026-07-12; the model itself is selected inside the transcription
        # config, not the URL.)
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.connect_url = f"{self.base_url}?intent=transcription"
        self._last_send_error_log: float = 0
        self._provider_event_metrics = (
            os.getenv("WHISPERWAVE_PROVIDER_EVENT_METRICS", "0") == "1"
        )
        self._provider_event_seq = 0

    @staticmethod
    def _event_item_tag(data: dict) -> str:
        item_id = data.get("item_id")
        if not item_id and isinstance(data.get("item"), dict):
            item_id = data["item"].get("id")
        if not item_id:
            return "-"
        return hashlib.sha256(str(item_id).encode("utf-8")).hexdigest()[:10]

    def _log_provider_event_metric(self, data: dict) -> None:
        """Log provider ordering/size metadata without frame or transcript text."""
        if not self._provider_event_metrics or not isinstance(data, dict):
            return
        self._provider_event_seq += 1
        delta = data.get("delta")
        transcript = data.get("transcript")
        logger.info(
            "Provider event metric seq=%d type=%s item=%s delta_len=%d transcript_len=%d",
            self._provider_event_seq,
            data.get("type", "unknown"),
            self._event_item_tag(data),
            len(delta) if isinstance(delta, str) else 0,
            len(transcript) if isinstance(transcript, str) else 0,
        )

    def _build_session_config(self) -> dict:
        transcription: dict = {"model": self.model}
        effective_delay = transcription_delay_for_model(self.model, self.delay)
        if effective_delay:
            transcription["delay"] = effective_delay
        effective_prompt = transcription_prompt_for_model(self.model, self.prompt)
        if effective_prompt:
            transcription["prompt"] = effective_prompt
        if self.language:
            transcription["language"] = self.language
        audio_input: dict = {
            "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
            "transcription": transcription,
            # Product contract: explicit stop commits the buffered utterance.
            "turn_detection": None,
        }
        if self.noise_reduction:
            audio_input["noise_reduction"] = {"type": self.noise_reduction}
        return {"type": "transcription", "audio": {"input": audio_input}}

    async def connect(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
    ):
        """Connect and configure a transcription session.

        modalities/instructions are accepted for interface parity with the base
        class but ignored — a transcription session has neither.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            self.ws = await websockets.connect(self.connect_url, extra_headers=headers)
        except TypeError:
            # Newer websockets uses 'additional_headers'
            self.ws = await websockets.connect(self.connect_url, additional_headers=headers)

        # Wait for the initial session.created frame (the Realtime endpoint always
        # creates a session on connect, then we switch it to transcription).
        try:
            response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for OpenAI session.created")
            try:
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                logger.warning("Timeout/error closing half-open WS after session.created timeout")
            self.ws = None
            raise
        response_data = json.loads(response)
        first_frame_type = response_data.get("type")
        if first_frame_type != "session.created":
            # Fail-closed on an unexpected first frame. Log only redacted error
            # metadata, never a full body that could carry transcript text.
            if first_frame_type == "error":
                err = response_data.get("error", {})
                logger.error(
                    "OpenAI connect: first frame is an error (type=%s code=%s) — aborting",
                    err.get("type", "unknown"), err.get("code", "unknown"),
                )
            else:
                logger.error(
                    "OpenAI connect: unexpected first frame type=%r (expected "
                    "session.created) — aborting connect", first_frame_type,
                )
            try:
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self.ws = None
            raise RuntimeError(
                f"OpenAI first frame was {first_frame_type!r}, not session.created"
            )

        self.session_id = response_data.get("session", {}).get("id")
        logger.info("Transcription session created with ID: %s", self.session_id)

        session_config_payload = self._build_session_config()
        transcription_config = session_config_payload["audio"]["input"]["transcription"]
        prompt_value = transcription_config.get("prompt", "")
        logger.info(
            "Configuring transcription session "
            "(model=%s delay=%s prompt_chars=%d language=%s noise=%s)",
            self.model,
            transcription_config.get("delay", "(omitted)"),
            len(prompt_value),
            self.language or "(auto)",
            self.noise_reduction or "(off)",
        )
        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": session_config_payload,
        }, ensure_ascii=False))

        # Wait for session.updated so the transcription config is active before
        # any audio is forwarded. On timeout, resend once; still failing → abort
        # so the caller rebuilds a clean session (fail-closed).
        confirmed = await self._recv_session_updated(timeout=5.0)
        if not confirmed:
            logger.warning("session.updated not confirmed; resending session.update once")
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": session_config_payload,
            }, ensure_ascii=False))
            confirmed = await self._recv_session_updated(timeout=5.0)
        if not confirmed:
            logger.error(
                "session.updated not confirmed after resend; aborting connect to "
                "force a clean session rebuild"
            )
            try:
                await asyncio.wait_for(self.ws.close(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self.ws = None
            raise RuntimeError("OpenAI session.updated not confirmed after resend")
        logger.info("OpenAI transcription session.updated confirmed")

        self.register_handler("default", self.default_handler)
        self.receive_task = asyncio.create_task(self.receive_messages())

    async def _recv_session_updated(self, timeout: float) -> bool:
        """Read frames until session.updated arrives or timeout elapses.

        Used during connect() before the receive loop starts. Interleaved
        non-session.updated frames are ignored; an error frame short-circuits to
        a negative result.
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
                    "Received error while awaiting session.updated (type=%s code=%s)",
                    err.get("type", "unknown"), err.get("code", "unknown"),
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
            logger.info("Refreshed transcription session configuration via session.update")
            await asyncio.wait_for(session_updated_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
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
            raise RuntimeError("WebSocket is not open. Cannot refresh transcription session.")

        session_config_payload = self._build_session_config()
        confirmed = await self._send_session_update_and_wait(session_config_payload, timeout=5.0)
        if not confirmed:
            logger.warning("session.updated not confirmed on refresh; resending once")
            confirmed = await self._send_session_update_and_wait(session_config_payload, timeout=5.0)
        if not confirmed:
            logger.error(
                "session.updated not confirmed after refresh resend; forcing rebuild"
            )
            raise RuntimeError("OpenAI session.updated not confirmed on refresh")
        logger.info("OpenAI transcription session.updated confirmed (refresh)")

    async def receive_messages(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                self._log_provider_event_metric(data)
                message_type = data.get("type", "default")
                handler = self.handlers.get(message_type, self.handlers.get("default"))
                if handler:
                    await handler(data)
                else:
                    logger.warning(f"No handler for message type: {message_type}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"OpenAI WebSocket connection closed: {e}")
        except asyncio.CancelledError:
            logger.info("OpenAI receive_messages task cancelled")
            return  # don't fire on_disconnect for intentional cancellation
        except Exception as e:
            logger.error(f"Error in receive_messages: {e}", exc_info=True)
        await self._fire_on_disconnect()

    async def default_handler(self, data: dict):
        message_type = data.get("type", "unknown")
        logger.warning(f"Unhandled message type received from OpenAI: {message_type}")

    async def send_audio(self, audio_data: bytes) -> bool:
        """Append PCM16 audio to the provider input buffer.

        Returns True iff the frame was actually sent. Returns False when the WS
        is not open so the caller can BUFFER the frame to conserve it (task
        0446) rather than treat a not-open send as delivered and lose it — this
        is the send-side half of the first-utterance loss fix.
        """
        if self._is_ws_open():
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode("utf-8"),
            }))
            return True
        now = time.time()
        if now - self._last_send_error_log >= 5.0:
            logger.error("WebSocket is not open. Cannot send audio. (suppressing repeats for 5s)")
            self._last_send_error_log = now
        return False

    async def commit_audio(self):
        """Commit the audio buffer. Triggers transcription; creates a user item
        but NOT a response (transcription sessions have no response surface)."""
        self._require_ws_open()
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        logger.info("Sent input_audio_buffer.commit to OpenAI (transcription trigger)")

    async def clear_audio_buffer(self):
        if self._is_ws_open():
            await self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            logger.info("Sent input_audio_buffer.clear to OpenAI")
        else:
            logger.error("WebSocket is not open. Cannot clear audio buffer.")

    async def close(self):
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
            logger.info("Closed OpenAI transcription WebSocket connection")
