import websockets
import json
import base64
import logging
import time
from typing import Optional, Callable, Dict, List
import asyncio
import os
from prompts import get_realtime_prompt
from config import (
    OPENAI_REALTIME_MODEL,
    BRAINWAVE_MAX_OUTPUT_TOKENS,
)
from realtime_client_base import RealtimeClientBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        # Connection dropped unexpectedly — notify the server
        await self._fire_on_disconnect()
    
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
    
    async def commit_audio(self):
        """Commit the audio buffer and notify OpenAI"""
        self._require_ws_open()
        commit_message = json.dumps({"type": "input_audio_buffer.commit"})
        await self.ws.send(commit_message)
        logger.info("Sent input_audio_buffer.commit message to OpenAI")
    
    async def clear_audio_buffer(self):
        """Clear the audio buffer"""
        if self._is_ws_open():
            clear_message = json.dumps({"type": "input_audio_buffer.clear"})
            await self.ws.send(clear_message)
            logger.info("Sent input_audio_buffer.clear message to OpenAI")
        else:
            logger.error("WebSocket is not open. Cannot clear audio buffer.")
    
    async def start_response(self, instructions: str):
        """Start a new response with given instructions"""
        self._require_ws_open()
        response_config = {
            "output_modalities": ["text"],
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
            "Started response (text-only)"
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
