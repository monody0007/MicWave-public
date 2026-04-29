import websockets
import json
import base64
import logging
import time
from typing import Optional, Callable, Dict, List
import asyncio
import os
from prompts import get_realtime_prompt
from config import OPENAI_REALTIME_MODEL
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
        return {
            "modalities": effective_modalities,
            "input_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "gpt-4o-transcribe"
            },
            # Disable server-side VAD; rely on manual buffering/commits.
            "turn_detection": None,
            "instructions": instructions or get_realtime_prompt(),
        }
        
    async def connect(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
    ):
        """Connect to OpenAI's realtime API and configure the session"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
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
        response = await self.ws.recv()
        response_data = json.loads(response)
        if response_data["type"] == "session.created":
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
            # Without this, audio may arrive at the API before instructions
            # are applied, causing default Q&A behaviour on the first turn.
            try:
                confirmation = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
                confirmation_data = json.loads(confirmation)
                if confirmation_data.get("type") == "session.updated":
                    logger.info("OpenAI session.updated confirmed")
                else:
                    logger.warning(
                        "Expected session.updated, got %s — instructions may not be active yet. Data: %s",
                        confirmation_data.get("type"),
                        json.dumps(confirmation_data, ensure_ascii=False)[:500],
                    )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for session.updated confirmation")

        # Register the default handler
        self.register_handler("default", self.default_handler)

        # Start the receiver coroutine
        self.receive_task = asyncio.create_task(self.receive_messages())

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
        # Use an Event + temporary handler to wait for session.updated,
        # because receive_messages is already consuming from the WebSocket.
        session_updated_event = asyncio.Event()
        original_handler = self.handlers.get("session.updated")

        async def _on_session_updated(data):
            session_updated_event.set()
            # Restore original handler
            if original_handler:
                self.handlers["session.updated"] = original_handler
                await original_handler(data)
            else:
                self.handlers.pop("session.updated", None)

        self.handlers["session.updated"] = _on_session_updated

        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": session_config_payload,
        }, ensure_ascii=False))
        logger.info("Refreshed OpenAI session configuration via session.update")

        # Wait for session.updated confirmation so instructions are active
        # before any audio is forwarded to the provider.
        try:
            await asyncio.wait_for(session_updated_event.wait(), timeout=5.0)
            logger.info("OpenAI session.updated confirmed (refresh)")
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for session.updated after refresh")
            # Restore handler on timeout
            if original_handler:
                self.handlers["session.updated"] = original_handler
            else:
                self.handlers.pop("session.updated", None)

    
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
        if self._is_ws_open():
            commit_message = json.dumps({"type": "input_audio_buffer.commit"})
            await self.ws.send(commit_message)
            logger.info("Sent input_audio_buffer.commit message to OpenAI")
        else:
            raise ConnectionError("WebSocket is not open. Cannot commit audio.")
    
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
        if self._is_ws_open():
            response_config = {
                "modalities": ["text"],
                "temperature": 0.6,  # OpenAI Realtime API 最小值是 0.6
            }
            if self.include_instructions_each_response and instructions:
                response_config["instructions"] = instructions

            await self.ws.send(json.dumps({
                "type": "response.create",
                "response": response_config
            }))
            logger.info(
                "Started response with temperature=0.6"
                + (
                    " (with per-response instructions)"
                    if self.include_instructions_each_response
                    else " (session instructions only)"
                )
            )
        else:
            raise ConnectionError("WebSocket is not open. Cannot start response.")
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            logger.info("Closed OpenAI WebSocket connection")
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
