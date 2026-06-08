import asyncio
import collections
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.signal
import uvicorn
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState

from config import (
    OPENAI_REALTIME_MODEL,
    OPENAI_REALTIME_MODALITIES,
    XAI_API_KEY,
    XAI_REALTIME_MODALITIES,
    REALTIME_PROVIDER,
)
from audio_persistence import TurnAudioCache
from openai_realtime_client import OpenAIRealtimeAudioTextClient
from prompts import get_optimize_prompt
from realtime_client_base import RealtimeClientBase
from realtime_text_utils import StreamingHomonymCorrector, extract_text_after_marker
from transcript_merge import merge_incremental_text
from xai_realtime_client import XAIRealtimeAudioTextClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
VERBOSE_SERVER_LOG = os.getenv("BRAINWAVE_VERBOSE_SERVER_LOG", "0") == "1"

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class AudioProcessor:
    def __init__(self, target_sample_rate=24000, source_sample_rate=48000):
        self.target_sample_rate = target_sample_rate
        self.source_sample_rate = source_sample_rate  # Most common sample rate for microphones

    def set_source_sample_rate(self, sample_rate: int):
        self.source_sample_rate = sample_rate
        
    def process_audio_chunk(self, audio_data):
        if self.source_sample_rate == self.target_sample_rate:
            return audio_data
        # Convert binary audio data to Int16 array
        pcm_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to float32 for better precision during resampling
        float_data = pcm_data.astype(np.float32) / 32768.0
        
        # Resample from 48kHz to 24kHz
        resampled_data = scipy.signal.resample_poly(
            float_data, 
            self.target_sample_rate, 
            self.source_sample_rate
        )
        
        # Convert back to int16 while preserving amplitude
        resampled_int16 = (resampled_data * 32768.0).clip(-32768, 32767).astype(np.int16)
        return resampled_int16.tobytes()


@dataclass(frozen=True)
class TurnSessionConfig:
    keep_provider_session: bool
    provider_session_max_turns: int
    provider_session_max_age_sec: int
    provider_init_max_attempts: int
    provider_init_retry_delay_sec: float
    response_finalize_timeout_sec: float
    input_transcript_grace_sec: float
    default_source_sample_rate: int
    marker_prefix: str
    max_prefix_deltas: int
    transcription_failure_rotate_threshold: int
    passthrough_without_marker: bool

    @classmethod
    def from_env(cls) -> "TurnSessionConfig":
        def int_env(name: str, default: int, minimum: int) -> int:
            raw = os.getenv(name, str(default))
            try:
                return max(minimum, int(raw))
            except ValueError:
                logger.warning("Invalid %s=%r, falling back to %s", name, raw, default)
                return default

        def float_env(name: str, default: float, minimum: float) -> float:
            raw = os.getenv(name, str(default))
            try:
                return max(minimum, float(raw))
            except ValueError:
                logger.warning("Invalid %s=%r, falling back to %s", name, raw, default)
                return default

        return cls(
            keep_provider_session=os.getenv("BRAINWAVE_KEEP_PROVIDER_SESSION", "1") == "1",
            provider_session_max_turns=int_env("BRAINWAVE_PROVIDER_SESSION_MAX_TURNS", 8, 0),
            provider_session_max_age_sec=int_env("BRAINWAVE_PROVIDER_SESSION_MAX_AGE_SEC", 7200, 0),
            provider_init_max_attempts=int_env("BRAINWAVE_PROVIDER_INIT_MAX_ATTEMPTS", 3, 1),
            provider_init_retry_delay_sec=float_env("BRAINWAVE_PROVIDER_INIT_RETRY_DELAY_SEC", 0.5, 0.0),
            response_finalize_timeout_sec=float_env("BRAINWAVE_RESPONSE_FINALIZE_TIMEOUT_SEC", 120.0, 5.0),
            input_transcript_grace_sec=float_env("BRAINWAVE_INPUT_TRANSCRIPT_GRACE_SEC", 1.2, 0.0),
            default_source_sample_rate=AudioProcessor().source_sample_rate,
            marker_prefix="下面是不改变语言的语音识别结果：\n\n",
            max_prefix_deltas=20,
            transcription_failure_rotate_threshold=2,
            passthrough_without_marker=os.getenv("BRAINWAVE_PASSTHROUGH_WITHOUT_MARKER", "0") == "1",
        )


class TurnSession:
    _RECONNECT_MAX_ATTEMPTS = 3
    _RECONNECT_BACKOFF_BASE_SEC = 1.0
    _FLAP_WINDOW_SEC = 60.0
    _FLAP_THRESHOLD = 3
    _FLAP_COOLDOWN_SEC = 120.0

    _SUCCESSFUL_FINALIZE_REASONS = {
        "response.done",
        "response.text.done",
        "response.output_text.done",
        "response.output_audio_transcript.done",
    }

    def __init__(self, websocket: WebSocket, config: TurnSessionConfig):
        self._websocket = websocket
        self._config = config

        self._client: Optional[RealtimeClientBase] = None
        self._active_provider: Optional[str] = None
        self._active_model: Optional[str] = None
        self._provider_session_turns = 0
        self._provider_session_started_at: Optional[float] = None
        self._consecutive_transcription_failures = 0
        self._openai_ready = asyncio.Event()
        self._audio_processor = AudioProcessor(source_sample_rate=config.default_source_sample_rate)
        self._audio_cache = TurnAudioCache.from_env()

        self._reconnect_lock = asyncio.Lock()
        self._reconnect_attempts = 0
        self._disconnect_timestamps: collections.deque = collections.deque()
        self._flap_cooldown_until: Optional[float] = None

        self._active_turn_id: Optional[int] = None
        self._finalized = False
        self._is_recording = False
        self._turn_done: Optional[asyncio.Event] = None
        self._input_transcript_done: Optional[asyncio.Event] = None
        self._pending_audio_chunks: list[bytes] = []
        self._response_buffer: list[str] = []
        self._marker_seen = False
        self._delta_counter = 0
        self._emitted_text = ""
        self._input_transcript_text = ""
        self._input_transcript_seen = False
        self._homonym_corrector = StreamingHomonymCorrector()

        self._closed = False
        self._reset_turn_state()

    @staticmethod
    def _normalize_turn_id(raw_turn_id) -> Optional[int]:
        if raw_turn_id is None:
            return None
        try:
            return int(raw_turn_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_input_transcription(data: dict) -> str:
        def _extract_from_object(obj) -> str:
            if not isinstance(obj, dict):
                return ""
            for key in ("transcript", "text", "delta"):
                value = obj.get(key)
                if isinstance(value, str) and value:
                    return value
            content = obj.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    for key in ("transcript", "text", "delta"):
                        value = part.get(key)
                        if isinstance(value, str) and value:
                            parts.append(value)
                if parts:
                    return "".join(parts)
            return ""

        text = _extract_from_object(data)
        if text:
            return text
        return _extract_from_object(data.get("item"))

    @staticmethod
    def _resolve_provider(provider: Optional[str], model: Optional[str]) -> str:
        if provider in {"openai", "xai"}:
            return provider
        if model and (model.startswith("grok-") or model in {"xai", "xai-grok"}):
            return "xai"
        return "openai"

    def _reset_turn_state(self, active_turn_id: Optional[int] = None):
        self._active_turn_id = active_turn_id
        self._finalized = False
        self._is_recording = False
        self._turn_done = asyncio.Event()
        self._input_transcript_done = asyncio.Event()
        self._pending_audio_chunks = []
        self._response_buffer = []
        self._marker_seen = False
        self._delta_counter = 0
        self._emitted_text = ""
        self._input_transcript_text = ""
        self._input_transcript_seen = False
        self._homonym_corrector = StreamingHomonymCorrector()

    async def _send_text_payload(self, content: str):
        if content and self._websocket.client_state == WebSocketState.CONNECTED:
            payload = {
                "type": "text",
                "content": content,
                "isNewResponse": False
            }
            if self._active_turn_id is not None:
                payload["turn_id"] = self._active_turn_id
            await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
            self._emitted_text = merge_incremental_text(self._emitted_text, content)

    async def _emit_text_delta(self, content: str):
        safe = self._homonym_corrector.push(content)
        if safe:
            await self._send_text_payload(safe)

    async def _flush_homonym_corrector(self):
        tail = self._homonym_corrector.flush()
        if tail:
            await self._send_text_payload(tail)

    async def _flush_buffer(self, with_warning: bool = False) -> bool:
        if not self._response_buffer:
            return False
        buffered_text = "".join(self._response_buffer)
        self._response_buffer = []
        found_marker, buffered_text = extract_text_after_marker(
            buffered_text,
            self._config.marker_prefix,
        )
        if not found_marker:
            if self._config.passthrough_without_marker:
                await self._emit_text_delta(buffered_text)
                return True
            if with_warning:
                logger.warning("Marker prefix not detected; dropping buffered text.")
            return False
        if with_warning and not buffered_text:
            logger.warning("Buffered text discarded after removing marker prefix.")
        await self._emit_text_delta(buffered_text)
        return True

    async def _finalize_turn(self, reason: str):
        done = self._turn_done
        if self._finalized:
            return
        self._finalized = True
        self._is_recording = False
        self._pending_audio_chunks.clear()
        if done:
            done.set()
        try:
            self._audio_cache.enqueue_turn(
                turn_id=self._active_turn_id,
                outcome=reason,
                sample_rate=self._audio_processor.target_sample_rate,
            )
        except Exception as e:
            logger.error(f"Error enqueueing audio cache on finalize ({reason}): {e}", exc_info=True)
        try:
            await self._flush_homonym_corrector()
        except Exception as e:
            logger.error(f"Error flushing homonym corrector on finalize ({reason}): {e}", exc_info=True)
        emitted_len = len(self._emitted_text.strip()) if self._emitted_text else 0
        if emitted_len < 10 and reason in self._SUCCESSFUL_FINALIZE_REASONS:
            logger.warning(
                "Suspicious short transcription on finalize (%s): "
                "emitted_text_len=%d, marker_seen=%s, input_transcript_len=%d",
                reason,
                emitted_len,
                self._marker_seen,
                len(self._input_transcript_text.strip()),
            )
        try:
            payload = {
                "type": "status",
                "status": "idle"
            }
            if self._active_turn_id is not None:
                payload["turn_id"] = self._active_turn_id
            await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Error sending status after {reason}: {str(e)}", exc_info=True)

        if (
            reason in self._SUCCESSFUL_FINALIZE_REASONS
            and self._flap_cooldown_until is not None
            and time.monotonic() >= self._flap_cooldown_until
        ):
            self._flap_cooldown_until = None

        if not self._client:
            self._active_provider = None
            self._active_model = None
            self._active_turn_id = None
            self._provider_session_turns = 0
            self._provider_session_started_at = None
            self._openai_ready.clear()
            return

        can_keep_session = (
            self._config.keep_provider_session
            and reason in self._SUCCESSFUL_FINALIZE_REASONS
        )
        if can_keep_session:
            logger.info(f"Finalizing turn ({reason}), keeping provider client session alive")
            try:
                await self._client.clear_audio_buffer()
            except Exception as e:
                logger.warning(
                    f"Failed to clear provider audio buffer on finalize ({reason}): {e}"
                )
            return

        logger.info(f"Finalizing turn ({reason}), closing provider client")
        try:
            await self._client.close()
        except Exception as e:
            logger.error(f"Error closing client after {reason}: {str(e)}", exc_info=True)
        self._client = None
        self._active_provider = None
        self._active_model = None
        self._active_turn_id = None
        self._provider_session_turns = 0
        self._provider_session_started_at = None
        self._openai_ready.clear()

    async def _create_client(
        self,
        provider: str = None,
        model: str = None,
    ) -> RealtimeClientBase:
        """
        Factory function to create appropriate realtime client.
        
        Args:
            provider: Provider name ("openai" or "xai"). Defaults to REALTIME_PROVIDER config.
            model: Model name (for OpenAI). Defaults to OPENAI_REALTIME_MODEL.
                      For x.ai, use "xai-grok", "xai", or any model name starting with "grok-".
        
        Returns:
            RealtimeClientBase instance
        """
        provider = provider or REALTIME_PROVIDER
        
        if provider == "xai":
            api_key = XAI_API_KEY
            if not api_key:
                raise ValueError("XAI_API_KEY not set in environment variables")
            logger.info("Creating x.ai client (text-only mode, no voice needed)")
            return XAIRealtimeAudioTextClient(api_key)
        else:  # default to openai
            api_key = OPENAI_API_KEY
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set in environment variables")
            selected_model = model or OPENAI_REALTIME_MODEL
            logger.info(f"Creating OpenAI client with model: {selected_model}")
            return OpenAIRealtimeAudioTextClient(api_key, model=selected_model)

    async def _init_or_reuse_client(
        self,
        provider: str = None,
        model: str = None,
        voice: str = None,
        instructions: Optional[str] = None,
        turn_id: Optional[int] = None,
    ):
        provider_name = provider or REALTIME_PROVIDER
        requested_model = (model or OPENAI_REALTIME_MODEL) if provider_name == "openai" else None
        max_attempts = self._config.provider_init_max_attempts
        retry_delay_sec = self._config.provider_init_retry_delay_sec

        can_reuse_existing_client = (
            self._config.keep_provider_session
            and self._client
            and self._openai_ready.is_set()
            and self._active_provider == provider_name
            and self._active_model == requested_model
            and self._client._is_ws_open()
        )
        turn_limit_reached = (
            self._config.provider_session_max_turns > 0
            and self._provider_session_turns >= self._config.provider_session_max_turns
        )
        session_age_sec = None
        if self._provider_session_started_at is not None:
            session_age_sec = max(0.0, time.time() - self._provider_session_started_at)
        age_limit_reached = (
            self._config.provider_session_max_age_sec > 0
            and session_age_sec is not None
            and session_age_sec >= self._config.provider_session_max_age_sec
        )

        transcription_failure_limit_reached = (
            self._consecutive_transcription_failures
            >= self._config.transcription_failure_rotate_threshold
        )

        if can_reuse_existing_client and turn_limit_reached:
            logger.info(
                "Provider session reached max turns (%d/%d), rotating session",
                self._provider_session_turns,
                self._config.provider_session_max_turns,
            )
        if can_reuse_existing_client and age_limit_reached:
            logger.info(
                "Provider session reached max age (%.0fs/%.0f), rotating session",
                session_age_sec or 0.0,
                float(self._config.provider_session_max_age_sec),
            )
        if can_reuse_existing_client and transcription_failure_limit_reached:
            logger.warning(
                "Input audio transcription failed %d consecutive times "
                "(threshold=%d), forcing session rotation",
                self._consecutive_transcription_failures,
                self._config.transcription_failure_rotate_threshold,
            )
            self._consecutive_transcription_failures = 0

        if (
            can_reuse_existing_client
            and not turn_limit_reached
            and not age_limit_reached
            and not transcription_failure_limit_reached
        ):
            logger.info(
                f"Reusing existing {provider_name} realtime session"
                + (f" ({requested_model})" if requested_model else "")
            )
            try:
                if provider_name == "xai":
                    await self._client.refresh_session(
                        modalities=XAI_REALTIME_MODALITIES,
                        instructions=instructions,
                    )
                else:
                    await self._client.refresh_session(
                        modalities=OPENAI_REALTIME_MODALITIES,
                        instructions=instructions,
                    )
            except Exception as refresh_err:
                logger.warning(
                    "Failed to refresh reused %s session (%s). Recreating session.",
                    provider_name,
                    refresh_err,
                )
                try:
                    await self._client.close()
                except Exception as close_err:
                    logger.error(f"Error closing client after refresh failure: {close_err}")
                self._client = None
                self._active_provider = None
                self._active_model = None
                self._provider_session_turns = 0
                self._provider_session_started_at = None
                self._openai_ready.clear()
            else:
                payload = {
                    "type": "status",
                    "status": "connected"
                }
                if turn_id is not None:
                    payload["turn_id"] = turn_id
                await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
                self._reconnect_attempts = 0
                return True

        # Clear the ready flag while initializing
        self._openai_ready.clear()

        for attempt in range(1, max_attempts + 1):
            try:
                # Close previous client if it wasn't properly finalized
                if self._client:
                    logger.warning("Previous client still exists during re-init, closing it")
                    try:
                        await self._client.close()
                    except Exception as e:
                        logger.error(f"Error closing stale client: {e}")
                    self._client = None
                    self._active_provider = None
                    self._active_model = None
                    self._provider_session_turns = 0
                    self._provider_session_started_at = None

                # Create client using factory function
                self._client = await self._create_client(provider=provider, model=model)
                
                # Pass appropriate modalities based on provider
                if provider_name == "xai":
                    await self._client.connect(
                        modalities=XAI_REALTIME_MODALITIES,
                        instructions=instructions,
                    )
                else:
                    await self._client.connect(
                        modalities=OPENAI_REALTIME_MODALITIES,
                        instructions=instructions,
                    )
                
                logger.info(f"Successfully connected to {provider_name} client (attempt {attempt}/{max_attempts})")
                
                # Register handlers after client is initialized
                self._client.register_handler("session.updated", lambda data: self._on_generic_event("session.updated", data))
                self._client.register_handler("input_audio_buffer.cleared", lambda data: self._on_generic_event("input_audio_buffer.cleared", data))
                self._client.register_handler("input_audio_buffer.speech_started", lambda data: self._on_generic_event("input_audio_buffer.speech_started", data))
                self._client.register_handler("rate_limits.updated", lambda data: self._on_generic_event("rate_limits.updated", data))
                self._client.register_handler("response.output_item.added", lambda data: self._on_generic_event("response.output_item.added", data))
                self._client.register_handler("conversation.item.created", lambda data: self._on_generic_event("conversation.item.created", data))
                self._client.register_handler("response.content_part.added", lambda data: self._on_generic_event("response.content_part.added", data))
                self._client.register_handler("response.text.done", self._on_response_text_done)
                self._client.register_handler("response.output_text.done", self._on_response_text_done)
                self._client.register_handler("response.content_part.done", lambda data: self._on_generic_event("response.content_part.done", data))
                self._client.register_handler("response.output_item.done", lambda data: self._on_generic_event("response.output_item.done", data))
                self._client.register_handler("response.done", self._on_response_done)
                self._client.register_handler("error", self._on_error)
                self._client.register_handler("response.text.delta", self._on_text_delta)
                self._client.register_handler("response.output_text.delta", self._on_text_delta)
                # x.ai uses response.output_audio_transcript.delta instead of response.text.delta
                self._client.register_handler("response.output_audio_transcript.delta", self._on_text_delta)
                self._client.register_handler("response.created", self._on_response_created)
                # x.ai specific message types
                self._client.register_handler("input_audio_buffer.speech_stopped", lambda data: self._on_generic_event("input_audio_buffer.speech_stopped", data))
                self._client.register_handler("input_audio_buffer.committed", lambda data: self._on_generic_event("input_audio_buffer.committed", data))
                self._client.register_handler("conversation.item.added", lambda data: self._on_generic_event("conversation.item.added", data))
                self._client.register_handler(
                    "conversation.item.input_audio_transcription.delta",
                    self._on_input_transcription_delta,
                )
                self._client.register_handler(
                    "conversation.item.input_audio_transcription.completed",
                    self._on_input_transcription_completed,
                )
                self._client.register_handler(
                    "conversation.item.input_audio_transcription.failed",
                    self._on_input_transcription_failed,
                )
                self._client.register_handler("response.output_audio_transcript.done", self._on_output_audio_transcript_done)
                self._client.register_handler("response.output_audio.delta", lambda data: self._on_generic_event("response.output_audio.delta", data))
                self._client.register_handler("response.output_audio.done", lambda data: self._on_generic_event("response.output_audio.done", data))
                self._client.register_handler("ping", lambda data: self._on_generic_event("ping", data))

                # Auto-reconnect when the provider WS drops unexpectedly
                self._client.set_on_disconnect(self._on_provider_disconnect)

                self._openai_ready.set()  # Set ready flag after successful initialization
                self._active_provider = provider_name
                self._active_model = requested_model
                self._provider_session_turns = 0
                self._provider_session_started_at = time.time()
                self._reconnect_attempts = 0
                payload = {
                    "type": "status",
                    "status": "connected"
                }
                if turn_id is not None:
                    payload["turn_id"] = turn_id
                await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
                return True
            except Exception as e:
                logger.error(
                    f"Failed to connect to {provider_name} client "
                    f"(attempt {attempt}/{max_attempts}): {e}"
                )
                self._openai_ready.clear()  # Ensure flag is cleared on failure
                if self._client:
                    try:
                        await self._client.close()
                    except Exception as close_err:
                        logger.error(f"Error closing failed client: {close_err}")
                    self._client = None
                    self._active_provider = None
                    self._active_model = None
                    self._provider_session_turns = 0
                    self._provider_session_started_at = None
                if attempt < max_attempts:
                    await asyncio.sleep(retry_delay_sec)
                    continue
                payload = {
                    "type": "error",
                    "content": f"Failed to initialize {provider_name} realtime connection"
                }
                if turn_id is not None:
                    payload["turn_id"] = turn_id
                await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
                return False

    async def _on_provider_disconnect(self):
        if self._finalized:
            return

        logger.warning("Provider WebSocket disconnected unexpectedly")
        now = time.monotonic()
        self._disconnect_timestamps.append(now)
        while (
            self._disconnect_timestamps
            and self._disconnect_timestamps[0] < now - self._FLAP_WINDOW_SEC
        ):
            self._disconnect_timestamps.popleft()
        if len(self._disconnect_timestamps) >= self._FLAP_THRESHOLD:
            if self._flap_cooldown_until is None or now >= self._flap_cooldown_until:
                self._flap_cooldown_until = now + self._FLAP_COOLDOWN_SEC
            if now < self._flap_cooldown_until:
                logger.error(
                    "Provider reconnect suppressed by flapping cooldown "
                    "(disconnects=%d cooldown_remaining=%.1fs)",
                    len(self._disconnect_timestamps),
                    self._flap_cooldown_until - now,
                )
                await self._finalize_turn("flapping_cooldown")
                return

        if self._reconnect_lock.locked():
            logger.info("Reconnect already in progress; ignoring duplicate disconnect")
            return

        async with self._reconnect_lock:
            if self._finalized:
                return
            if self._reconnect_attempts >= self._RECONNECT_MAX_ATTEMPTS:
                await self._finalize_turn("reconnect_exhausted")
                return
            self._reconnect_attempts += 1
            backoff = self._RECONNECT_BACKOFF_BASE_SEC * (2 ** (self._reconnect_attempts - 1))
            self._openai_ready.clear()  # audio will be buffered to pending_audio_chunks

            if self._client:
                try:
                    await asyncio.wait_for(self._client.close(), timeout=5.0)
                except (asyncio.TimeoutError, Exception):
                    pass
                self._client = None

            await asyncio.sleep(backoff)

            if not self._is_recording:
                if not self._finalized:
                    await self._finalize_turn("provider_disconnect_during_response")
                return

            if not self._active_provider:
                logger.error("Provider disconnected with no active provider; finalizing turn")
                await self._finalize_turn("provider_unavailable")
                return

            logger.info("Recording active during disconnect; auto-reconnecting")
            ok = await self._init_or_reuse_client(
                provider=self._active_provider,
                model=self._active_model,
                turn_id=self._active_turn_id,
            )
            if ok and self._client:
                self._reconnect_attempts = 0
                if self._pending_audio_chunks:
                    chunks = list(self._pending_audio_chunks)
                    self._pending_audio_chunks.clear()
                    logger.info(
                        "Auto-reconnected, flushing %d buffered audio chunks",
                        len(chunks),
                    )
                    for chunk in chunks:
                        await self._client.send_audio(chunk)
            elif not ok:
                logger.error("Auto-reconnect failed, audio buffered for next attempt")

    async def _on_text_delta(self, data):
        try:
            if self._websocket.client_state != WebSocketState.CONNECTED:
                logger.warning("WebSocket not connected, ignoring text delta")
                return
            delta = data.get("delta", "")
            logger.debug(
                "Received text delta: %r (marker_seen=%s, buffer_size=%d, delta_counter=%d)",
                delta[:50],
                self._marker_seen,
                len(self._response_buffer),
                self._delta_counter,
            )

            if self._marker_seen:
                if delta:
                    await self._emit_text_delta(delta)
                    logger.debug(f"Handled response.text.delta (passthrough): {repr(delta[:50])}")
                return

            if not delta:
                return

            self._response_buffer.append(delta)
            self._delta_counter += 1

            joined = "".join(self._response_buffer)
            found_marker, remaining = extract_text_after_marker(joined, self._config.marker_prefix)
            if found_marker:
                self._marker_seen = True
                self._response_buffer = []
                if remaining:
                    await self._emit_text_delta(remaining)
                logger.debug(
                    f"Handled response.text.delta (marker stripped), emitted: {repr(remaining[:50])}"
                )
                return

            if self._config.passthrough_without_marker:
                # Keep a short tail for cross-chunk marker matching; pass through the rest.
                keep_tail = max(0, len(self._config.marker_prefix) - 1)
                if len(joined) > keep_tail:
                    passthrough_text = joined[:-keep_tail] if keep_tail else joined
                    self._response_buffer = [joined[-keep_tail:]] if keep_tail else []
                    if passthrough_text:
                        await self._emit_text_delta(passthrough_text)
                    logger.debug(
                        "Handled response.text.delta (safe passthrough), "
                        f"emitted_len={len(passthrough_text)} keep_tail={keep_tail}"
                    )
                return

            if self._delta_counter >= self._config.max_prefix_deltas:
                buffered = "".join(self._response_buffer)
                self._response_buffer = []
                self._delta_counter = 0
                self._marker_seen = True
                if buffered:
                    await self._emit_text_delta(buffered)
                logger.warning(
                    "Marker prefix not detected after %d deltas; emitting as-is (len=%d).",
                    self._config.max_prefix_deltas,
                    len(buffered),
                )
            else:
                logger.debug(f"Handled response.text.delta (buffering), total buffer length: {len(joined)}")
        except Exception as e:
            logger.error(f"Error in handle_text_delta: {str(e)}", exc_info=True)

    async def _on_input_transcription_delta(self, data):
        try:
            delta = self._extract_input_transcription(data)
            if not delta:
                return
            self._input_transcript_seen = True
            self._input_transcript_text = merge_incremental_text(
                self._input_transcript_text,
                delta,
            )
        except Exception as e:
            logger.error(f"Error in handle_input_audio_transcription_delta: {e}", exc_info=True)

    async def _on_input_transcription_completed(self, data):
        try:
            completed_text = self._extract_input_transcription(data)
            self._input_transcript_seen = True
            self._consecutive_transcription_failures = 0
            if completed_text and len(completed_text) >= len(self._input_transcript_text):
                self._input_transcript_text = completed_text
            logger.info(
                "Handled conversation.item.input_audio_transcription.completed "
                "(len=%d)",
                len(self._input_transcript_text),
            )
        except Exception as e:
            logger.error(f"Error in handle_input_audio_transcription_completed: {e}", exc_info=True)
        finally:
            if self._input_transcript_done:
                self._input_transcript_done.set()

    async def _on_input_transcription_failed(self, data):
        error_info = data.get("error", {})
        error_type = error_info.get("type", "unknown")
        error_code = error_info.get("code", "unknown")
        error_message = error_info.get("message", "no message")
        self._consecutive_transcription_failures += 1
        logger.error(
            "Input audio transcription FAILED (type=%s, code=%s, message=%s, "
            "consecutive_failures=%d/%d)",
            error_type,
            error_code,
            error_message,
            self._consecutive_transcription_failures,
            self._config.transcription_failure_rotate_threshold,
        )
        # Unblock grace wait so finalize_turn doesn't wait 1.2s for nothing
        if self._input_transcript_done:
            self._input_transcript_done.set()

    async def _apply_input_transcription_fallback(self, event_type: str):
        if self._active_provider != "openai":
            return
        done = self._input_transcript_done
        # Only wait for input transcription when marker_seen is False
        # (model may have answered instead of transcribing).  When the
        # marker was seen the model followed the transcription format,
        # so the fallback is very unlikely to be needed and we skip the
        # grace wait to avoid adding ~1.2 s of unnecessary latency.
        if done and not done.is_set() and self._config.input_transcript_grace_sec > 0:
            if not self._marker_seen:
                try:
                    await asyncio.wait_for(
                        done.wait(),
                        timeout=self._config.input_transcript_grace_sec,
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "Input transcription grace wait timed out (%.2fs) on %s",
                        self._config.input_transcript_grace_sec,
                        event_type,
                    )
            else:
                logger.info(
                    "Skipping input transcription grace wait (marker_seen=True) on %s",
                    event_type,
                )

        fallback_text = self._input_transcript_text.strip()
        if not fallback_text:
            return
        current_text = self._emitted_text.strip()

        # When marker_seen is False the model did NOT follow the
        # transcription-only format (likely answered the user's speech
        # instead of transcribing it).  In that case, unconditionally
        # replace the emitted text with the input transcription from
        # gpt-4o-transcribe, which is always a faithful transcription.
        if not self._marker_seen and current_text:
            logger.warning(
                "Marker not seen — model likely answered instead of "
                "transcribing. Replacing emitted text with input "
                "transcription (emitted_len=%d, fallback_len=%d)",
                len(current_text),
                len(fallback_text),
            )
        elif self._marker_seen and current_text:
            # Model followed the transcription format and produced output;
            # trust the model over input_audio_transcription (which can
            # return garbled text for very short utterances).
            return
        elif current_text and (
            fallback_text == current_text
            or len(fallback_text) <= len(current_text) + 1
        ):
            return

        # About to replace the whole stream via isNewResponse=True; drop the
        # corrector's held tail so finalize_turn's flush won't append it.
        self._homonym_corrector.reset()

        payload = {
            "type": "text",
            "content": fallback_text,
            "isNewResponse": True,
        }
        if self._active_turn_id is not None:
            payload["turn_id"] = self._active_turn_id
        await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
        self._emitted_text = fallback_text
        logger.info(
            "Applied input transcription fallback before finalize "
            "(fallback_len=%d current_len=%d marker_seen=%s)",
            len(fallback_text),
            len(current_text),
            self._marker_seen,
        )

    async def _on_response_created(self, data):
        if self._emitted_text:
            logger.warning(
                "response.created received but _emitted_text already has %d chars; "
                "preserving existing text and resetting buffer state only",
                len(self._emitted_text),
            )
            self._response_buffer = []
            self._marker_seen = False
            self._delta_counter = 0
            return
        self._response_buffer = []
        self._marker_seen = False
        self._delta_counter = 0
        self._emitted_text = ""
        logger.info("Handled response.created, clearing buffer and resetting marker state")

    async def _on_error(self, data):
        error_msg = data.get("error", {}).get("message", "Unknown error")
        logger.error(f"Provider error: {error_msg}")
        payload = {
            "type": "error",
            "content": error_msg
        }
        if self._active_turn_id is not None:
            payload["turn_id"] = self._active_turn_id
        await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
        await self._finalize_turn("error")
        logger.info("Handled error message from provider")

    async def _on_text_completed(self, event_type: str, data):
        logger.info(
            f"Handled {event_type} "
            f"(marker_seen={self._marker_seen}, buffer_size={len(self._response_buffer)})"
        )
        if not self._marker_seen and self._emitted_text:
            logger.warning(
                "Model output without marker prefix (possible Q&A instead of "
                "transcription): %s",
                repr(self._emitted_text[:120]),
            )
        if self._response_buffer:
            logger.info("Flushing remaining buffer content")
            flushed = await self._flush_buffer()
            if flushed:
                self._marker_seen = True
        await self._apply_input_transcription_fallback(event_type)
        await self._finalize_turn(event_type)

    async def _on_response_done(self, data):
        await self._on_text_completed("response.done", data)

    async def _on_response_text_done(self, data):
        await self._on_text_completed("response.text.done", data)

    async def _on_output_audio_transcript_done(self, data):
        await self._on_text_completed("response.output_audio_transcript.done", data)

    async def _on_generic_event(self, event_type, data):
        if VERBOSE_SERVER_LOG:
            logger.info(f"Handled {event_type} with data: {json.dumps(data, ensure_ascii=False)}")
        else:
            logger.debug(f"Handled {event_type}")

    async def _handle_start_recording(self, msg: dict):
        logger.info("Processing start_recording request")
        requested_turn_id = self._normalize_turn_id(msg.get("turn_id"))
        if msg.get("turn_id") is not None and requested_turn_id is None:
            logger.warning(f"Ignoring invalid turn_id from client: {msg.get('turn_id')!r}")
        self._reset_turn_state(requested_turn_id)

        # Update status to connecting while initializing realtime client
        status_payload = {
            "type": "status",
            "status": "connecting"
        }
        if self._active_turn_id is not None:
            status_payload["turn_id"] = self._active_turn_id
        await self._websocket.send_text(json.dumps(status_payload, ensure_ascii=False))
        # Extract provider and model from message
        provider = msg.get("provider")  # "openai" or "xai"
        model = msg.get("model")  # OpenAI model name
        # Optimized-only mode: use optimize prompt
        logger.info("Optimized-only mode: using optimize prompt")
        input_sample_rate = msg.get("input_sample_rate")

        logger.info(f"Received start_recording: provider={provider}, model={model}")

        provider = self._resolve_provider(provider, model)
        logger.info(f"Using provider: {provider}")
        instructions = get_optimize_prompt()

        if input_sample_rate:
            try:
                self._audio_processor.set_source_sample_rate(int(input_sample_rate))
                logger.info(f"Using input sample rate: {self._audio_processor.source_sample_rate}Hz")
            except (TypeError, ValueError):
                logger.warning(
                    f"Invalid input_sample_rate '{input_sample_rate}', "
                    f"using default {self._config.default_source_sample_rate}Hz"
                )
                self._audio_processor.set_source_sample_rate(self._config.default_source_sample_rate)
        else:
            self._audio_processor.set_source_sample_rate(self._config.default_source_sample_rate)

        if not await self._init_or_reuse_client(
            provider=provider,
            model=model,
            instructions=instructions,
            turn_id=self._active_turn_id,
        ):
            return
        self._provider_session_turns += 1
        if self._config.provider_session_max_turns > 0:
            logger.info(
                "Provider session turn started: %d/%d",
                self._provider_session_turns,
                self._config.provider_session_max_turns,
            )
        else:
            logger.info(
                "Provider session turn started: %d (max-turn rotation disabled)",
                self._provider_session_turns,
            )
        # Immediately clear transcript for a new client-initiated request
        reset_payload = {
            "type": "text",
            "content": "",
            "isNewResponse": True
        }
        if self._active_turn_id is not None:
            reset_payload["turn_id"] = self._active_turn_id
        await self._websocket.send_text(json.dumps(reset_payload, ensure_ascii=False))
        self._is_recording = True
        self._audio_cache.start_turn(self._active_turn_id)

        # Send any buffered chunks
        if self._pending_audio_chunks and self._client:
            logger.info(f"Sending {len(self._pending_audio_chunks)} buffered chunks")
            for chunk in self._pending_audio_chunks:
                await self._client.send_audio(chunk)
            self._pending_audio_chunks.clear()

    async def _handle_stop_recording(self, msg: dict):
        requested_turn_id = self._normalize_turn_id(msg.get("turn_id"))
        if (
            requested_turn_id is not None
            and self._active_turn_id is not None
            and requested_turn_id != self._active_turn_id
        ):
            logger.warning(
                "Ignoring stop_recording for stale turn_id=%s (active_turn_id=%s)",
                requested_turn_id,
                self._active_turn_id,
            )
            return

        # On explicit Stop, force-commit and force-create a response, then wait for completion.
        if self._client:
            done_event = self._turn_done
            # Immediately stop accepting further audio for this turn
            self._is_recording = False
            try:
                await self._client.commit_audio()
                logger.info("Audio committed, starting response...")
                # Use text-only modalities for x.ai if configured
                if isinstance(self._client, XAIRealtimeAudioTextClient):
                    modalities = XAI_REALTIME_MODALITIES
                    await self._client.start_response(get_optimize_prompt(), modalities=modalities)
                else:
                    # OpenAI: by default rely on session-level instructions
                    # (can be overridden via env flag on client side).
                    await self._client.start_response(get_optimize_prompt())
                logger.info("Response started successfully")
            except ConnectionError as e:
                logger.error(
                    f"Provider unavailable while committing/starting response on stop: {str(e)}",
                    exc_info=True,
                )
                await self._finalize_turn("provider_unavailable")
                return
            except Exception as e:
                logger.error(f"Error committing/starting response on stop: {str(e)}", exc_info=True)
                # If we fail to kick off a response, surface that we're no longer recording
                status_payload = {
                    "type": "status",
                    "status": "idle"
                }
                if self._active_turn_id is not None:
                    status_payload["turn_id"] = self._active_turn_id
                await self._websocket.send_text(json.dumps(status_payload, ensure_ascii=False))
                return
            if done_event is None:
                logger.error("Turn has no completion Event, forcing finalization")
                await self._finalize_turn("missing_turn_done_event")
                return
            # Wait until the response is finished; timeout is configurable for long turns.
            try:
                await asyncio.wait_for(
                    done_event.wait(),
                    timeout=self._config.response_finalize_timeout_sec,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Response timed out after %.1fs, forcing finalization",
                    self._config.response_finalize_timeout_sec,
                )
                await self._finalize_turn("timeout")

    async def run(self):
        logger.info("receive_messages task started")
        try:
            await self._websocket.send_text(json.dumps({
                "type": "status",
                "status": "idle"  # Set initial status to idle (blue)
            }, ensure_ascii=False))

            while True:
                if self._websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info("WebSocket client disconnected")
                    self._openai_ready.clear()
                    break
                    
                try:
                    # Add timeout to prevent infinite waiting
                    logger.debug("Waiting for message from client (timeout=30s)...")
                    data = await asyncio.wait_for(self._websocket.receive(), timeout=30.0)
                    logger.debug(f"Received data from client: {list(data.keys())}")
                except asyncio.CancelledError:
                    logger.info("Receive messages task cancelled")
                    raise
                except asyncio.TimeoutError:
                    logger.debug("No message received for 30 seconds")
                    continue
                except Exception as e:
                    logger.error(f"Error receiving message: {str(e)}", exc_info=True)
                    break
                
                if "bytes" in data:
                    processed_audio = self._audio_processor.process_audio_chunk(data["bytes"])
                    if self._is_recording:
                        self._audio_cache.accumulate(processed_audio)
                    if not self._openai_ready.is_set():
                        logger.debug("Provider not ready, buffering audio chunk")
                        self._pending_audio_chunks.append(processed_audio)
                    elif self._client and self._is_recording:
                        await self._client.send_audio(processed_audio)
                        logger.debug(f"Sent audio chunk, size: {len(processed_audio)} bytes")
                    else:
                        logger.debug("Received audio but client is not initialized")
                            
                elif "text" in data:
                    msg = json.loads(data["text"])
                    logger.debug(f"Received message from client: {msg.get('type')}")
                    
                    if msg.get("type") == "start_recording":
                        await self._handle_start_recording(msg)
                    elif msg.get("type") == "stop_recording":
                        await self._handle_stop_recording(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
        finally:
            logger.info("Receive messages loop ended")

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.error(f"Error closing client in TurnSession.close: {str(e)}")
            self._client = None
        try:
            self._audio_cache.close()
        except Exception as e:
            logger.warning(f"Ignoring error closing audio_cache: {e}")
        if self._websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await self._websocket.close()
            except RuntimeError as e:
                logger.warning(f"Ignoring error during websocket close: {e}")
        logger.info("WebSocket connection closed for /api/v1/ws")


@app.websocket("/ws")
@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    config = TurnSessionConfig.from_env()
    session = TurnSession(websocket, config)
    try:
        await session.run()
    finally:
        await session.close()

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=23456)
