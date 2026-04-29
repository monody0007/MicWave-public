import asyncio
import json
import logging
import os
import time
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


@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info("New WebSocket connection attempt")
    await websocket.accept()
    logger.info("WebSocket connection accepted, starting receive_messages task")
    
    # Add initial status update here
    await websocket.send_text(json.dumps({
        "type": "status",
        "status": "idle"  # Set initial status to idle (blue)
    }, ensure_ascii=False))
    
    client = None
    active_provider = None
    active_model = None
    active_turn_id = None
    keep_provider_session = os.getenv("BRAINWAVE_KEEP_PROVIDER_SESSION", "1") == "1"
    provider_session_max_turns_raw = os.getenv("BRAINWAVE_PROVIDER_SESSION_MAX_TURNS", "8")
    try:
        provider_session_max_turns = max(0, int(provider_session_max_turns_raw))
    except ValueError:
        logger.warning(
            "Invalid BRAINWAVE_PROVIDER_SESSION_MAX_TURNS=%r, falling back to 8",
            provider_session_max_turns_raw,
        )
        provider_session_max_turns = 8
    provider_session_max_age_sec_raw = os.getenv("BRAINWAVE_PROVIDER_SESSION_MAX_AGE_SEC", "7200")
    try:
        provider_session_max_age_sec = max(0, int(provider_session_max_age_sec_raw))
    except ValueError:
        logger.warning(
            "Invalid BRAINWAVE_PROVIDER_SESSION_MAX_AGE_SEC=%r, falling back to 7200",
            provider_session_max_age_sec_raw,
        )
        provider_session_max_age_sec = 7200
    provider_init_max_attempts_raw = os.getenv("BRAINWAVE_PROVIDER_INIT_MAX_ATTEMPTS", "3")
    try:
        provider_init_max_attempts = max(1, int(provider_init_max_attempts_raw))
    except ValueError:
        logger.warning(
            "Invalid BRAINWAVE_PROVIDER_INIT_MAX_ATTEMPTS=%r, falling back to 3",
            provider_init_max_attempts_raw,
        )
        provider_init_max_attempts = 3
    provider_init_retry_delay_raw = os.getenv("BRAINWAVE_PROVIDER_INIT_RETRY_DELAY_SEC", "0.5")
    try:
        provider_init_retry_delay_sec = max(0.0, float(provider_init_retry_delay_raw))
    except ValueError:
        logger.warning(
            "Invalid BRAINWAVE_PROVIDER_INIT_RETRY_DELAY_SEC=%r, falling back to 0.5",
            provider_init_retry_delay_raw,
        )
        provider_init_retry_delay_sec = 0.5
    response_finalize_timeout_raw = os.getenv("BRAINWAVE_RESPONSE_FINALIZE_TIMEOUT_SEC", "120")
    try:
        response_finalize_timeout_sec = max(5.0, float(response_finalize_timeout_raw))
    except ValueError:
        logger.warning(
            "Invalid BRAINWAVE_RESPONSE_FINALIZE_TIMEOUT_SEC=%r, falling back to 120",
            response_finalize_timeout_raw,
        )
        response_finalize_timeout_sec = 120.0
    input_transcript_grace_raw = os.getenv("BRAINWAVE_INPUT_TRANSCRIPT_GRACE_SEC", "1.2")
    try:
        input_transcript_grace_sec = max(0.0, float(input_transcript_grace_raw))
    except ValueError:
        logger.warning(
            "Invalid BRAINWAVE_INPUT_TRANSCRIPT_GRACE_SEC=%r, falling back to 1.2",
            input_transcript_grace_raw,
        )
        input_transcript_grace_sec = 1.2
    provider_session_turns = 0
    provider_session_started_at = None
    audio_processor = AudioProcessor()
    default_source_sample_rate = audio_processor.source_sample_rate
    recording_stopped = asyncio.Event()
    openai_ready = asyncio.Event()
    pending_audio_chunks = []
    is_recording = False
    finalized = False
    marker_prefix = "下面是不改变语言的语音识别结果：\n\n"
    max_prefix_deltas = 20
    response_buffer = []
    marker_seen = False
    delta_counter = 0
    emitted_text = ""
    input_transcript_text = ""
    input_transcript_seen = False
    input_transcript_done = asyncio.Event()
    consecutive_transcription_failures = 0
    TRANSCRIPTION_FAILURE_ROTATE_THRESHOLD = 2
    passthrough_without_marker = os.getenv("BRAINWAVE_PASSTHROUGH_WITHOUT_MARKER", "0") == "1"
    homonym_corrector = StreamingHomonymCorrector()

    def normalize_turn_id(raw_turn_id) -> Optional[int]:
        if raw_turn_id is None:
            return None
        try:
            return int(raw_turn_id)
        except (TypeError, ValueError):
            return None

    async def _send_text_payload(content: str):
        nonlocal active_turn_id, emitted_text
        if content and websocket.client_state == WebSocketState.CONNECTED:
            payload = {
                "type": "text",
                "content": content,
                "isNewResponse": False
            }
            if active_turn_id is not None:
                payload["turn_id"] = active_turn_id
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
            emitted_text = merge_incremental_text(emitted_text, content)

    async def emit_text_delta(content: str):
        safe = homonym_corrector.push(content)
        if safe:
            await _send_text_payload(safe)

    async def flush_homonym_corrector():
        tail = homonym_corrector.flush()
        if tail:
            await _send_text_payload(tail)

    def extract_input_transcription_text(data: dict) -> str:
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

    async def flush_buffer(with_warning: bool = False):
        nonlocal response_buffer
        if not response_buffer:
            return
        buffered_text = "".join(response_buffer)
        response_buffer = []
        found_marker, buffered_text = extract_text_after_marker(buffered_text, marker_prefix)
        if not found_marker:
            if passthrough_without_marker:
                await emit_text_delta(buffered_text)
                return
            if with_warning:
                logger.warning("Marker prefix not detected; dropping buffered text.")
            return
        if with_warning and not buffered_text:
            logger.warning("Buffered text discarded after removing marker prefix.")
        await emit_text_delta(buffered_text)

    async def finalize_turn(reason: str):
        nonlocal client, finalized, is_recording, active_provider, active_model, active_turn_id
        nonlocal provider_session_turns, provider_session_started_at
        if finalized:
            return
        finalized = True
        is_recording = False
        pending_audio_chunks.clear()
        recording_stopped.set()
        try:
            await flush_homonym_corrector()
        except Exception as e:
            logger.error(f"Error flushing homonym corrector on finalize ({reason}): {e}", exc_info=True)
        try:
            payload = {
                "type": "status",
                "status": "idle"
            }
            if active_turn_id is not None:
                payload["turn_id"] = active_turn_id
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Error sending status after {reason}: {str(e)}", exc_info=True)
        if not client:
            return

        can_keep_session = keep_provider_session and reason in {
            "response.done",
            "response.text.done",
            "response.output_audio_transcript.done",
        }
        if can_keep_session:
            logger.info(f"Finalizing turn ({reason}), keeping provider client session alive")
            try:
                await client.clear_audio_buffer()
            except Exception as e:
                logger.warning(
                    f"Failed to clear provider audio buffer on finalize ({reason}): {e}"
                )
            return

        logger.info(f"Finalizing turn ({reason}), closing provider client")
        try:
            await client.close()
        except Exception as e:
            logger.error(f"Error closing client after {reason}: {str(e)}", exc_info=True)
        client = None
        active_provider = None
        active_model = None
        active_turn_id = None
        provider_session_turns = 0
        provider_session_started_at = None
        openai_ready.clear()

    def resolve_provider(provider: Optional[str], model: Optional[str]) -> str:
        if provider in {"openai", "xai"}:
            return provider
        if model and (model.startswith("grok-") or model in {"xai", "xai-grok"}):
            return "xai"
        return "openai"
    
    async def create_realtime_client(provider: str = None, model: str = None) -> RealtimeClientBase:
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
    
    async def initialize_realtime_client(
        provider: str = None,
        model: str = None,
        voice: str = None,
        instructions: Optional[str] = None,
        turn_id: Optional[int] = None,
    ):
        nonlocal client, active_provider, active_model, provider_session_turns
        nonlocal provider_session_started_at, consecutive_transcription_failures
        provider_name = provider or REALTIME_PROVIDER
        requested_model = model or OPENAI_REALTIME_MODEL if provider_name == "openai" else None
        max_attempts = provider_init_max_attempts
        retry_delay_sec = provider_init_retry_delay_sec

        can_reuse_existing_client = (
            keep_provider_session
            and client
            and openai_ready.is_set()
            and active_provider == provider_name
            and active_model == requested_model
            and client._is_ws_open()
        )
        turn_limit_reached = (
            provider_session_max_turns > 0
            and provider_session_turns >= provider_session_max_turns
        )
        session_age_sec = None
        if provider_session_started_at is not None:
            session_age_sec = max(0.0, time.time() - provider_session_started_at)
        age_limit_reached = (
            provider_session_max_age_sec > 0
            and session_age_sec is not None
            and session_age_sec >= provider_session_max_age_sec
        )

        transcription_failure_limit_reached = (
            consecutive_transcription_failures >= TRANSCRIPTION_FAILURE_ROTATE_THRESHOLD
        )

        if can_reuse_existing_client and turn_limit_reached:
            logger.info(
                "Provider session reached max turns (%d/%d), rotating session",
                provider_session_turns,
                provider_session_max_turns,
            )
        if can_reuse_existing_client and age_limit_reached:
            logger.info(
                "Provider session reached max age (%.0fs/%.0fs), rotating session",
                session_age_sec or 0.0,
                float(provider_session_max_age_sec),
            )
        if can_reuse_existing_client and transcription_failure_limit_reached:
            logger.warning(
                "Input audio transcription failed %d consecutive times "
                "(threshold=%d), forcing session rotation",
                consecutive_transcription_failures,
                TRANSCRIPTION_FAILURE_ROTATE_THRESHOLD,
            )
            consecutive_transcription_failures = 0

        if can_reuse_existing_client and not turn_limit_reached and not age_limit_reached and not transcription_failure_limit_reached:
            logger.info(
                f"Reusing existing {provider_name} realtime session"
                + (f" ({requested_model})" if requested_model else "")
            )
            try:
                if provider_name == "xai":
                    await client.refresh_session(
                        modalities=XAI_REALTIME_MODALITIES,
                        instructions=instructions,
                    )
                else:
                    await client.refresh_session(
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
                    await client.close()
                except Exception as close_err:
                    logger.error(f"Error closing client after refresh failure: {close_err}")
                client = None
                active_provider = None
                active_model = None
                provider_session_turns = 0
                provider_session_started_at = None
                openai_ready.clear()
            else:
                payload = {
                    "type": "status",
                    "status": "connected"
                }
                if turn_id is not None:
                    payload["turn_id"] = turn_id
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))
                return True

        # Clear the ready flag while initializing
        openai_ready.clear()

        for attempt in range(1, max_attempts + 1):
            try:
                # Close previous client if it wasn't properly finalized
                if client:
                    logger.warning("Previous client still exists during re-init, closing it")
                    try:
                        await client.close()
                    except Exception as e:
                        logger.error(f"Error closing stale client: {e}")
                    client = None
                    active_provider = None
                    active_model = None
                    provider_session_turns = 0
                    provider_session_started_at = None

                # Create client using factory function
                client = await create_realtime_client(provider=provider, model=model)
                
                # Pass appropriate modalities based on provider
                if provider_name == "xai":
                    await client.connect(
                        modalities=XAI_REALTIME_MODALITIES,
                        instructions=instructions,
                    )
                else:
                    await client.connect(
                        modalities=OPENAI_REALTIME_MODALITIES,
                        instructions=instructions,
                    )
                
                logger.info(f"Successfully connected to {provider_name} client (attempt {attempt}/{max_attempts})")
                
                # Register handlers after client is initialized
                client.register_handler("session.updated", lambda data: handle_generic_event("session.updated", data))
                client.register_handler("input_audio_buffer.cleared", lambda data: handle_generic_event("input_audio_buffer.cleared", data))
                client.register_handler("input_audio_buffer.speech_started", lambda data: handle_generic_event("input_audio_buffer.speech_started", data))
                client.register_handler("rate_limits.updated", lambda data: handle_generic_event("rate_limits.updated", data))
                client.register_handler("response.output_item.added", lambda data: handle_generic_event("response.output_item.added", data))
                client.register_handler("conversation.item.created", lambda data: handle_generic_event("conversation.item.created", data))
                client.register_handler("response.content_part.added", lambda data: handle_generic_event("response.content_part.added", data))
                client.register_handler("response.text.done", lambda data: handle_response_text_done(data))
                client.register_handler("response.content_part.done", lambda data: handle_generic_event("response.content_part.done", data))
                client.register_handler("response.output_item.done", lambda data: handle_generic_event("response.output_item.done", data))
                client.register_handler("response.done", lambda data: handle_response_done(data))
                client.register_handler("error", lambda data: handle_error(data))
                client.register_handler("response.text.delta", lambda data: handle_text_delta(data))
                # x.ai uses response.output_audio_transcript.delta instead of response.text.delta
                client.register_handler("response.output_audio_transcript.delta", lambda data: handle_text_delta(data))
                client.register_handler("response.created", lambda data: handle_response_created(data))
                # x.ai specific message types
                client.register_handler("input_audio_buffer.speech_stopped", lambda data: handle_generic_event("input_audio_buffer.speech_stopped", data))
                client.register_handler("input_audio_buffer.committed", lambda data: handle_generic_event("input_audio_buffer.committed", data))
                client.register_handler("conversation.item.added", lambda data: handle_generic_event("conversation.item.added", data))
                client.register_handler(
                    "conversation.item.input_audio_transcription.delta",
                    lambda data: handle_input_audio_transcription_delta(data),
                )
                client.register_handler(
                    "conversation.item.input_audio_transcription.completed",
                    lambda data: handle_input_audio_transcription_completed(data),
                )
                client.register_handler(
                    "conversation.item.input_audio_transcription.failed",
                    lambda data: handle_input_audio_transcription_failed(data),
                )
                client.register_handler("response.output_audio_transcript.done", lambda data: handle_output_audio_transcript_done(data))
                client.register_handler("response.output_audio.delta", lambda data: handle_generic_event("response.output_audio.delta", data))
                client.register_handler("response.output_audio.done", lambda data: handle_generic_event("response.output_audio.done", data))
                client.register_handler("ping", lambda data: handle_generic_event("ping", data))

                # Auto-reconnect when the provider WS drops unexpectedly
                async def _on_provider_disconnect():
                    nonlocal client
                    logger.warning("Provider WebSocket disconnected unexpectedly")
                    openai_ready.clear()  # audio will be buffered to pending_audio_chunks
                    if is_recording:
                        logger.info("Recording active during disconnect — auto-reconnecting")
                        # Close the dead client
                        if client:
                            try:
                                await client.close()
                            except Exception:
                                pass
                            client = None
                        # Re-init (uses same provider/model)
                        ok = await initialize_realtime_client(
                            provider=active_provider,
                            model=active_model,
                            turn_id=active_turn_id,
                        )
                        if ok and pending_audio_chunks:
                            logger.info(
                                "Auto-reconnected, flushing %d buffered audio chunks",
                                len(pending_audio_chunks),
                            )
                            for chunk in pending_audio_chunks:
                                await client.send_audio(chunk)
                            pending_audio_chunks.clear()
                        elif not ok:
                            logger.error("Auto-reconnect failed, audio buffered for next attempt")
                client.set_on_disconnect(_on_provider_disconnect)

                openai_ready.set()  # Set ready flag after successful initialization
                active_provider = provider_name
                active_model = requested_model
                provider_session_turns = 0
                provider_session_started_at = time.time()
                payload = {
                    "type": "status",
                    "status": "connected"
                }
                if turn_id is not None:
                    payload["turn_id"] = turn_id
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))
                return True
            except Exception as e:
                logger.error(
                    f"Failed to connect to {provider_name} client "
                    f"(attempt {attempt}/{max_attempts}): {e}"
                )
                openai_ready.clear()  # Ensure flag is cleared on failure
                if client:
                    try:
                        await client.close()
                    except Exception as close_err:
                        logger.error(f"Error closing failed client: {close_err}")
                    client = None
                    active_provider = None
                    active_model = None
                    provider_session_turns = 0
                    provider_session_started_at = None
                if attempt < max_attempts:
                    await asyncio.sleep(retry_delay_sec)
                    continue
                payload = {
                    "type": "error",
                    "content": f"Failed to initialize {provider_name} realtime connection"
                }
                if turn_id is not None:
                    payload["turn_id"] = turn_id
                await websocket.send_text(json.dumps(payload, ensure_ascii=False))
                return False

    # Move the handler definitions here (before initialize_realtime_client)
    async def handle_text_delta(data):
        nonlocal response_buffer, marker_seen, delta_counter
        try:
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.warning("WebSocket not connected, ignoring text delta")
                return
            delta = data.get("delta", "")
            logger.debug(f"Received text delta: {repr(delta[:50])} (marker_seen={marker_seen}, buffer_size={len(response_buffer)}, delta_counter={delta_counter})")

            if marker_seen:
                if delta:
                    await emit_text_delta(delta)
                    logger.debug(f"Handled response.text.delta (passthrough): {repr(delta[:50])}")
                return

            if not delta:
                return

            response_buffer.append(delta)
            delta_counter += 1

            joined = "".join(response_buffer)
            found_marker, remaining = extract_text_after_marker(joined, marker_prefix)
            if found_marker:
                marker_seen = True
                response_buffer = []
                if remaining:
                    await emit_text_delta(remaining)
                logger.debug(
                    f"Handled response.text.delta (marker stripped), emitted: {repr(remaining[:50])}"
                )
                return

            if passthrough_without_marker:
                # Keep a short tail for cross-chunk marker matching; pass through the rest.
                keep_tail = max(0, len(marker_prefix) - 1)
                if len(joined) > keep_tail:
                    passthrough_text = joined[:-keep_tail] if keep_tail else joined
                    response_buffer = [joined[-keep_tail:]] if keep_tail else []
                    if passthrough_text:
                        await emit_text_delta(passthrough_text)
                    logger.debug(
                        "Handled response.text.delta (safe passthrough), "
                        f"emitted_len={len(passthrough_text)} keep_tail={keep_tail}"
                    )
                return

            if delta_counter >= max_prefix_deltas:
                response_buffer = []
                delta_counter = 0
                logger.warning("Marker prefix not detected after max deltas; dropping buffered text.")
            else:
                logger.debug(f"Handled response.text.delta (buffering), total buffer length: {len(joined)}")
        except Exception as e:
            logger.error(f"Error in handle_text_delta: {str(e)}", exc_info=True)

    async def handle_input_audio_transcription_delta(data):
        nonlocal input_transcript_text, input_transcript_seen
        try:
            delta = extract_input_transcription_text(data)
            if not delta:
                return
            input_transcript_seen = True
            input_transcript_text = merge_incremental_text(input_transcript_text, delta)
        except Exception as e:
            logger.error(f"Error in handle_input_audio_transcription_delta: {e}", exc_info=True)

    async def handle_input_audio_transcription_completed(data):
        nonlocal input_transcript_text, input_transcript_seen, consecutive_transcription_failures
        try:
            completed_text = extract_input_transcription_text(data)
            input_transcript_seen = True
            consecutive_transcription_failures = 0
            if completed_text and len(completed_text) >= len(input_transcript_text):
                input_transcript_text = completed_text
            logger.info(
                "Handled conversation.item.input_audio_transcription.completed "
                "(len=%d)",
                len(input_transcript_text),
            )
        except Exception as e:
            logger.error(f"Error in handle_input_audio_transcription_completed: {e}", exc_info=True)
        finally:
            input_transcript_done.set()

    async def handle_input_audio_transcription_failed(data):
        nonlocal consecutive_transcription_failures
        error_info = data.get("error", {})
        error_type = error_info.get("type", "unknown")
        error_code = error_info.get("code", "unknown")
        error_message = error_info.get("message", "no message")
        consecutive_transcription_failures += 1
        logger.error(
            "Input audio transcription FAILED (type=%s, code=%s, message=%s, "
            "consecutive_failures=%d/%d)",
            error_type,
            error_code,
            error_message,
            consecutive_transcription_failures,
            TRANSCRIPTION_FAILURE_ROTATE_THRESHOLD,
        )
        # Unblock grace wait so finalize_turn doesn't wait 1.2s for nothing
        input_transcript_done.set()

    async def maybe_apply_input_transcription_fallback(event_type: str):
        nonlocal emitted_text, input_transcript_text
        if active_provider != "openai":
            return
        # Only wait for input transcription when marker_seen is False
        # (model may have answered instead of transcribing).  When the
        # marker was seen the model followed the transcription format,
        # so the fallback is very unlikely to be needed and we skip the
        # grace wait to avoid adding ~1.2 s of unnecessary latency.
        if not input_transcript_done.is_set() and input_transcript_grace_sec > 0:
            if not marker_seen:
                try:
                    await asyncio.wait_for(
                        input_transcript_done.wait(),
                        timeout=input_transcript_grace_sec,
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "Input transcription grace wait timed out (%.2fs) on %s",
                        input_transcript_grace_sec,
                        event_type,
                    )
            else:
                logger.info(
                    "Skipping input transcription grace wait (marker_seen=True) on %s",
                    event_type,
                )

        fallback_text = input_transcript_text.strip()
        if not fallback_text:
            return
        current_text = emitted_text.strip()

        # When marker_seen is False the model did NOT follow the
        # transcription-only format (likely answered the user's speech
        # instead of transcribing it).  In that case, unconditionally
        # replace the emitted text with the input transcription from
        # gpt-4o-transcribe, which is always a faithful transcription.
        if not marker_seen and current_text:
            logger.warning(
                "Marker not seen — model likely answered instead of "
                "transcribing. Replacing emitted text with input "
                "transcription (emitted_len=%d, fallback_len=%d)",
                len(current_text),
                len(fallback_text),
            )
        elif marker_seen and current_text:
            # Model followed the transcription format and produced output;
            # trust the model over input_audio_transcription (which can
            # return garbled text for very short utterances).
            return
        elif current_text and (
            fallback_text == current_text
            or len(fallback_text) <= len(current_text) + 1
        ):
            return

        payload = {
            "type": "text",
            "content": fallback_text,
            "isNewResponse": True,
        }
        if active_turn_id is not None:
            payload["turn_id"] = active_turn_id
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        emitted_text = fallback_text
        logger.info(
            "Applied input transcription fallback before finalize "
            "(fallback_len=%d current_len=%d marker_seen=%s)",
            len(fallback_text),
            len(current_text),
            marker_seen,
        )

    async def handle_response_created(data):
        nonlocal response_buffer, marker_seen, delta_counter, emitted_text
        response_buffer = []
        marker_seen = False
        delta_counter = 0
        emitted_text = ""
        logger.info(f"Handled response.created, clearing buffer and resetting marker state")
        logger.info("Handled response.created")

    async def handle_error(data):
        error_msg = data.get("error", {}).get("message", "Unknown error")
        logger.error(f"Provider error: {error_msg}")
        payload = {
            "type": "error",
            "content": error_msg
        }
        if active_turn_id is not None:
            payload["turn_id"] = active_turn_id
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        await finalize_turn("error")
        logger.info("Handled error message from provider")

    async def handle_text_completed(event_type: str, data):
        nonlocal response_buffer, marker_seen
        logger.info(
            f"Handled {event_type} "
            f"(marker_seen={marker_seen}, buffer_size={len(response_buffer)})"
        )
        if not marker_seen and emitted_text:
            logger.warning(
                "Model output without marker prefix (possible Q&A instead of "
                "transcription): %s",
                repr(emitted_text[:120]),
            )
        if response_buffer:
            logger.info("Flushing remaining buffer content")
            await flush_buffer()
            marker_seen = True
        await maybe_apply_input_transcription_fallback(event_type)
        await finalize_turn(event_type)

    async def handle_response_done(data):
        await handle_text_completed("response.done", data)

    async def handle_response_text_done(data):
        await handle_text_completed("response.text.done", data)

    async def handle_output_audio_transcript_done(data):
        await handle_text_completed("response.output_audio_transcript.done", data)

    async def handle_generic_event(event_type, data):
        if VERBOSE_SERVER_LOG:
            logger.info(f"Handled {event_type} with data: {json.dumps(data, ensure_ascii=False)}")
        else:
            logger.debug(f"Handled {event_type}")

    async def receive_messages():
        nonlocal client, finalized, is_recording, provider_session_turns, active_turn_id
        nonlocal response_buffer, marker_seen, delta_counter
        nonlocal emitted_text, input_transcript_text, input_transcript_seen
        logger.info("receive_messages task started")
        
        try:
            while True:
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info("WebSocket client disconnected")
                    openai_ready.clear()
                    break
                    
                try:
                    # Add timeout to prevent infinite waiting
                    logger.debug("Waiting for message from client (timeout=30s)...")
                    data = await asyncio.wait_for(websocket.receive(), timeout=30.0)
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
                    processed_audio = audio_processor.process_audio_chunk(data["bytes"])
                    if not openai_ready.is_set():
                        logger.debug("Provider not ready, buffering audio chunk")
                        pending_audio_chunks.append(processed_audio)
                    elif client and is_recording:
                        await client.send_audio(processed_audio)
                        logger.debug(f"Sent audio chunk, size: {len(processed_audio)} bytes")
                    else:
                        logger.debug("Received audio but client is not initialized")
                            
                elif "text" in data:
                    msg = json.loads(data["text"])
                    logger.debug(f"Received message from client: {msg.get('type')}")
                    
                    if msg.get("type") == "start_recording":
                        logger.info("Processing start_recording request")
                        requested_turn_id = normalize_turn_id(msg.get("turn_id"))
                        if msg.get("turn_id") is not None and requested_turn_id is None:
                            logger.warning(f"Ignoring invalid turn_id from client: {msg.get('turn_id')!r}")
                        active_turn_id = requested_turn_id
                        
                        # Update status to connecting while initializing realtime client
                        status_payload = {
                            "type": "status",
                            "status": "connecting"
                        }
                        if active_turn_id is not None:
                            status_payload["turn_id"] = active_turn_id
                        await websocket.send_text(json.dumps(status_payload, ensure_ascii=False))
                        # Extract provider and model from message
                        provider = msg.get("provider")  # "openai" or "xai"
                        model = msg.get("model")  # OpenAI model name
                        # Optimized-only mode: use optimize prompt
                        logger.info("Optimized-only mode: using optimize prompt")
                        input_sample_rate = msg.get("input_sample_rate")
                        
                        logger.info(f"Received start_recording: provider={provider}, model={model}")
                        
                        provider = resolve_provider(provider, model)
                        logger.info(f"Using provider: {provider}")
                        instructions = get_optimize_prompt()

                        if input_sample_rate:
                            try:
                                audio_processor.set_source_sample_rate(int(input_sample_rate))
                                logger.info(f"Using input sample rate: {audio_processor.source_sample_rate}Hz")
                            except (TypeError, ValueError):
                                logger.warning(f"Invalid input_sample_rate '{input_sample_rate}', using default {default_source_sample_rate}Hz")
                                audio_processor.set_source_sample_rate(default_source_sample_rate)
                        else:
                            audio_processor.set_source_sample_rate(default_source_sample_rate)
                        
                        if not await initialize_realtime_client(
                            provider=provider,
                            model=model,
                            instructions=instructions,
                            turn_id=active_turn_id,
                        ):
                            continue
                        provider_session_turns += 1
                        if provider_session_max_turns > 0:
                            logger.info(
                                "Provider session turn started: %d/%d",
                                provider_session_turns,
                                provider_session_max_turns,
                            )
                        else:
                            logger.info(
                                "Provider session turn started: %d (max-turn rotation disabled)",
                                provider_session_turns,
                            )
                        finalized = False
                        recording_stopped.clear()
                        response_buffer = []
                        marker_seen = False
                        delta_counter = 0
                        emitted_text = ""
                        input_transcript_text = ""
                        input_transcript_seen = False
                        input_transcript_done.clear()
                        # Immediately clear transcript for a new client-initiated request
                        reset_payload = {
                            "type": "text",
                            "content": "",
                            "isNewResponse": True
                        }
                        if active_turn_id is not None:
                            reset_payload["turn_id"] = active_turn_id
                        await websocket.send_text(json.dumps(reset_payload, ensure_ascii=False))
                        is_recording = True
                        
                        # Send any buffered chunks
                        if pending_audio_chunks and client:
                            logger.info(f"Sending {len(pending_audio_chunks)} buffered chunks")
                            for chunk in pending_audio_chunks:
                                await client.send_audio(chunk)
                            pending_audio_chunks.clear()
                        
                    elif msg.get("type") == "stop_recording":
                        requested_turn_id = normalize_turn_id(msg.get("turn_id"))
                        if (
                            requested_turn_id is not None
                            and active_turn_id is not None
                            and requested_turn_id != active_turn_id
                        ):
                            logger.warning(
                                "Ignoring stop_recording for stale turn_id=%s (active_turn_id=%s)",
                                requested_turn_id,
                                active_turn_id,
                            )
                            continue
                        # On explicit Stop, force-commit and force-create a response, then wait for completion.
                        if client:
                            # Immediately stop accepting further audio for this turn
                            is_recording = False
                            try:
                                await client.commit_audio()
                                logger.info("Audio committed, starting response...")
                                # Use text-only modalities for x.ai if configured
                                if isinstance(client, XAIRealtimeAudioTextClient):
                                    modalities = XAI_REALTIME_MODALITIES
                                    await client.start_response(get_optimize_prompt(), modalities=modalities)
                                else:
                                    # OpenAI: by default rely on session-level instructions
                                    # (can be overridden via env flag on client side).
                                    await client.start_response(get_optimize_prompt())
                                logger.info("Response started successfully")
                            except Exception as e:
                                logger.error(f"Error committing/starting response on stop: {str(e)}", exc_info=True)
                                # If we fail to kick off a response, surface that we're no longer recording
                                status_payload = {
                                    "type": "status",
                                    "status": "idle"
                                }
                                if active_turn_id is not None:
                                    status_payload["turn_id"] = active_turn_id
                                await websocket.send_text(json.dumps(status_payload, ensure_ascii=False))
                                continue
                            # Wait until the response is finished; timeout is configurable for long turns.
                            try:
                                await asyncio.wait_for(
                                    recording_stopped.wait(),
                                    timeout=response_finalize_timeout_sec,
                                )
                            except asyncio.TimeoutError:
                                logger.error(
                                    "Response timed out after %.1fs, forcing finalization",
                                    response_finalize_timeout_sec,
                                )
                                await finalize_turn("timeout")
                
        finally:
            # Cleanup when the loop exits
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.error(f"Error closing client in receive_messages: {str(e)}")
            logger.info("Receive messages loop ended")

    # Start task for receiving client messages
    receive_task = asyncio.create_task(receive_messages())

    try:
        await receive_task
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
    finally:
        # Cancel background task before cleanup
        receive_task.cancel()
        
        # Wait for task to be cancelled (with timeout)
        try:
            await asyncio.wait_for(asyncio.gather(receive_task, return_exceptions=True), timeout=1.0)
        except asyncio.TimeoutError:
            logger.warning("Task did not cancel within timeout")
        except Exception as e:
            logger.debug(f"Error cancelling tasks: {e}")
        
        if client:
            await client.close()
        if websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError as e:
                logger.warning(f"Ignoring error during websocket close: {e}")
        logger.info("WebSocket connection closed for /api/v1/ws")

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=23456)
