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
)
from audio_persistence import TurnAudioCache
from openai_realtime_client import OpenAIRealtimeAudioTextClient
from prompts import get_optimize_prompt
from realtime_client_base import RealtimeClientBase
from realtime_text_utils import (
    NO_SPEECH_PLACEHOLDER,
    SIMILARITY_HARD_CAP_CHARS,
    StreamingHomonymCorrector,
    StreamingNoSpeechPrefixGuard,
    answer_guard_fingerprint,
    emitted_novel_material_ratio,
    extract_text_after_marker,
    is_no_speech_placeholder_only,
    parse_ratio_env,
    transcription_similarity_ratio,
)
from transcript_merge import merge_incremental_text

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
    suspicious_marker_audio_sec: float
    suspicious_marker_emitted_chars: int
    suspicious_input_transcript_grace_sec: float
    input_transcript_replacement_min_ratio: float
    input_transcript_replacement_min_delta_chars: int
    default_source_sample_rate: int
    marker_prefix: str
    max_prefix_deltas: int
    transcription_failure_rotate_threshold: int
    passthrough_without_marker: bool
    answer_guard_min_similarity: float
    answer_guard_novel_material_ratio: float
    answer_guard_grace_sec: float
    similarity_hard_cap_chars: int
    no_speech_guard_enabled: bool

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
            suspicious_marker_audio_sec=float_env("BRAINWAVE_SUSPICIOUS_MARKER_AUDIO_SEC", 20.0, 0.0),
            suspicious_marker_emitted_chars=int_env("BRAINWAVE_SUSPICIOUS_MARKER_EMITTED_CHARS", 120, 0),
            suspicious_input_transcript_grace_sec=float_env("BRAINWAVE_SUSPICIOUS_INPUT_TRANSCRIPT_GRACE_SEC", 3.0, 0.0),
            input_transcript_replacement_min_ratio=float_env("BRAINWAVE_INPUT_TRANSCRIPT_REPLACEMENT_MIN_RATIO", 1.75, 1.0),
            input_transcript_replacement_min_delta_chars=int_env("BRAINWAVE_INPUT_TRANSCRIPT_REPLACEMENT_MIN_DELTA_CHARS", 80, 0),
            default_source_sample_rate=AudioProcessor().source_sample_rate,
            marker_prefix="下面是不改变语言的语音识别结果：\n\n",
            max_prefix_deltas=20,
            transcription_failure_rotate_threshold=2,
            passthrough_without_marker=os.getenv("BRAINWAVE_PASSTHROUGH_WITHOUT_MARKER", "0") == "1",
            # Answer-similarity guard (task 0436 G1/M4). min_similarity<=0
            # disables the guard. Default 0.60: a pure deletion-shaped clean-up
            # (ratio = 2r/(1+r)) needs >57% of chars deleted to trip 0.60, so
            # legitimate filler cleanup (~30% deletion → ~0.82) is safe, while
            # restate-then-answer forms (~0.53/0.58) are caught. The
            # novel-material ratio is a second signal: emitted text carrying more
            # than this fraction of material unaligned with the transcript also
            # flags an answer. Both fail open. grace now defaults to 1.2s so the
            # guard can wait for the faithful transcript when it has not arrived
            # (S1); an already-completed transcript short-circuits with no wait.
            answer_guard_min_similarity=parse_ratio_env("BRAINWAVE_ANSWER_GUARD_MIN_SIMILARITY", 0.60),
            answer_guard_novel_material_ratio=parse_ratio_env("BRAINWAVE_ANSWER_GUARD_NOVEL_MATERIAL_RATIO", 0.55),
            answer_guard_grace_sec=float_env("BRAINWAVE_ANSWER_GUARD_GRACE_SEC", 1.2, 0.0),
            similarity_hard_cap_chars=SIMILARITY_HARD_CAP_CHARS,
            no_speech_guard_enabled=os.getenv("BRAINWAVE_NO_SPEECH_GUARD", "1") == "1",
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
        self._emitted_without_marker = False
        self._delta_counter = 0
        self._emitted_text = ""
        self._input_transcript_text = ""
        self._input_transcript_seen = False
        self._input_transcript_completed = False
        self._homonym_corrector = StreamingHomonymCorrector()
        self._processed_audio_bytes = 0

        self._legacy_provider_warned = False
        self._closed = False
        # Tracked finalize barrier task for the current turn (task 0436 S1 R4).
        # response.done schedules it and returns so the provider dispatcher is
        # not blocked waiting on a late ASR event it must itself deliver.
        self._finalize_task: Optional[asyncio.Task] = None
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

    def _reset_turn_state(
        self,
        active_turn_id: Optional[int] = None,
        preserve_pending_audio: bool = False,
    ):
        # A finalize barrier from the previous turn must never run against this
        # new turn's state (task 0436 S1 R4). Cancel it here; it unwinds at its
        # next await and its captured turn token blocks any late send onto this
        # turn.
        finalize_task = getattr(self, "_finalize_task", None)
        if finalize_task is not None and not finalize_task.done():
            finalize_task.cancel()
        self._finalize_task = None
        # On a same-turn retry (confirmation failure) the not-yet-forwarded PCM
        # already received for this turn must survive the reset so it is not
        # dropped on rebuild (task 0436 S2).
        preserved_pending = self._pending_audio_chunks if preserve_pending_audio else []
        self._active_turn_id = active_turn_id
        self._finalized = False
        self._is_recording = False
        self._turn_done = asyncio.Event()
        self._input_transcript_done = asyncio.Event()
        self._pending_audio_chunks = preserved_pending
        self._response_buffer = []
        self._marker_seen = False
        self._emitted_without_marker = False
        self._delta_counter = 0
        self._emitted_text = ""
        self._input_transcript_text = ""
        self._input_transcript_seen = False
        self._input_transcript_completed = False
        self._input_transcript_wait_timeouts = 0
        self._no_speech_prefix_guard = StreamingNoSpeechPrefixGuard()
        self._homonym_corrector = StreamingHomonymCorrector()
        self._processed_audio_bytes = 0

    async def _send_text_payload(self, content: str, is_new_response: bool = False):
        if content and self._websocket.client_state == WebSocketState.CONNECTED:
            payload = {
                "type": "text",
                "content": content,
                "isNewResponse": is_new_response,
            }
            if self._active_turn_id is not None:
                payload["turn_id"] = self._active_turn_id
            await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
            if is_new_response:
                self._emitted_text = content
            else:
                self._emitted_text = merge_incremental_text(self._emitted_text, content)

    async def _emit_text_delta(self, content: str):
        safe = self._homonym_corrector.push(content)
        if safe:
            await self._send_text_payload(safe)

    async def _flush_homonym_corrector(self):
        tail = self._homonym_corrector.flush()
        if tail:
            await self._send_text_payload(tail)

    async def _emit_body_text_delta(self, content: str):
        if not self._config.no_speech_guard_enabled:
            await self._emit_text_delta(content)
            return
        safe = self._no_speech_prefix_guard.push(content)
        if safe:
            await self._emit_text_delta(safe)

    async def _flush_no_speech_prefix_guard(self):
        if not self._config.no_speech_guard_enabled:
            return
        safe = self._no_speech_prefix_guard.flush()
        if safe:
            await self._emit_text_delta(safe)

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
        await self._emit_body_text_delta(buffered_text)
        return True

    def _processed_audio_duration_sec(self) -> float:
        bytes_per_second = self._audio_processor.target_sample_rate * 2
        if bytes_per_second <= 0:
            return 0.0
        return self._processed_audio_bytes / bytes_per_second

    def _is_suspicious_marker_output(self, current_text: str) -> bool:
        if not self._marker_seen:
            return False
        if (
            self._config.suspicious_marker_audio_sec <= 0
            or self._config.suspicious_marker_emitted_chars <= 0
        ):
            return False
        return (
            self._processed_audio_duration_sec() >= self._config.suspicious_marker_audio_sec
            and len(current_text.strip()) <= self._config.suspicious_marker_emitted_chars
        )

    def _is_material_input_transcript_replacement(
        self,
        current_text: str,
        fallback_text: str,
    ) -> bool:
        current_len = len(current_text.strip())
        fallback_len = len(fallback_text.strip())
        if fallback_len == 0:
            return False
        if current_len == 0:
            return True
        if fallback_text.strip() == current_text.strip():
            return False
        min_delta = self._config.input_transcript_replacement_min_delta_chars
        min_ratio = self._config.input_transcript_replacement_min_ratio
        return (
            fallback_len >= current_len + min_delta
            and fallback_len >= current_len * min_ratio
        )

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
            await self._flush_no_speech_prefix_guard()
        except Exception as e:
            logger.error(
                f"Error flushing no-speech prefix guard on finalize ({reason}): {e}",
                exc_info=True,
            )
        try:
            await self._flush_homonym_corrector()
        except Exception as e:
            logger.error(f"Error flushing homonym corrector on finalize ({reason}): {e}", exc_info=True)
        emitted_is_placeholder_only = is_no_speech_placeholder_only(self._emitted_text)
        emitted_is_effectively_empty = (
            not self._emitted_text.strip() or emitted_is_placeholder_only
        )
        if (
            self._config.no_speech_guard_enabled
            and reason in self._SUCCESSFUL_FINALIZE_REASONS
            and emitted_is_effectively_empty
            and not self._input_transcript_text.strip()
        ):
            if emitted_is_placeholder_only:
                self._homonym_corrector.reset()
            await self._send_text_payload(
                NO_SPEECH_PLACEHOLDER,
                is_new_response=emitted_is_placeholder_only,
            )
            logger.warning(
                "No speech recognized on successful finalize (%s): "
                "audio_sec=%.2f processed_audio_bytes=%d; emitted canonical placeholder",
                reason,
                self._processed_audio_duration_sec(),
                self._processed_audio_bytes,
            )
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
        self._active_model = None
        self._active_turn_id = None
        self._provider_session_turns = 0
        self._provider_session_started_at = None
        self._openai_ready.clear()

    async def _create_client(
        self,
        model: str = None,
    ) -> RealtimeClientBase:
        """Create the OpenAI realtime client.

        Args:
            model: OpenAI model name. Defaults to OPENAI_REALTIME_MODEL.

        Returns:
            RealtimeClientBase instance
        """
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        selected_model = model or OPENAI_REALTIME_MODEL
        logger.info(f"Creating OpenAI client with model: {selected_model}")
        return OpenAIRealtimeAudioTextClient(api_key, model=selected_model)

    async def _init_or_reuse_client(
        self,
        model: str = None,
        instructions: Optional[str] = None,
        turn_id: Optional[int] = None,
    ):
        requested_model = model or OPENAI_REALTIME_MODEL
        max_attempts = self._config.provider_init_max_attempts
        retry_delay_sec = self._config.provider_init_retry_delay_sec

        can_reuse_existing_client = (
            self._config.keep_provider_session
            and self._client
            and self._openai_ready.is_set()
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
                "Reusing existing OpenAI realtime session"
                + (f" ({requested_model})" if requested_model else "")
            )
            try:
                await self._client.refresh_session(
                    modalities=OPENAI_REALTIME_MODALITIES,
                    instructions=instructions,
                )
            except Exception as refresh_err:
                logger.warning(
                    "Failed to refresh reused OpenAI session (%s). Recreating session.",
                    refresh_err,
                )
                try:
                    await self._client.close()
                except Exception as close_err:
                    logger.error(f"Error closing client after refresh failure: {close_err}")
                self._client = None
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
                    self._active_model = None
                    self._provider_session_turns = 0
                    self._provider_session_started_at = None

                # Create the OpenAI realtime client
                self._client = await self._create_client(model=model)

                await self._client.connect(
                    modalities=OPENAI_REALTIME_MODALITIES,
                    instructions=instructions,
                )

                logger.info(f"Successfully connected to OpenAI client (attempt {attempt}/{max_attempts})")
                
                # Register handlers after client is initialized
                self._register_client_handlers(self._client)

                self._openai_ready.set()  # Set ready flag after successful initialization
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
                    f"Failed to connect to OpenAI client "
                    f"(attempt {attempt}/{max_attempts}): {e}"
                )
                self._openai_ready.clear()  # Ensure flag is cleared on failure
                if self._client:
                    try:
                        await self._client.close()
                    except Exception as close_err:
                        logger.error(f"Error closing failed client: {close_err}")
                    self._client = None
                    self._active_model = None
                    self._provider_session_turns = 0
                    self._provider_session_started_at = None
                if attempt < max_attempts:
                    await asyncio.sleep(retry_delay_sec)
                    continue
                payload = {
                    "type": "error",
                    "content": "Failed to initialize OpenAI realtime connection"
                }
                if turn_id is not None:
                    payload["turn_id"] = turn_id
                await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
                return False

    def _register_client_handlers(self, client: RealtimeClientBase):
        """Register all provider event handlers on ``client`` and wire the
        disconnect callback. Extracted from _init_or_reuse_client so the exact
        production handler set can be driven through the real serial dispatcher
        (OpenAIRealtimeAudioTextClient.receive_messages) in tests (task 0436 S1
        R4).
        """
        client.register_handler("session.updated", lambda data: self._on_generic_event("session.updated", data))
        client.register_handler("input_audio_buffer.cleared", lambda data: self._on_generic_event("input_audio_buffer.cleared", data))
        client.register_handler("input_audio_buffer.speech_started", lambda data: self._on_generic_event("input_audio_buffer.speech_started", data))
        client.register_handler("rate_limits.updated", lambda data: self._on_generic_event("rate_limits.updated", data))
        client.register_handler("response.output_item.added", lambda data: self._on_generic_event("response.output_item.added", data))
        client.register_handler("conversation.item.created", lambda data: self._on_generic_event("conversation.item.created", data))
        client.register_handler("response.content_part.added", lambda data: self._on_generic_event("response.content_part.added", data))
        client.register_handler("response.text.done", self._on_response_text_done)
        client.register_handler("response.output_text.done", self._on_response_text_done)
        client.register_handler("response.content_part.done", lambda data: self._on_generic_event("response.content_part.done", data))
        client.register_handler("response.output_item.done", lambda data: self._on_generic_event("response.output_item.done", data))
        client.register_handler("response.done", self._on_response_done)
        client.register_handler("error", self._on_error)
        client.register_handler("response.text.delta", self._on_text_delta)
        client.register_handler("response.output_text.delta", self._on_text_delta)
        # Audio-transcript deltas route to the text handler too (unused in text-only mode)
        client.register_handler("response.output_audio_transcript.delta", self._on_text_delta)
        client.register_handler("response.created", self._on_response_created)
        # Additional realtime event types
        client.register_handler("input_audio_buffer.speech_stopped", lambda data: self._on_generic_event("input_audio_buffer.speech_stopped", data))
        client.register_handler("input_audio_buffer.committed", lambda data: self._on_generic_event("input_audio_buffer.committed", data))
        client.register_handler("conversation.item.added", lambda data: self._on_generic_event("conversation.item.added", data))
        client.register_handler(
            "conversation.item.input_audio_transcription.delta",
            self._on_input_transcription_delta,
        )
        client.register_handler(
            "conversation.item.input_audio_transcription.completed",
            self._on_input_transcription_completed,
        )
        client.register_handler(
            "conversation.item.input_audio_transcription.failed",
            self._on_input_transcription_failed,
        )
        client.register_handler("response.output_audio_transcript.done", self._on_output_audio_transcript_done)
        client.register_handler("response.output_audio.delta", lambda data: self._on_generic_event("response.output_audio.delta", data))
        client.register_handler("response.output_audio.done", lambda data: self._on_generic_event("response.output_audio.done", data))
        client.register_handler("ping", lambda data: self._on_generic_event("ping", data))

        # Auto-reconnect when the provider WS drops unexpectedly
        client.set_on_disconnect(self._on_provider_disconnect)

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

            if not self._active_model:
                logger.error("Provider disconnected with no active session; finalizing turn")
                await self._finalize_turn("provider_unavailable")
                return

            logger.info("Recording active during disconnect; auto-reconnecting")
            ok = await self._init_or_reuse_client(
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
                "Received text delta (delta_len=%d, marker_seen=%s, buffer_size=%d, delta_counter=%d)",
                len(delta),
                self._marker_seen,
                len(self._response_buffer),
                self._delta_counter,
            )

            if self._marker_seen or self._emitted_without_marker:
                if delta:
                    if self._marker_seen:
                        await self._emit_body_text_delta(delta)
                    else:
                        await self._emit_text_delta(delta)
                    logger.debug("Handled response.text.delta (passthrough, delta_len=%d)", len(delta))
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
                    await self._emit_body_text_delta(remaining)
                logger.debug(
                    "Handled response.text.delta (marker stripped, emitted_len=%d)",
                    len(remaining),
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
                # Keep streaming so we never drop words (task 0281), but do NOT
                # fake marker_seen — flag emitted-without-marker so finalize's
                # no-marker replacement keeps governing this path (task 0436 G3).
                self._emitted_without_marker = True
                if buffered:
                    await self._emit_text_delta(buffered)
                logger.warning(
                    "Marker prefix not detected after %d deltas; emitting as-is "
                    "without marker (len=%d).",
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
            self._input_transcript_completed = True
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

    def _current_emitted_view(self) -> str:
        """Full corrected text produced this turn, including the homonym
        corrector's not-yet-sent held tail (task 0436 M6). Falls back to the
        already-sent text when the corrector view is empty (e.g. after a
        replacement reset it).
        """
        view = self._homonym_corrector.peek_full_text()
        if view:
            return view
        return self._emitted_text

    async def _apply_input_transcription_fallback(self, event_type: str, turn_token=None):
        done = self._input_transcript_done

        def current_text_for_fallback() -> str:
            current = self._current_emitted_view().strip()
            if (
                self._config.no_speech_guard_enabled
                and is_no_speech_placeholder_only(current)
            ):
                return ""
            return current

        current_text = current_text_for_fallback()
        suspicious_marker_output = self._is_suspicious_marker_output(current_text)
        answer_guard_enabled = self._config.answer_guard_min_similarity > 0
        # Single bounded barrier before status:idle (task 0436 S1). Wait for the
        # faithful transcript only when it may be needed and has not arrived, so
        # a normal turn (transcript already completed) adds no perceptible
        # latency. On timeout we fail open (no replacement) and count it.
        grace_sec = 0.0
        if not self._marker_seen:
            grace_sec = self._config.input_transcript_grace_sec
        elif suspicious_marker_output:
            grace_sec = self._config.suspicious_input_transcript_grace_sec
        elif answer_guard_enabled:
            grace_sec = self._config.answer_guard_grace_sec

        if done and not done.is_set() and grace_sec > 0:
            try:
                await asyncio.wait_for(done.wait(), timeout=grace_sec)
            except asyncio.TimeoutError:
                self._input_transcript_wait_timeouts += 1
                logger.info(
                    "Input transcription wait timed out (%.2fs) on %s; failing "
                    "open without replacement (marker_seen=%s suspicious=%s "
                    "audio_sec=%.2f timeout_count=%d)",
                    grace_sec,
                    event_type,
                    self._marker_seen,
                    suspicious_marker_output,
                    self._processed_audio_duration_sec(),
                    self._input_transcript_wait_timeouts,
                )
            else:
                logger.debug(
                    "Input transcription completed within grace on %s "
                    "(marker_seen=%s suspicious=%s)",
                    event_type,
                    self._marker_seen,
                    suspicious_marker_output,
                )
        elif done and not done.is_set() and self._marker_seen:
            logger.debug(
                "Skipping input transcription grace wait (grace<=0) on %s "
                "(marker_seen=True suspicious=%s)",
                event_type,
                suspicious_marker_output,
            )

        # A new turn may have started (start_recording → reset) while we waited
        # on the grace barrier, or the turn may already be finalized by another
        # path. Never apply a replacement onto a superseded/finalized turn
        # (task 0436 S1 R4 lifecycle).
        if self._finalized or self._is_stale_turn(turn_token):
            return

        fallback_text = self._input_transcript_text.strip()
        if not fallback_text:
            return
        current_text = current_text_for_fallback()

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
            # Marker-following output is normally trusted. Two exceptions can
            # still replace it with the faithful input transcription:
            #   (a) answer-similarity guard (task 0436 G1): the emitted text
            #       diverges materially from the completed transcript — i.e. the
            #       model answered instead of transcribing.
            #   (b) long-audio/short-output drift: marker present but the output
            #       is materially shorter than the transcript.
            answer_guard_triggered = False
            if (
                answer_guard_enabled
                and self._input_transcript_completed
                and fallback_text
            ):
                cap = self._config.similarity_hard_cap_chars
                ratio = None
                novel_ratio = None
                try:
                    # Run the bounded diffs off the event loop (task 0436 M5).
                    ratio = await asyncio.to_thread(
                        transcription_similarity_ratio,
                        current_text, fallback_text, cap,
                    )
                    novel_ratio = await asyncio.to_thread(
                        emitted_novel_material_ratio,
                        current_text, fallback_text, cap,
                    )
                except Exception as guard_err:
                    logger.error(
                        "[answer-guard] similarity computation failed: %s", guard_err
                    )
                    ratio = None
                    novel_ratio = None
                low_similarity = (
                    ratio is not None
                    and ratio < self._config.answer_guard_min_similarity
                )
                high_novelty = (
                    novel_ratio is not None
                    and self._config.answer_guard_novel_material_ratio > 0
                    and novel_ratio > self._config.answer_guard_novel_material_ratio
                )
                if low_similarity or high_novelty:
                    answer_guard_triggered = True
                    emitted_len, emitted_sha = answer_guard_fingerprint(current_text)
                    input_len, input_sha = answer_guard_fingerprint(fallback_text)
                    logger.warning(
                        "[answer-guard] marker-following output flagged as answer "
                        "(ratio=%s<%.3f=%s novel=%s>%.3f=%s); replacing "
                        "(emitted_len=%d emitted_sha1=%s input_len=%d input_sha1=%s)",
                        f"{ratio:.3f}" if ratio is not None else "n/a",
                        self._config.answer_guard_min_similarity,
                        low_similarity,
                        f"{novel_ratio:.3f}" if novel_ratio is not None else "n/a",
                        self._config.answer_guard_novel_material_ratio,
                        high_novelty,
                        emitted_len,
                        emitted_sha,
                        input_len,
                        input_sha,
                    )
            if not answer_guard_triggered:
                if not suspicious_marker_output:
                    return
                if not self._input_transcript_completed:
                    logger.info(
                        "Suspicious marker output fallback skipped because input "
                        "transcription has not completed "
                        "(fallback_len=%d current_len=%d audio_sec=%.2f)",
                        len(fallback_text),
                        len(current_text),
                        self._processed_audio_duration_sec(),
                    )
                    return
                if not self._is_material_input_transcript_replacement(current_text, fallback_text):
                    logger.info(
                        "Suspicious marker output fallback skipped because input "
                        "transcription was not materially longer "
                        "(fallback_len=%d current_len=%d audio_sec=%.2f)",
                        len(fallback_text),
                        len(current_text),
                        self._processed_audio_duration_sec(),
                    )
                    return
                logger.warning(
                    "Suspicious marker output replaced with materially longer "
                    "input transcription (audio_sec=%.2f emitted_len=%d fallback_len=%d)",
                    self._processed_audio_duration_sec(),
                    len(current_text),
                    len(fallback_text),
                )
        elif current_text and (
            fallback_text == current_text
            or len(fallback_text) <= len(current_text) + 1
        ):
            return

        # A concurrent _finalize_turn (reachable via a provider `error`) may have
        # sent status:idle and cleared _active_turn_id while we awaited the
        # answer-guard similarity diffs off the event loop above. Re-check the
        # same lifecycle guard used before the barrier so a replacement never
        # lands after this turn's single idle (task 0436 S1 R6 mid-guard).
        if self._finalized or self._is_stale_turn(turn_token):
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
            self._emitted_without_marker = False
            self._delta_counter = 0
            self._no_speech_prefix_guard.reset()
            return
        self._response_buffer = []
        self._marker_seen = False
        self._emitted_without_marker = False
        self._delta_counter = 0
        self._no_speech_prefix_guard.reset()
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

    async def _note_response_text_stream_complete(self, event_type: str, data):
        """Record that the model's text stream is complete and flush any
        buffered pre-marker text.

        This does NOT finalize the turn. Finalization is driven solely by
        response.done so the input-transcription replacement runs as a single
        barrier immediately before status:idle (task 0436 S1). text.done /
        output_text.done only record completion here.
        """
        logger.info(
            f"Handled {event_type} "
            f"(marker_seen={self._marker_seen}, buffer_size={len(self._response_buffer)})"
        )
        if not self._marker_seen and self._emitted_text:
            emitted_len, emitted_sha = answer_guard_fingerprint(self._emitted_text)
            logger.warning(
                "Model output without marker prefix (possible Q&A instead of "
                "transcription): emitted_len=%d emitted_sha1=%s",
                emitted_len,
                emitted_sha,
            )
        if self._response_buffer:
            logger.info("Flushing remaining buffer content")
            flushed = await self._flush_buffer()
            if flushed:
                self._marker_seen = True
        await self._flush_no_speech_prefix_guard()

    async def _on_response_done(self, data):
        # Single finalize barrier (task 0436 S1). Record text completion inline
        # (flush the buffer if response.text.done never arrived), then hand the
        # bounded input-transcription barrier + status:idle to a tracked finalize
        # task and return immediately. The provider receive loop dispatches
        # handlers serially, so awaiting the ASR barrier here would deadlock the
        # dispatcher against the very
        # conversation.item.input_audio_transcription.completed event the barrier
        # waits for when that event is queued just after response.done (task 0436
        # S1 R4). Decoupling keeps the dispatcher consuming so a late-but-within-
        # grace ASR still drives the replacement before the single idle.
        await self._note_response_text_stream_complete("response.done", data)
        self._schedule_finalize("response.done")

    def _is_stale_turn(self, turn_token) -> bool:
        # A new turn (start_recording) reassigns _active_turn_id; a finalize task
        # captured before that must not send onto the new turn (0436 S1 R4).
        return turn_token is not None and turn_token != self._active_turn_id

    def _schedule_finalize(self, reason: str):
        """Spawn the bounded finalize barrier for the current turn exactly once.

        Idempotent: if this turn is already finalized, or a finalize task is
        already scheduled/running for it (a second terminal event, or the
        stop_recording fallback racing response.done), no second task starts.
        The captured turn token lets the task refuse to touch a superseded turn.
        """
        if self._finalized:
            return
        existing = self._finalize_task
        if existing is not None and not existing.done():
            return
        self._finalize_task = asyncio.create_task(
            self._finalize_after_barrier(reason, self._active_turn_id)
        )

    async def _finalize_after_barrier(self, reason: str, turn_token):
        try:
            if self._finalized or self._is_stale_turn(turn_token):
                return
            await self._apply_input_transcription_fallback(reason, turn_token=turn_token)
            if self._finalized or self._is_stale_turn(turn_token):
                return
            await self._finalize_turn(reason)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "Error in finalize barrier (%s): %s", reason, e, exc_info=True
            )

    async def _on_response_text_done(self, data):
        await self._note_response_text_stream_complete("response.text.done", data)

    async def _on_output_audio_transcript_done(self, data):
        await self._note_response_text_stream_complete(
            "response.output_audio_transcript.done", data
        )

    async def _on_generic_event(self, event_type, data):
        if VERBOSE_SERVER_LOG:
            # Log only the event schema (type + top-level keys + body length),
            # never the body itself, which can carry transcript text (0436 M8).
            keys = sorted(data.keys()) if isinstance(data, dict) else []
            body_len = len(json.dumps(data, ensure_ascii=False)) if data else 0
            logger.info(
                "Handled %s (keys=%s body_len=%d)", event_type, keys, body_len
            )
        else:
            logger.debug(f"Handled {event_type}")

    async def _handle_start_recording(self, msg: dict):
        logger.info("Processing start_recording request")
        requested_turn_id = self._normalize_turn_id(msg.get("turn_id"))
        if msg.get("turn_id") is not None and requested_turn_id is None:
            logger.warning(f"Ignoring invalid turn_id from client: {msg.get('turn_id')!r}")
        # A same-turn start_recording is a confirmation-failure retry: keep the
        # PCM already buffered for this turn so nothing is dropped on rebuild
        # (task 0436 S2). A new turn wipes pending as before.
        same_turn_retry = (
            requested_turn_id is not None
            and self._active_turn_id is not None
            and requested_turn_id == self._active_turn_id
            and not self._finalized
        )
        if same_turn_retry and self._pending_audio_chunks:
            logger.info(
                "Same-turn start_recording retry for turn %s; preserving %d "
                "pending audio chunk(s)",
                requested_turn_id,
                len(self._pending_audio_chunks),
            )
        self._reset_turn_state(
            requested_turn_id, preserve_pending_audio=same_turn_retry
        )

        # Update status to connecting while initializing realtime client
        status_payload = {
            "type": "status",
            "status": "connecting"
        }
        if self._active_turn_id is not None:
            status_payload["turn_id"] = self._active_turn_id
        await self._websocket.send_text(json.dumps(status_payload, ensure_ascii=False))
        # `provider` is accepted for backward compatibility but ignored — OpenAI
        # is the only supported provider now (task 0436 F1). A stale value never
        # errors or drops the connection; we just log it once per session.
        legacy_provider = msg.get("provider")
        if legacy_provider is not None and not self._legacy_provider_warned:
            self._legacy_provider_warned = True
            logger.info(
                "Ignoring legacy 'provider' field in start_recording (=%r); "
                "OpenAI is the only supported provider",
                legacy_provider,
            )
        model = msg.get("model")  # OpenAI model name
        # Optimized-only mode: use optimize prompt
        logger.info("Optimized-only mode: using optimize prompt")
        input_sample_rate = msg.get("input_sample_rate")

        logger.info(f"Received start_recording: model={model}")
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

    async def _handle_audio_bytes(self, raw: bytes):
        """Process one inbound PCM chunk: account it, then forward or buffer.

        Buffered when the provider is not ready (init/reconnect in progress) so
        nothing is lost across a confirmation-failure rebuild; the pending
        buffer is flushed to the provider once ready (task 0436 S2). Extracted
        from run() so the audio path is directly testable.
        """
        processed_audio = self._audio_processor.process_audio_chunk(raw)
        if self._is_recording:
            self._processed_audio_bytes += len(processed_audio)
            self._audio_cache.accumulate(processed_audio)
        if not self._openai_ready.is_set():
            logger.debug("Provider not ready, buffering audio chunk")
            self._pending_audio_chunks.append(processed_audio)
        elif self._client and self._is_recording:
            await self._client.send_audio(processed_audio)
            logger.debug(f"Sent audio chunk, size: {len(processed_audio)} bytes")
        else:
            logger.debug("Received audio but client is not initialized")

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
                    await self._handle_audio_bytes(data["bytes"])

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
        # Cancel any in-flight finalize barrier so it never sends onto a torn-down
        # session (task 0436 S1 R4).
        finalize_task = self._finalize_task
        self._finalize_task = None
        if finalize_task is not None and not finalize_task.done():
            finalize_task.cancel()
            try:
                await finalize_task
            except (asyncio.CancelledError, Exception):
                pass
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
