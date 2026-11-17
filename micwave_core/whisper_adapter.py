"""WhisperWave sidecar building blocks: dotenv loader, silence-aligned soft
segmentation, and OpenAI ``type:"transcription"`` WebSocket client.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import websockets

from .config import (
    DEFAULT_WHISPER_TRANSCRIBE_MODEL,
    WHISPER_TRANSCRIBE_MODEL,
    WHISPERWAVE_DELAY,
    WHISPERWAVE_INPUT_SAMPLE_RATE,
    WHISPERWAVE_LANGUAGE,
    WHISPERWAVE_NOISE_REDUCTION,
    transcription_delay_for_model,
)
from .prompts import TRANSCRIPTION_BIAS_PROMPT, transcription_prompt_for_model
from .realtime_client_base import RealtimeClientBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ── Allowlisted, space-preserving, secret-safe sidecar dotenv loader ──
DOTENV_ALLOWED_KEYS = frozenset(
    {
        "WHISPERWAVE_DEBUG_LOG",
        "WHISPERWAVE_DELAY",
        "WHISPERWAVE_FORCE_RECONNECT_AFTER_MS",
        "WHISPERWAVE_KEEP_PROVIDER_SESSION",
        "WHISPERWAVE_LANGUAGE",
        "WHISPERWAVE_MAX_TURN_AUDIO_BYTES",
        "WHISPERWAVE_MODALITIES",
        "WHISPERWAVE_MODEL",
        "WHISPERWAVE_NOISE_REDUCTION",
        "WHISPERWAVE_PORT",
        "WHISPERWAVE_PROVIDER_EVENT_METRICS",
        "WHISPERWAVE_PROVIDER_INIT_MAX_ATTEMPTS",
        "WHISPERWAVE_PROVIDER_INIT_RETRY_DELAY_SEC",
        "WHISPERWAVE_PROVIDER_SESSION_MAX_AGE_SEC",
        "WHISPERWAVE_PROVIDER_SESSION_MAX_TURNS",
        "WHISPERWAVE_SERVER_AUDIO_CACHE_DIR",
        "WHISPERWAVE_SERVER_AUDIO_CACHE_ENABLED",
        "WHISPERWAVE_SERVER_AUDIO_CACHE_LIMIT",
        "WHISPERWAVE_SERVER_AUDIO_FILENAME_PREFIX",
        "WHISPERWAVE_SOFT_SEGMENT_GRACE_SEC",
        "WHISPERWAVE_SOFT_SEGMENT_SEC",
        "WHISPERWAVE_SOFT_SEGMENT_SILENCE_RMS",
        "WHISPERWAVE_TRANSCRIPTION_ACK_GRACE_SEC",
        "WHISPERWAVE_TRANSCRIPTION_FAILURE_ROTATE_THRESHOLD",
        "WHISPERWAVE_TRANSCRIPTION_FINALIZE_TIMEOUT_SEC",
        "WHISPERWAVE_VERBOSE_SERVER_LOG",
    }
)


def parse_env_file(path) -> list[tuple[str, str]]:
    """Return allowlist-filtered ``(key, value)`` pairs preserving inner spaces."""
    pairs: list[tuple[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].lstrip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if key not in DOTENV_ALLOWED_KEYS:
                    continue
                pairs.append((key, value))
    except FileNotFoundError:
        pass
    return pairs


def load_into_environ(path, override: bool = False) -> list[str]:
    """Load allowlisted keys without overwriting already-set environment keys."""
    del override
    loaded: list[str] = []
    for key, value in parse_env_file(path):
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def _emit_shell(path) -> None:
    for key, value in parse_env_file(path):
        sys.stdout.write(f"{key}={value}\n")


# ── Silence-aligned soft segment boundary tracker ──
WINDOW_MS = 20
MIN_SILENCE_MS = 300
DEFAULT_SILENCE_RMS = 150.0
MAX_SILENCE_RMS = 32_768.0
MAX_SEGMENT_SEC = 86_400.0
DEFAULT_SOFT_SEGMENT_SEC = 60.0
DEFAULT_SOFT_GRACE_SEC = 30.0


def window_bytes(sample_rate: int, window_ms: int = WINDOW_MS) -> int:
    return max(1, sample_rate * window_ms // 1000) * 2


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class SoftSegmentThresholds:
    soft_bytes: int
    force_bytes: int
    min_silence_bytes: int
    rms_threshold: float

    def reason(self, segment_bytes: int, quiet_run_bytes: int) -> Optional[str]:
        if segment_bytes < self.soft_bytes:
            return None
        if quiet_run_bytes >= self.min_silence_bytes:
            return "silence"
        if segment_bytes >= self.force_bytes:
            return "forced"
        return None


def resolve_soft_segment_thresholds(
    *,
    soft_sec,
    grace_sec,
    rms_threshold,
    bytes_per_sec: int,
    min_commit_bytes: int,
    hard_cap_bytes: int,
) -> Optional[SoftSegmentThresholds]:
    if not _is_finite_number(soft_sec):
        logger.warning("Soft segment target %r is not a finite number; disabled", soft_sec)
        return None
    if soft_sec <= 0:
        return None
    if soft_sec > MAX_SEGMENT_SEC:
        logger.warning(
            "Soft segment target %r s exceeds %.0f s; disabled", soft_sec, MAX_SEGMENT_SEC
        )
        return None
    if not _is_finite_number(grace_sec) or not 0 <= grace_sec <= MAX_SEGMENT_SEC:
        logger.warning(
            "Soft segment grace %r s is out of range; using %.0f s",
            grace_sec,
            DEFAULT_SOFT_GRACE_SEC,
        )
        grace_sec = DEFAULT_SOFT_GRACE_SEC
    if not _is_finite_number(rms_threshold) or not 0 < rms_threshold <= MAX_SILENCE_RMS:
        logger.warning(
            "Soft segment silence RMS %r is out of range; using %.0f",
            rms_threshold,
            DEFAULT_SILENCE_RMS,
        )
        rms_threshold = DEFAULT_SILENCE_RMS

    soft_bytes = int(soft_sec * bytes_per_sec) // 2 * 2
    if soft_bytes < min_commit_bytes:
        logger.info(
            "Soft segment target %.3f s is below the %d-byte provider minimum; "
            "raised to the minimum",
            soft_sec,
            min_commit_bytes,
        )
        soft_bytes = min_commit_bytes
    if soft_bytes >= hard_cap_bytes:
        return None
    return SoftSegmentThresholds(
        soft_bytes=soft_bytes,
        force_bytes=soft_bytes + int(grace_sec * bytes_per_sec) // 2 * 2,
        min_silence_bytes=MIN_SILENCE_MS * bytes_per_sec // 1000,
        rms_threshold=float(rms_threshold),
    )


class QuietRunTracker:
    """Track trailing run of consecutive quiet windows in a PCM16 stream."""

    def __init__(self, *, sample_rate: int, rms_threshold: float = DEFAULT_SILENCE_RMS):
        if not _is_finite_number(rms_threshold) or not 0 < rms_threshold <= MAX_SILENCE_RMS:
            raise ValueError(
                f"rms_threshold must be in (0, {MAX_SILENCE_RMS:.0f}], got {rms_threshold!r}"
            )
        self._window_bytes = window_bytes(sample_rate)
        self._threshold_sq = float(rms_threshold) ** 2
        self._carry = b""
        self.run_bytes = 0

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        data = self._carry + chunk if self._carry else chunk
        windows = len(data) // self._window_bytes
        usable = windows * self._window_bytes
        self._carry = bytes(data[usable:])
        if not windows:
            return
        samples = np.frombuffer(data, dtype="<i2", count=usable // 2)
        energy = (samples.reshape(windows, -1).astype(np.float64) ** 2).mean(axis=1)
        loud = np.flatnonzero(energy >= self._threshold_sq)
        if loud.size == 0:
            self.run_bytes += usable
        else:
            self.run_bytes = (windows - 1 - int(loud[-1])) * self._window_bytes


# ── OpenAI Realtime transcription-session client ──
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
        del modalities, instructions
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            self.ws = await websockets.connect(self.connect_url, extra_headers=headers)
        except TypeError:
            self.ws = await websockets.connect(self.connect_url, additional_headers=headers)

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
            if first_frame_type == "error":
                err = response_data.get("error", {})
                logger.error(
                    "OpenAI connect: first frame is an error (type=%s code=%s) — aborting",
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
        await self.ws.send(
            json.dumps(
                {"type": "session.update", "session": session_config_payload},
                ensure_ascii=False,
            )
        )

        confirmed = await self._recv_session_updated(timeout=5.0)
        if not confirmed:
            logger.warning("session.updated not confirmed; resending session.update once")
            await self.ws.send(
                json.dumps(
                    {"type": "session.update", "session": session_config_payload},
                    ensure_ascii=False,
                )
            )
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
            await self.ws.send(
                json.dumps(
                    {"type": "session.update", "session": session_config_payload},
                    ensure_ascii=False,
                )
            )
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
        del modalities, instructions
        if not self._is_ws_open():
            raise RuntimeError("WebSocket is not open. Cannot refresh transcription session.")

        session_config_payload = self._build_session_config()
        confirmed = await self._send_session_update_and_wait(session_config_payload, timeout=5.0)
        if not confirmed:
            logger.warning("session.updated not confirmed on refresh; resending once")
            confirmed = await self._send_session_update_and_wait(
                session_config_payload, timeout=5.0
            )
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
                    logger.warning("No handler for message type: %s", message_type)
        except websockets.exceptions.ConnectionClosed as e:
            logger.error("OpenAI WebSocket connection closed: %s", e)
        except asyncio.CancelledError:
            logger.info("OpenAI receive_messages task cancelled")
            return
        except Exception as e:
            logger.error("Error in receive_messages: %s", e, exc_info=True)
        await self._fire_on_disconnect()

    async def default_handler(self, data: dict):
        message_type = data.get("type", "unknown")
        logger.warning("Unhandled message type received from OpenAI: %s", message_type)

    async def send_audio(self, audio_data: bytes) -> bool:
        if self._is_ws_open():
            await self.ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio_data).decode("utf-8"),
                    }
                )
            )
            return True
        now = time.time()
        if now - self._last_send_error_log >= 5.0:
            logger.error(
                "WebSocket is not open. Cannot send audio. (suppressing repeats for 5s)"
            )
            self._last_send_error_log = now
        return False

    async def commit_audio(self):
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


__all__ = [
    "DEFAULT_SILENCE_RMS",
    "DEFAULT_SOFT_GRACE_SEC",
    "DEFAULT_SOFT_SEGMENT_SEC",
    "DEFAULT_WHISPER_TRANSCRIBE_MODEL",
    "DOTENV_ALLOWED_KEYS",
    "MAX_SEGMENT_SEC",
    "MAX_SILENCE_RMS",
    "MIN_SILENCE_MS",
    "QuietRunTracker",
    "SoftSegmentThresholds",
    "TRANSCRIPTION_BIAS_PROMPT",
    "WHISPER_TRANSCRIBE_MODEL",
    "WINDOW_MS",
    "WhisperRealtimeTranscriptionClient",
    "_emit_shell",
    "load_into_environ",
    "parse_env_file",
    "resolve_soft_segment_thresholds",
    "transcription_prompt_for_model",
    "window_bytes",
]
