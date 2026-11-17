"""Server-side per-turn PCM audio cache (env-gated, off by default).

Shared by realtime WebSocket servers, WhisperWave sidecar, and micwave_phone.

Default env namespace (``env_prefix="MICWAVE_SERVER_AUDIO_CACHE"``):
  MICWAVE_SERVER_AUDIO_CACHE_ENABLED       "1" to enable (default "0")
  MICWAVE_SERVER_AUDIO_CACHE_DIR           target dir; default
                                           ~/Library/Application Support/Brainwave IME/recent_audio
  MICWAVE_SERVER_AUDIO_FILENAME_PREFIX     prefix to mark the originating service
  MICWAVE_SERVER_AUDIO_CACHE_LIMIT         queue maxsize hint, 0 = unlimited

WhisperWave sidecar passes ``env_prefix="WHISPERWAVE_SERVER_AUDIO_CACHE"`` and
``max_buffer_bytes`` so sidecar memory stays bounded per turn.

Filename pattern: {prefix}{YYYYmmdd_HHMMSS_ffffff}_T{turn_id}_{outcome}.wav
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import wave
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "~/Library/Application Support/Brainwave IME/recent_audio"
DEFAULT_WHISPER_CACHE_DIR = "~/Library/Application Support/EchoWave/whisper/recent_audio"
DEFAULT_WHISPER_MAX_BUFFER_BYTES = 28_800_000


class TurnAudioCache:
    """Per-session audio accumulator + async worker that writes WAV files.

    All public methods are no-ops when constructed with ``enabled=False``, so
    call sites stay branch-free. When ``max_buffer_bytes`` is set, turns whose
    accumulated PCM exceeds the cap discard their sidecar buffer once with a
    warning while allowing live transcription to continue.
    """

    _ENV_PREFIX = "MICWAVE_SERVER_AUDIO_CACHE"
    _DEFAULT_CACHE_DIR = _DEFAULT_DIR
    _DEFAULT_MAX_BUFFER_BYTES: Optional[int] = None

    @classmethod
    def from_env(
        cls,
        *,
        env_prefix: Optional[str] = None,
        default_dir: Optional[str] = None,
        max_buffer_bytes: Optional[int] = None,
    ) -> "TurnAudioCache":
        effective_prefix = env_prefix or cls._ENV_PREFIX
        effective_dir = default_dir or cls._DEFAULT_CACHE_DIR
        effective_cap = (
            max_buffer_bytes
            if max_buffer_bytes is not None
            else cls._DEFAULT_MAX_BUFFER_BYTES
        )
        enabled = os.getenv(f"{effective_prefix}_ENABLED", "0") == "1"
        cache_dir = os.getenv(f"{effective_prefix}_DIR", effective_dir)
        base_prefix = effective_prefix.removesuffix("_CACHE")
        prefix = os.getenv(
            f"{base_prefix}_FILENAME_PREFIX",
            os.getenv(f"{effective_prefix}_FILENAME_PREFIX", ""),
        )
        limit_key = f"{effective_prefix}_LIMIT"
        limit_raw = os.getenv(limit_key, "0")
        try:
            limit = max(0, int(limit_raw))
        except ValueError:
            logger.warning(
                "Invalid %s=%r, falling back to 0",
                limit_key,
                limit_raw,
            )
            limit = 0
        return cls(
            enabled=enabled,
            cache_dir=cache_dir,
            filename_prefix=prefix,
            cache_limit=limit,
            max_buffer_bytes=effective_cap,
        )

    def __init__(
        self,
        enabled: bool,
        cache_dir: str = _DEFAULT_DIR,
        filename_prefix: str = "",
        cache_limit: int = 0,
        max_buffer_bytes: Optional[int] = None,
    ):
        self._enabled = enabled
        self._buffer = bytearray()
        self._current_turn_id: Optional[int] = None
        self._max_buffer_bytes: Optional[int] = (
            max(1, int(max_buffer_bytes)) if max_buffer_bytes is not None else None
        )
        self._turn_discarded = False
        self._enqueued_turn_ids: set = set()
        if not enabled:
            return

        self._cache_dir = os.path.expanduser(cache_dir)
        self._filename_prefix = filename_prefix
        self._cache_limit = max(0, cache_limit)
        maxsize = max(8, self._cache_limit * 2) if self._cache_limit else 0
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"micwave-audio-cache-{filename_prefix or 'default'}",
        )
        self._worker.start()
        if self._max_buffer_bytes is not None:
            logger.info(
                "[audio_cache] enabled dir=%s prefix=%r limit=%d max_buffer_bytes=%d",
                self._cache_dir,
                self._filename_prefix,
                self._cache_limit,
                self._max_buffer_bytes,
            )
        else:
            logger.info(
                "[audio_cache] enabled dir=%s prefix=%r limit=%d",
                self._cache_dir,
                self._filename_prefix,
                self._cache_limit,
            )

    def start_turn(self, turn_id: Optional[int]) -> None:
        if not self._enabled:
            return
        self._buffer.clear()
        self._current_turn_id = turn_id
        self._turn_discarded = False

    def accumulate(self, pcm_bytes: bytes) -> None:
        if not self._enabled or not pcm_bytes or self._turn_discarded:
            return
        if (
            self._max_buffer_bytes is not None
            and len(self._buffer) + len(pcm_bytes) > self._max_buffer_bytes
        ):
            self._buffer.clear()
            self._turn_discarded = True
            logger.warning(
                "[audio_cache] server-cache sidecar exceeded cap and was dropped "
                "for turn %s (max_buffer_bytes=%d); transcription continues",
                self._current_turn_id,
                self._max_buffer_bytes,
            )
            return
        self._buffer.extend(pcm_bytes)

    def enqueue_turn(
        self,
        turn_id: Optional[int],
        outcome: str,
        sample_rate: int,
        channels: int = 1,
    ) -> bool:
        return self.enqueue_turn_file(turn_id, outcome, sample_rate, channels) is not None

    def enqueue_turn_file(
        self,
        turn_id: Optional[int],
        outcome: str,
        sample_rate: int,
        channels: int = 1,
    ) -> Optional[str]:
        """Queue the turn's WAV; return the basename the worker will write, or
        None when nothing was queued (disabled, empty buffer, duplicate, full).
        The name is fixed here, so callers can reference the file before the
        worker has written it."""
        if not self._enabled or not self._buffer:
            return None
        if turn_id is not None and turn_id in self._enqueued_turn_ids:
            return None
        file_name = self._file_name(turn_id, outcome)
        task = {
            "turn_id": turn_id,
            "file_name": file_name,
            "audio_bytes": bytes(self._buffer),
            "sample_rate": sample_rate,
            "channels": channels,
        }
        self._buffer.clear()
        ok = self._enqueue(task)
        if ok and turn_id is not None:
            self._enqueued_turn_ids.add(turn_id)
        elif not ok:
            logger.warning("[audio_cache] queue full, dropped turn %s", turn_id)
        return file_name if ok else None

    def _file_name(self, turn_id: Optional[int], outcome: str) -> str:
        safe_outcome = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(outcome)
        ).strip("_") or "unknown"
        turn_label = f"T{turn_id}" if turn_id is not None else "Tunknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{self._filename_prefix}{timestamp}_{turn_label}_{safe_outcome}.wav"

    def close(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _enqueue(self, task: dict) -> bool:
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                if dropped is None:
                    self._queue.put_nowait(None)
            except Exception:
                pass
            try:
                self._queue.put_nowait(task)
                return True
            except Exception:
                return False

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                task = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if task is None:
                    break
                path = self._write_wav(task)
                if path:
                    logger.info("[audio_cache] wrote %s", path)
            except Exception as exc:
                logger.error("[audio_cache] worker error: %s", exc, exc_info=True)
            finally:
                self._queue.task_done()

    def _write_wav(self, task: dict) -> Optional[str]:
        audio_bytes = task.get("audio_bytes")
        if not audio_bytes:
            return None
        sample_rate = int(task.get("sample_rate", 24000))
        channels = int(task.get("channels", 1))

        os.makedirs(self._cache_dir, exist_ok=True)
        file_path = os.path.join(self._cache_dir, task["file_name"])
        with wave.open(file_path, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        return file_path


class WhisperTurnAudioCache(TurnAudioCache):
    """WhisperWave-namespaced ``TurnAudioCache`` (``WHISPERWAVE_SERVER_AUDIO_CACHE_*``)."""

    _ENV_PREFIX = "WHISPERWAVE_SERVER_AUDIO_CACHE"
    _DEFAULT_CACHE_DIR = DEFAULT_WHISPER_CACHE_DIR
    _DEFAULT_MAX_BUFFER_BYTES = DEFAULT_WHISPER_MAX_BUFFER_BYTES

    def __init__(
        self,
        enabled: bool,
        cache_dir: str = DEFAULT_WHISPER_CACHE_DIR,
        filename_prefix: str = "",
        cache_limit: int = 0,
        max_buffer_bytes: Optional[int] = DEFAULT_WHISPER_MAX_BUFFER_BYTES,
    ):
        super().__init__(
            enabled=enabled,
            cache_dir=cache_dir,
            filename_prefix=filename_prefix,
            cache_limit=cache_limit,
            max_buffer_bytes=max_buffer_bytes,
        )
