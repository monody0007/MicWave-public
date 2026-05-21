"""Server-side per-turn PCM audio cache (env-gated, off by default).

Mirrors the IME client's recent_audio cache but lives on the realtime
server, so deployments without the IME (e.g. browser-based clients in
anydoor7) can still persist raw audio. MicWave IME deployments leave this
disabled because the client already caches audio on the user machine.

Env (read once at construction):
  MICWAVE_SERVER_AUDIO_CACHE_ENABLED   "1" to enable (default "0")
  MICWAVE_SERVER_AUDIO_CACHE_DIR       target dir; default
                                       ~/Library/Application Support/Brainwave IME/recent_audio
  MICWAVE_SERVER_AUDIO_FILENAME_PREFIX prefix to mark the originating service
                                       (e.g. "anydoor7_" → distinguishes from
                                       IME-cached files at a glance in `ls`)
  MICWAVE_SERVER_AUDIO_CACHE_LIMIT     queue maxsize hint, 0 = unlimited

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


class TurnAudioCache:
    """Per-WebSocket audio accumulator + async worker that writes WAV files.

    All public methods are no-ops when constructed with enabled=False, so call
    sites stay branch-free.
    """

    @classmethod
    def from_env(cls) -> "TurnAudioCache":
        enabled = os.getenv("MICWAVE_SERVER_AUDIO_CACHE_ENABLED", "0") == "1"
        cache_dir = os.getenv("MICWAVE_SERVER_AUDIO_CACHE_DIR", _DEFAULT_DIR)
        prefix = os.getenv("MICWAVE_SERVER_AUDIO_FILENAME_PREFIX", "")
        limit_raw = os.getenv("MICWAVE_SERVER_AUDIO_CACHE_LIMIT", "0")
        try:
            limit = max(0, int(limit_raw))
        except ValueError:
            logger.warning(
                "Invalid MICWAVE_SERVER_AUDIO_CACHE_LIMIT=%r, falling back to 0",
                limit_raw,
            )
            limit = 0
        return cls(
            enabled=enabled,
            cache_dir=cache_dir,
            filename_prefix=prefix,
            cache_limit=limit,
        )

    def __init__(
        self,
        enabled: bool,
        cache_dir: str = _DEFAULT_DIR,
        filename_prefix: str = "",
        cache_limit: int = 0,
    ):
        self._enabled = enabled
        self._buffer = bytearray()
        self._current_turn_id: Optional[int] = None
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

    def accumulate(self, pcm_bytes: bytes) -> None:
        if not self._enabled or not pcm_bytes:
            return
        self._buffer.extend(pcm_bytes)

    def enqueue_turn(
        self,
        turn_id: Optional[int],
        outcome: str,
        sample_rate: int,
        channels: int = 1,
    ) -> bool:
        if not self._enabled or not self._buffer:
            return False
        # Dedup: stop_recording and finalize_turn may both fire for the same turn.
        if turn_id is not None and turn_id in self._enqueued_turn_ids:
            return False
        task = {
            "turn_id": turn_id,
            "outcome": outcome,
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
        return ok

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
        turn_id = task.get("turn_id")
        outcome = str(task.get("outcome", "unknown"))
        sample_rate = int(task.get("sample_rate", 24000))
        channels = int(task.get("channels", 1))

        safe_outcome = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in outcome
        ).strip("_") or "unknown"
        turn_label = f"T{turn_id}" if turn_id is not None else "Tunknown"

        os.makedirs(self._cache_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"{self._filename_prefix}{timestamp}_{turn_label}_{safe_outcome}.wav"
        file_path = os.path.join(self._cache_dir, file_name)
        with wave.open(file_path, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        return file_path
