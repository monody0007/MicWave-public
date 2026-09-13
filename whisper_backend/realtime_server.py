"""WhisperWave realtime transcription server.

Same menubar↔server WebSocket protocol as MicWave (start_recording / binary PCM /
stop_recording; replies {type:text,...} + {type:status,...}), but the OpenAI side
is a pure `type:"transcription"` session (gpt-4o-transcribe by default). Consequences:

  * A vocabulary-bias transcription prompt may steer spelling, but there is no
    response marker or answer-similarity guard because the session has no answer
    surface.
  * Output is driven by conversation.item.input_audio_transcription.delta /
    .completed, aligned to the committed item via item_id (the API does not
    guarantee cross-turn ordering, so we match on item_id).
  * stop_recording commits the buffer (no response.create); the `.completed`
    event is authoritative and, if it diverges from the streamed deltas, replaces
    the on-screen text via isNewResponse=True strictly BEFORE status:idle (the S1
    invariant carried over from MicWave).
"""
import asyncio
import collections
import enum
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

from .audio_persistence import TurnAudioCache
from .config import WHISPER_TRANSCRIBE_MODEL
from .realtime_client_base import RealtimeClientBase
from .realtime_text_utils import normalize_final_transcript
from .transcript_merge import append_transcript_delta
from .whisper_realtime_client import WhisperRealtimeTranscriptionClient

logging.basicConfig(
    # DEBUG surfaces the audio-ingest path (buffer/send/late-frame) for probes and
    # incident triage; default INFO keeps steady-state logs quiet. Frame logs carry
    # only metadata (never transcript/audio content), so DEBUG stays privacy-safe.
    level=logging.DEBUG if os.getenv("WHISPERWAVE_DEBUG_LOG") == "1" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
# Root DEBUG is useful for our metadata-only ingest diagnostics, but the
# `websockets` package logs raw wire frames at DEBUG (including audio base64 and
# transcript text). Keep third-party wire logging fail-closed regardless of the
# application debug setting.
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
VERBOSE_SERVER_LOG = os.getenv("WHISPERWAVE_VERBOSE_SERVER_LOG", "0") == "1"

app = FastAPI()


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "backend": "whisper",
        "contract": {"name": "echowave-pcm-websocket", "version": 1},
        "model": WHISPER_TRANSCRIBE_MODEL,
    }


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class AudioProcessor:
    def __init__(self, target_sample_rate=24000, source_sample_rate=48000):
        self.target_sample_rate = target_sample_rate
        self.source_sample_rate = source_sample_rate

    def set_source_sample_rate(self, sample_rate: int):
        self.source_sample_rate = sample_rate

    def process_audio_chunk(self, audio_data):
        if self.source_sample_rate == self.target_sample_rate:
            return audio_data
        pcm_data = np.frombuffer(audio_data, dtype=np.int16)
        float_data = pcm_data.astype(np.float32) / 32768.0
        resampled_data = scipy.signal.resample_poly(
            float_data, self.target_sample_rate, self.source_sample_rate
        )
        resampled_int16 = (resampled_data * 32768.0).clip(-32768, 32767).astype(np.int16)
        return resampled_int16.tobytes()


class _TaskOwnedLock(asyncio.Lock):
    """An asyncio lock whose invariants can verify the *current* task owns it."""

    def __init__(self):
        super().__init__()
        self._owner: Optional[asyncio.Task] = None

    async def acquire(self):
        acquired = await super().acquire()
        if acquired:
            self._owner = asyncio.current_task()
        return acquired

    def release(self):
        if self._owner is not asyncio.current_task():
            raise RuntimeError("transport lock may only be released by its owner")
        self._owner = None
        super().release()

    def owned_by_current_task(self) -> bool:
        return self.locked() and self._owner is asyncio.current_task()


@dataclass(frozen=True)
class TranscriptionTurnSessionConfig:
    keep_provider_session: bool
    provider_session_max_turns: int
    provider_session_max_age_sec: int
    provider_init_max_attempts: int
    provider_init_retry_delay_sec: float
    transcription_finalize_timeout_sec: float
    transcription_ack_grace_sec: float
    transcription_failure_rotate_threshold: int
    default_source_sample_rate: int
    # 300s PCM16/24k segment boundary. Reaching it commits the current segment
    # instead of terminating the turn; the bound remains the replay-memory guard.
    max_turn_audio_bytes: int = 14_400_000
    force_reconnect_after_ms: int = 0

    @classmethod
    def from_env(cls) -> "TranscriptionTurnSessionConfig":
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
            keep_provider_session=os.getenv("WHISPERWAVE_KEEP_PROVIDER_SESSION", "1") == "1",
            provider_session_max_turns=int_env("WHISPERWAVE_PROVIDER_SESSION_MAX_TURNS", 8, 0),
            # Preventive rebuild aligned to the 60-minute Realtime session cap:
            # rotate at <=55 min (or the turn cap, whichever comes first).
            provider_session_max_age_sec=int_env("WHISPERWAVE_PROVIDER_SESSION_MAX_AGE_SEC", 3300, 0),
            provider_init_max_attempts=int_env("WHISPERWAVE_PROVIDER_INIT_MAX_ATTEMPTS", 3, 1),
            provider_init_retry_delay_sec=float_env("WHISPERWAVE_PROVIDER_INIT_RETRY_DELAY_SEC", 0.5, 0.0),
            transcription_finalize_timeout_sec=float_env("WHISPERWAVE_TRANSCRIPTION_FINALIZE_TIMEOUT_SEC", 120.0, 5.0),
            # Bounded grace after an early terminal (a completed/failed that raced
            # ahead of its own commit ACK): if the ACK never arrives within this
            # window we finalize with a failure reason and rotate, rather than
            # hanging until the 120s stop timeout (H1 residual). 0 disables.
            transcription_ack_grace_sec=float_env("WHISPERWAVE_TRANSCRIPTION_ACK_GRACE_SEC", 2.0, 0.0),
            transcription_failure_rotate_threshold=int_env("WHISPERWAVE_TRANSCRIPTION_FAILURE_ROTATE_THRESHOLD", 2, 1),
            default_source_sample_rate=AudioProcessor().source_sample_rate,
            max_turn_audio_bytes=int_env("WHISPERWAVE_MAX_TURN_AUDIO_BYTES", 14_400_000, 48_000),
            force_reconnect_after_ms=int_env("WHISPERWAVE_FORCE_RECONNECT_AFTER_MS", 0, 0),
        )


class TransportState(enum.Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    RECORDING = "recording"
    RECOVERING = "recovering"
    STOP_REQUESTED = "stop_requested"
    COMMITTING = "committing"
    COMMIT_ACKED = "commit_acked"
    FINALIZED = "finalized"


class TranscriptionTurnSession:
    _RECONNECT_MAX_ATTEMPTS = 5
    _RECONNECT_BACKOFF_BASE_SEC = 0.3
    _MIN_COMMIT_AUDIO_BYTES = 4_800  # 100ms of PCM16 mono at 24 kHz
    _FINALIZE_MAX_ATTEMPTS = 2
    # Bound on remembered retired item ids (past turns / superseded tentative
    # bindings). A session rotates at 8 turns / 55 min, so this never fills in
    # practice; the cap only guards a pathological single long-lived session.
    _RETIRED_ITEMS_CAP = 64
    # Bound on terminal events (completed/failed) that raced ahead of their own
    # commit ACK and are stashed awaiting confirmation (H1 residual). A turn has
    # one real item, so this only ever holds a couple of same/foreign ids.
    _PENDING_TERMINALS_CAP = 4

    _SUCCESSFUL_FINALIZE_REASONS = {
        "transcription.completed",
        "digital_silence",
    }

    def __init__(self, websocket: WebSocket, config: TranscriptionTurnSessionConfig):
        self._websocket = websocket
        self._config = config

        self._client: Optional[RealtimeClientBase] = None
        self._active_model: Optional[str] = None
        self._provider_session_turns = 0
        self._provider_session_started_at: Optional[float] = None
        self._consecutive_transcription_failures = 0
        self._openai_ready = asyncio.Event()
        self._audio_processor = AudioProcessor(source_sample_rate=config.default_source_sample_rate)
        self._audio_cache = TurnAudioCache.from_env(
            max_buffer_bytes=config.max_turn_audio_bytes * 2
        )

        # One owner serializes cache mutation and every provider transport write;
        # stop intent alone may latch before waiting for that owner.
        self._transport_lock = _TaskOwnedLock()
        # Text mutation + client text writes share an ordering barrier with
        # final replacement and idle (S1: never text after idle).
        self._output_lock = asyncio.Lock()

        self._active_turn_id: Optional[int] = None
        self._finalized = False
        self._finalizing = False
        self._is_recording = False
        self._turn_done: Optional[asyncio.Event] = asyncio.Event()
        self._transport_state = TransportState.IDLE
        self._turn_generation = 1
        self._provider_generation = 0
        self._commit_key: Optional[tuple[Optional[int], int, int]] = None
        self._finalize_commit_attempts = 0
        # One append-before-send cache; replacements replay it from offset zero.
        # The byte counter covers successful sends to the current provider only.
        self._turn_audio_chunks: list[bytes] = []
        self._turn_audio_bytes = 0
        self._provider_audio_bytes = 0
        # Direction 2 (commit-then-rebuild): while the bounded prefix waits for
        # its authoritative completed event, subsequent client audio is retained
        # separately. Once the prefix is finalized, this tail becomes the next
        # provider buffer and is replayed into a fresh session.
        self._segment_rollover_pending = False
        self._segment_rollover_done = asyncio.Event()
        self._segment_rollover_done.set()
        self._rollover_tail_chunks: list[bytes] = []
        self._rollover_tail_bytes = 0
        self._committed_transcript_text = ""
        self._stop_requested = False

        # Item-identity state machine (H1): an item id moves tentative (a pre-ACK
        # delta may optimistically bind it so streaming is not delayed) →
        # confirmed (input_audio_buffer.committed ACK) → retired (past turns or a
        # superseded tentative binding; permanently rejected). Only a *confirmed*
        # item may trigger terminal authority (finalize / replacement).
        self._item_confirmed = False
        self._retired_item_ids: set[str] = set()
        self._retired_item_order: collections.deque = collections.deque()
        # Terminal events that arrived before their own commit ACK, keyed by
        # item_id. The ACK confirms the item and replays the stashed terminal;
        # an ACK-grace timer finalizes if the ACK never arrives (H1 residual).
        self._pending_terminals: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
        self._ack_grace_task: Optional[asyncio.Task] = None

        # Transcription-turn state
        self._current_item_id: Optional[str] = None
        # Partial candidates are provider-epoch scoped: replaying the full cache
        # must not append a replacement epoch's full text onto the old epoch.
        self._transcript_text = ""          # longest nonblank partial candidate
        self._provider_epoch_text = ""      # deltas from the current provider
        self._best_partial_text = ""         # best completed prior provider epoch
        self._provider_completed_text = ""  # raw authoritative provider final
        self._normalized_final_text = ""    # final-only local policy output
        self._transcript_completed = False
        self._emitted_text = ""             # raw stream or final replacement
        self._processed_audio_bytes = 0

        self._legacy_provider_warned = False
        self._closed = False
        self._finalize_task: Optional[asyncio.Task] = None
        self._finalize_retry_task: Optional[asyncio.Task] = None
        self._force_reconnect_task: Optional[asyncio.Task] = None
        self._segment_rollover_task: Optional[asyncio.Task] = None
        self._segment_rollover_timeout_task: Optional[asyncio.Task] = None

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_turn_id(raw_turn_id) -> Optional[int]:
        if raw_turn_id is None:
            return None
        try:
            return int(raw_turn_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_delta(data: dict) -> str:
        for key in ("delta", "text", "transcript"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    @staticmethod
    def _extract_completed(data: dict) -> str:
        for key in ("transcript", "text", "delta"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        item = data.get("item")
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, list):
                parts = [
                    p.get("transcript") or p.get("text")
                    for p in content
                    if isinstance(p, dict) and (p.get("transcript") or p.get("text"))
                ]
                if parts:
                    return "".join(parts)
        return ""

    @staticmethod
    def _select_completed_authority(completed: str, partial: str) -> str:
        """Resolve finalize text defensively by keeping the longer nonblank view."""

        trimmed_completed = completed.strip()
        trimmed_partial = partial.strip()
        if not trimmed_partial:
            return trimmed_completed
        if not trimmed_completed:
            return partial
        return partial if len(trimmed_partial) >= len(trimmed_completed) else trimmed_completed

    @staticmethod
    def _select_partial_candidate(current: str, previous: str) -> str:
        """Keep the longer provider-epoch partial without concatenating epochs."""

        trimmed_current = current.strip()
        trimmed_previous = previous.strip()
        if not trimmed_previous:
            return current
        if not trimmed_current:
            return previous
        return current if len(trimmed_current) >= len(trimmed_previous) else previous

    @staticmethod
    def _is_recoverable_buffer_too_small(data: dict) -> bool:
        error = data.get("error", {}) if isinstance(data, dict) else {}
        return isinstance(error, dict) and "buffer too small" in str(
            error.get("message", "")
        ).casefold()

    def _reset_turn_state(
        self,
        active_turn_id: Optional[int] = None,
    ):
        self._assert_transport_locked()
        finalize_task = getattr(self, "_finalize_task", None)
        if finalize_task is not None and not finalize_task.done():
            finalize_task.cancel()
        self._finalize_task = None
        self._cancel_finalize_retry_task()
        ack_grace_task = getattr(self, "_ack_grace_task", None)
        if ack_grace_task is not None and not ack_grace_task.done():
            ack_grace_task.cancel()
        self._ack_grace_task = None
        self._cancel_force_reconnect_task()
        self._cancel_segment_rollover_task()
        self._cancel_segment_rollover_timeout_task()
        self._segment_rollover_done.set()
        self._segment_rollover_done = asyncio.Event()
        self._segment_rollover_done.set()
        self._pending_terminals = collections.OrderedDict()
        # Retire the previous turn's bound item so a late/stale event from it can
        # never rebind or finalize the new turn (H1).
        prev_item = getattr(self, "_current_item_id", None)
        if prev_item:
            self._retire_item(prev_item)
        self._turn_generation += 1
        self._active_turn_id = active_turn_id
        self._finalized = False
        self._finalizing = False
        self._is_recording = False
        self._transport_state = TransportState.IDLE
        self._commit_key = None
        self._finalize_commit_attempts = 0
        self._turn_done = asyncio.Event()
        self._turn_audio_chunks = []
        self._turn_audio_bytes = 0
        self._provider_audio_bytes = 0
        self._segment_rollover_pending = False
        self._rollover_tail_chunks = []
        self._rollover_tail_bytes = 0
        self._committed_transcript_text = ""
        self._stop_requested = False
        self._current_item_id = None
        self._item_confirmed = False
        self._transcript_text = ""
        self._provider_epoch_text = ""
        self._best_partial_text = ""
        self._provider_completed_text = ""
        self._normalized_final_text = ""
        self._transcript_completed = False
        self._emitted_text = ""
        self._processed_audio_bytes = 0

    def _retire_item(self, item_id: Optional[str]) -> None:
        """Permanently reject future events for this item id (bounded memory)."""
        if not item_id or item_id in self._retired_item_ids:
            return
        self._retired_item_ids.add(item_id)
        self._retired_item_order.append(item_id)
        while len(self._retired_item_order) > self._RETIRED_ITEMS_CAP:
            evicted = self._retired_item_order.popleft()
            self._retired_item_ids.discard(evicted)

    def _is_stale_turn(self, turn_token) -> bool:
        return turn_token is not None and turn_token != self._active_turn_id

    async def _send_text_payload(self, content: str):
        """Send one delta while the caller owns `_output_lock`."""
        if content and self._websocket.client_state == WebSocketState.CONNECTED:
            payload = {"type": "text", "content": content, "isNewResponse": False}
            if self._active_turn_id is not None:
                payload["turn_id"] = self._active_turn_id
            await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
            self._emitted_text = append_transcript_delta(self._emitted_text, content)

    async def _emit_text_delta(self, content: str):
        """Emit provider partials verbatim; semantic policy is final-only."""
        await self._send_text_payload(content)

    def _current_emitted_view(self) -> str:
        return self._emitted_text

    def _match_item(self, data: dict, event_name: str) -> bool:
        """Gate a streaming delta through the identity state machine.

        Deltas may OPTIMISTICALLY bind an item before the commit ACK so streaming
        is not delayed (live logs show a delta can precede the client commit by
        ~12ms). A tentative binding never grants terminal authority — that is
        reserved for a commit-ACK-confirmed item (see _classify_terminal).
        Retired item ids (past turns, or a tentative binding superseded by the
        ACK) are always rejected.
        """
        ev_item = data.get("item_id")
        if ev_item and ev_item in self._retired_item_ids:
            logger.warning("Dropping %s for retired item_id=%s", event_name, ev_item)
            return False
        if self._current_item_id is None:
            if ev_item:
                self._current_item_id = ev_item
                self._item_confirmed = False
                logger.info(
                    "Tentatively bound item_id=%s (%s, pre-ACK streaming)",
                    ev_item, event_name,
                )
            # No item_id yet: allow the streamed text through; the ACK will bind
            # and confirm the real item.
            return True
        if ev_item is not None and ev_item != self._current_item_id:
            logger.warning(
                "Dropping %s for foreign item_id=%s (active item_id=%s)",
                event_name, ev_item, self._current_item_id,
            )
            return False
        return True

    def _classify_terminal(self, data: dict, event_name: str) -> str:
        """Classify a terminal (completed/failed) event: 'accept'|'pending'|'drop'.

        Terminal authority is fail-closed and NEVER self-binds an item:

          * missing item_id         -> drop (malformed; the official schema makes
                                       item_id required on committed/terminal
                                       events, so a missing one has no authority)
          * retired item_id         -> drop (past turn / superseded binding)
          * confirmed + matching id -> accept (normal terminal authority)
          * confirmed + foreign id  -> drop
          * unconfirmed + active commit -> pending (the terminal may have raced
                                           ahead of its own commit ACK)
          * unconfirmed + no commit     -> drop (pre-commit provider noise must
                                           never terminate active recording)
        """
        ev_item = data.get("item_id")
        if not ev_item:
            logger.warning("Fail-closed: %s missing item_id; not finalizing", event_name)
            return "drop"
        if ev_item in self._retired_item_ids:
            logger.warning("Dropping terminal %s for retired item_id=%s", event_name, ev_item)
            return "drop"
        if self._item_confirmed:
            if ev_item == self._current_item_id:
                return "accept"
            logger.warning(
                "Dropping terminal %s for foreign item_id=%s (confirmed item_id=%s)",
                event_name, ev_item, self._current_item_id,
            )
            return "drop"
        if not self._has_active_commit():
            logger.warning(
                "Dropping pre-commit terminal %s for item_id=%s",
                event_name, ev_item,
            )
            return "drop"
        return "pending"

    def _stash_pending_terminal(self, event_name: str, data: dict) -> None:
        """Hold a terminal that arrived before its own commit ACK (H1 residual).

        Stashing never binds or confirms the item (terminal cannot self-bind);
        the commit ACK does that and then replays the stashed terminal.
        """
        ev_item = data.get("item_id")
        if not ev_item or not self._has_active_commit():
            return
        if ev_item not in self._pending_terminals:
            self._pending_terminals[ev_item] = (event_name, data)
            while len(self._pending_terminals) > self._PENDING_TERMINALS_CAP:
                evicted_id, _ = self._pending_terminals.popitem(last=False)
                logger.warning(
                    "Evicted pending terminal for item_id=%s (cap=%d)",
                    evicted_id, self._PENDING_TERMINALS_CAP,
                )
        logger.info(
            "Stashed early %s for unconfirmed item_id=%s awaiting commit ACK (pending=%d)",
            event_name, ev_item, len(self._pending_terminals),
        )
        self._arm_ack_grace_timer()

    async def _consume_pending_terminal(self, item_id: str) -> None:
        """Replay a terminal that raced ahead of the now-confirmed item's ACK."""
        entry = self._pending_terminals.pop(item_id, None)
        if entry is None:
            return
        self._cancel_ack_grace_timer()
        event_name, data = entry
        logger.info("Replaying stashed %s for now-confirmed item_id=%s", event_name, item_id)
        if event_name == "transcription.completed":
            self._process_completed(data)
        elif event_name == "transcription.failed":
            self._process_failed(data)

    def _arm_ack_grace_timer(self) -> None:
        if not self._has_active_commit():
            return
        if self._config.transcription_ack_grace_sec <= 0:
            return
        if self._ack_grace_task is not None and not self._ack_grace_task.done():
            return
        self._ack_grace_task = asyncio.create_task(
            self._ack_grace_timeout(self._active_turn_id)
        )

    def _cancel_ack_grace_timer(self) -> None:
        task = self._ack_grace_task
        self._ack_grace_task = None
        if task is not None and not task.done():
            task.cancel()

    def _cancel_force_reconnect_task(self) -> None:
        task = getattr(self, "_force_reconnect_task", None)
        self._force_reconnect_task = None
        if task is not None and not task.done():
            task.cancel()

    def _cancel_finalize_retry_task(self) -> None:
        task = getattr(self, "_finalize_retry_task", None)
        self._finalize_retry_task = None
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()

    def _cancel_segment_rollover_task(self) -> None:
        task = getattr(self, "_segment_rollover_task", None)
        self._segment_rollover_task = None
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()

    def _cancel_segment_rollover_timeout_task(self) -> None:
        task = getattr(self, "_segment_rollover_timeout_task", None)
        self._segment_rollover_timeout_task = None
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()

    def _assert_transport_locked(self) -> None:
        if not self._transport_lock.owned_by_current_task():
            raise RuntimeError("transport mutation requires _transport_lock")

    def _current_commit_key(self) -> tuple[Optional[int], int, int]:
        return (
            self._active_turn_id,
            self._turn_generation,
            self._provider_generation,
        )

    def _has_active_commit(self) -> bool:
        return (
            self._commit_key == self._current_commit_key()
            and self._transport_state in {
                TransportState.COMMITTING,
                TransportState.COMMIT_ACKED,
            }
        )

    def _provider_event_is_current(
        self,
        provider_generation: Optional[int],
        provider_client: Optional[RealtimeClientBase],
        turn_generation: Optional[int],
    ) -> bool:
        if (
            provider_generation is None
            and provider_client is None
            and turn_generation is None
        ):
            return True  # direct internal/test invocation
        return (
            provider_generation == self._provider_generation
            and provider_client is self._client
            and turn_generation == self._turn_generation
        )

    def _begin_provider_epoch_locked(self) -> None:
        """Reset provider-scoped authority while preserving the turn transcript.

        A replacement provider creates a new committed item. Retiring the old
        binding lets the replacement ACK become authoritative, while the
        accumulated partial/completed text remains available for longer-result
        reconciliation at finalize.
        """

        self._assert_transport_locked()
        prior_resolved = self._select_completed_authority(
            self._provider_completed_text, self._transcript_text
        )
        self._best_partial_text = self._select_partial_candidate(
            self._provider_epoch_text,
            self._select_partial_candidate(prior_resolved, self._best_partial_text),
        )
        self._provider_epoch_text = ""
        self._transcript_text = self._best_partial_text
        self._provider_completed_text = ""
        self._normalized_final_text = ""
        self._transcript_completed = False
        if self._current_item_id:
            self._retire_item(self._current_item_id)
        self._current_item_id = None
        self._item_confirmed = False
        self._pending_terminals.clear()
        self._cancel_ack_grace_timer()

    def _append_turn_audio_locked(self, chunk: bytes) -> bool:
        """Append before send without evicting the replayable segment prefix."""

        self._assert_transport_locked()
        if not chunk:
            return True
        next_size = self._turn_audio_bytes + len(chunk)
        if next_size > self._config.max_turn_audio_bytes:
            logger.info(
                "Segment audio cache limit reached "
                "(current_bytes=%d chunk_bytes=%d limit_bytes=%d)",
                self._turn_audio_bytes, len(chunk), self._config.max_turn_audio_bytes,
            )
            return False
        self._turn_audio_chunks.append(chunk)
        self._turn_audio_bytes = next_size
        return True

    def _append_rollover_tail_locked(self, chunk: bytes) -> bool:
        """Retain audio received while the committed prefix awaits completion."""

        self._assert_transport_locked()
        if not chunk:
            return True
        next_size = self._rollover_tail_bytes + len(chunk)
        if next_size > self._config.max_turn_audio_bytes:
            logger.info(
                "Rollover tail reached segment limit; applying receive backpressure "
                "(tail_bytes=%d chunk_bytes=%d limit_bytes=%d)",
                self._rollover_tail_bytes, len(chunk),
                self._config.max_turn_audio_bytes,
            )
            return False
        self._rollover_tail_chunks.append(chunk)
        self._rollover_tail_bytes = next_size
        return True

    def _aggregate_segment_text(self, segment_text: str) -> str:
        """Build the full client view from finalized prefix plus active segment."""

        return append_transcript_delta(self._committed_transcript_text, segment_text)

    def _reset_active_segment_state_locked(self) -> None:
        """Retire one completed item while keeping the surrounding client turn."""

        self._assert_transport_locked()
        if self._current_item_id:
            self._retire_item(self._current_item_id)
        self._current_item_id = None
        self._item_confirmed = False
        self._pending_terminals.clear()
        self._cancel_ack_grace_timer()
        self._provider_epoch_text = ""
        self._best_partial_text = ""
        self._transcript_text = ""
        self._provider_completed_text = ""
        self._normalized_final_text = ""
        self._transcript_completed = False
        self._commit_key = None
        self._finalize_commit_attempts = 0

    def _promote_rollover_tail_locked(self) -> None:
        """Make the buffered post-boundary tail the next replayable segment."""

        self._assert_transport_locked()
        self._turn_audio_chunks = self._rollover_tail_chunks
        self._turn_audio_bytes = self._rollover_tail_bytes
        self._rollover_tail_chunks = []
        self._rollover_tail_bytes = 0
        self._provider_audio_bytes = 0

    def _turn_is_digital_silence_locked(self) -> bool:
        """True only for nonempty PCM whose every byte is exactly zero.

        This is deliberately narrower than voice activity detection: it blocks
        provider hallucinations on synthetic/digital silence without inventing
        an amplitude threshold that could discard quiet real speech.
        """

        self._assert_transport_locked()
        return self._turn_audio_bytes > 0 and not any(
            any(chunk) for chunk in self._turn_audio_chunks
        )

    async def _send_provider_audio_locked(self, chunk: bytes) -> bool:
        self._assert_transport_locked()
        if self._client is None:
            return False
        try:
            sent = bool(await self._client.send_audio(chunk))
        except Exception as exc:
            logger.warning("Provider audio send failed (%s)", type(exc).__name__)
            return False
        if sent:
            self._provider_audio_bytes += len(chunk)
        return sent

    def _mark_transport_sync_failed_locked(self) -> None:
        self._assert_transport_locked()
        self._openai_ready.clear()
        if self._transport_state not in {
            TransportState.STOP_REQUESTED,
            TransportState.COMMITTING,
            TransportState.COMMIT_ACKED,
            TransportState.FINALIZED,
        }:
            self._transport_state = TransportState.RECOVERING

    async def _replay_turn_audio_locked(self) -> bool:
        """Replay the complete immutable-under-lock turn cache from offset zero."""

        self._assert_transport_locked()
        if self._client is None:
            return False
        self._provider_audio_bytes = 0
        if not self._turn_audio_chunks:
            return True
        logger.info(
            "Turn cache replay start turn=%s chunks=%d bytes=%d offset=0 provider_generation=%d",
            self._active_turn_id, len(self._turn_audio_chunks), self._turn_audio_bytes,
            self._provider_generation,
        )
        turn_generation = self._turn_generation
        for chunk in self._turn_audio_chunks:
            sent = await self._send_provider_audio_locked(chunk)
            if self._finalized or turn_generation != self._turn_generation:
                return False
            if not sent:
                self._mark_transport_sync_failed_locked()
                return False
        synced = self._provider_audio_bytes == self._turn_audio_bytes
        logger.info(
            "Turn cache replay complete turn=%s sent_bytes=%d cached_bytes=%d provider_generation=%d",
            self._active_turn_id, self._provider_audio_bytes, self._turn_audio_bytes,
            self._provider_generation,
        )
        return synced

    async def _publish_transport_ready(self, *, publish_status: bool = True) -> None:
        self._transport_state = TransportState.RECORDING
        self._is_recording = True
        self._openai_ready.set()
        if not publish_status:
            return
        try:
            if self._websocket.client_state == WebSocketState.CONNECTED:
                await self._send_status("connected", self._active_turn_id)
        except Exception as exc:
            logger.warning(
                "Failed to send transport-ready status (%s)",
                type(exc).__name__,
            )

    async def _commit_transport_locked(self) -> bool:
        self._assert_transport_locked()
        key = self._current_commit_key()
        if self._commit_key == key:
            return True
        if self._client is None or not self._client._is_ws_open():
            return False
        if self._turn_audio_bytes < self._MIN_COMMIT_AUDIO_BYTES:
            logger.info(
                "Commit skipped: turn audio below local minimum (bytes=%d minimum=%d)",
                self._turn_audio_bytes, self._MIN_COMMIT_AUDIO_BYTES,
            )
            return False
        if self._turn_is_digital_silence_locked():
            logger.info(
                "Commit skipped: turn contains digital silence only (bytes=%d)",
                self._turn_audio_bytes,
            )
            return False
        if self._provider_audio_bytes != self._turn_audio_bytes:
            logger.warning(
                "Commit blocked: provider session is not fully synchronized "
                "(sent_bytes=%d cached_bytes=%d)",
                self._provider_audio_bytes, self._turn_audio_bytes,
            )
            return False
        if self._finalize_commit_attempts >= self._FINALIZE_MAX_ATTEMPTS:
            logger.warning(
                "Commit blocked: finalize attempt budget exhausted (%d/%d)",
                self._finalize_commit_attempts, self._FINALIZE_MAX_ATTEMPTS,
            )
            return False
        self._commit_key = key
        self._transport_state = TransportState.COMMITTING
        # Reserve the attempt before the await: provider ACK/terminal handlers
        # may run while commit_audio() is suspended and must see a stable count.
        self._finalize_commit_attempts += 1
        finalize_attempt = self._finalize_commit_attempts
        try:
            await self._client.commit_audio()
            logger.info(
                "Audio committed turn=%s provider_generation=%d finalize_attempt=%d/%d; "
                "awaiting transcription completion",
                self._active_turn_id, self._provider_generation,
                finalize_attempt, self._FINALIZE_MAX_ATTEMPTS,
            )
            return True
        except Exception as exc:
            # Keep the complete cache and make a replacement provider eligible
            # for another commit. Finalization owns the bounded recovery loop.
            logger.warning(
                "Provider commit failed; preserving turn for retry "
                "(attempt=%d/%d type=%s)",
                finalize_attempt, self._FINALIZE_MAX_ATTEMPTS, type(exc).__name__,
            )
            if self._commit_key == key:
                self._commit_key = None
            self._openai_ready.clear()
            self._transport_state = TransportState.STOP_REQUESTED
        return False

    async def _begin_segment_rollover_locked(self) -> bool:
        """Commit the bounded prefix without terminating the surrounding turn."""

        self._assert_transport_locked()
        if self._segment_rollover_pending:
            return True
        self._segment_rollover_done.clear()
        if self._turn_is_digital_silence_locked():
            # An all-zero prefix has no transcript to wait for. Retire it as an
            # empty segment and rebuild the crossing tail; ending the whole turn
            # here would discard speech that began exactly at the boundary.
            silent_bytes = self._turn_audio_bytes
            self._reset_active_segment_state_locked()
            self._promote_rollover_tail_locked()
            self._openai_ready.clear()
            if self._stop_requested:
                self._transport_state = TransportState.STOP_REQUESTED
                self._is_recording = False
            else:
                self._transport_state = TransportState.RECOVERING
                self._is_recording = True
            logger.info(
                "Retired digital-silence segment without ending turn "
                "(turn=%s silent_bytes=%d next_bytes=%d)",
                self._active_turn_id, silent_bytes, self._turn_audio_bytes,
            )
            recovered = await self._recover_transport_locked(
                self._turn_generation,
                planned_rollover=True,
            )
            if not recovered and not self._finalized:
                await self._fail_segment_rollover_locked(
                    "silent_segment_rollover_reconnect_exhausted"
                )
            self._segment_rollover_done.set()
            return recovered or self._finalized
        self._segment_rollover_pending = True
        committed = await self._commit_transport_locked()
        if not committed and not self._finalized:
            # Reuse the proven replay/recommit path. STOP_REQUESTED here describes
            # the provider buffer, not user stop; `_stop_requested` keeps those
            # meanings separate and audio continues into the rollover tail.
            self._transport_state = TransportState.STOP_REQUESTED
            self._openai_ready.clear()
            committed = await self._recover_transport_locked(self._turn_generation)
        if committed and not self._finalized:
            self._arm_segment_rollover_timeout()
            logger.info(
                "Committed bounded audio segment; retaining post-boundary tail "
                "(turn=%s prefix_bytes=%d tail_bytes=%d)",
                self._active_turn_id, self._turn_audio_bytes,
                self._rollover_tail_bytes,
            )
            return True
        if not self._finalized:
            await self._fail_segment_rollover_locked(
                "segment_rollover_commit_failed"
            )
        return False

    def _arm_segment_rollover_timeout(self) -> None:
        task = self._segment_rollover_timeout_task
        if task is not None and not task.done():
            return
        self._segment_rollover_timeout_task = asyncio.create_task(
            self._segment_rollover_timeout(self._active_turn_id)
        )

    def _segment_rollover_timeout_budget_sec(self) -> float:
        """Upper bound shared by the watchdog and receive backpressure waiter."""

        reconnect_backoff_budget = self._RECONNECT_BACKOFF_BASE_SEC * (
            (2 ** (self._RECONNECT_MAX_ATTEMPTS - 1)) - 1
        )
        return max(
            0.001,
            self._config.transcription_finalize_timeout_sec
            * self._FINALIZE_MAX_ATTEMPTS
            + reconnect_backoff_budget,
        )

    async def _segment_rollover_timeout(self, turn_token) -> None:
        """Bound a segment terminal wait and reuse the existing one-retry policy."""

        try:
            while True:
                waited_attempt = self._finalize_commit_attempts
                await asyncio.sleep(self._config.transcription_finalize_timeout_sec)
                async with self._transport_lock:
                    if (
                        self._finalized
                        or self._finalizing
                        or self._is_stale_turn(turn_token)
                        or not self._segment_rollover_pending
                        or self._transcript_completed
                    ):
                        return
                    if self._finalize_commit_attempts != waited_attempt:
                        # A failure/disconnect owner already advanced the provider
                        # attempt while this timer slept. Give that fresh attempt
                        # its own full terminal window.
                        continue
                    logger.error(
                        "Segment transcription timed out after %.1fs "
                        "(attempt=%d/%d tail_bytes=%d)",
                        self._config.transcription_finalize_timeout_sec,
                        self._finalize_commit_attempts,
                        self._FINALIZE_MAX_ATTEMPTS,
                        self._rollover_tail_bytes,
                    )
                    if await self._retry_finalize_result_locked(
                        "segment_rollover_timeout"
                    ):
                        continue
                    await self._fail_segment_rollover_locked(
                        "segment_rollover_timeout"
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Segment rollover watchdog failed (%s)",
                type(exc).__name__,
                exc_info=True,
            )
            if not self._finalized and not self._is_stale_turn(turn_token):
                try:
                    await self._fail_segment_rollover(
                        "segment_rollover_watchdog_error"
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as finalize_exc:
                    logger.error(
                        "Segment rollover watchdog failure barrier failed (%s)",
                        type(finalize_exc).__name__,
                        exc_info=True,
                    )
        finally:
            # Every watchdog exit releases receive backpressure. Normal completion
            # has a coordinator ready to advance the state; abnormal completion
            # has already attempted the shared failure/finalize barrier above.
            self._segment_rollover_done.set()
            if self._segment_rollover_timeout_task is asyncio.current_task():
                self._segment_rollover_timeout_task = None

    async def _force_reconnect_after_delay(self, turn_token) -> None:
        try:
            await asyncio.sleep(self._config.force_reconnect_after_ms / 1000)
            async with self._transport_lock:
                if (
                    self._is_stale_turn(turn_token)
                    or not self._is_recording
                    or self._client is None
                    or not self._client._is_ws_open()
                ):
                    return
                ws = getattr(self._client, "ws", None)
                if ws is None:
                    logger.warning(
                        "TEST hook: provider ws unavailable; cannot force mid-turn disconnect "
                        "(turn=%s)", turn_token,
                    )
                    return
                logger.warning(
                    "TEST hook: forcing mid-turn provider disconnect (turn=%s)", turn_token
                )
                # The forced close itself begins a recovery window. Publish that
                # state before the potentially backpressured close so a real
                # client stop arriving concurrently can latch STOP_REQUESTED
                # without waiting for a later failed audio send to notice the
                # disconnect. The provider callback / stop owner still performs
                # the actual rebuild and offset-zero replay under this same lock.
                self._openai_ready.clear()
                self._transport_state = TransportState.RECOVERING
                await self._send_recovery_event("recovery_started")
                await ws.close()
        except asyncio.CancelledError:
            raise

    async def _ack_grace_timeout(self, turn_token) -> None:
        """If a stashed early terminal is never confirmed by a commit ACK within
        the grace window, finalize with a failure reason and rotate — never hang
        until the 120s stop timeout (H1 residual)."""
        try:
            await asyncio.sleep(self._config.transcription_ack_grace_sec)
        except asyncio.CancelledError:
            raise
        # Detach so the finalize we are about to trigger does not cancel this
        # still-running task.
        if self._ack_grace_task is asyncio.current_task():
            self._ack_grace_task = None
        async with self._transport_lock:
            if self._finalized or self._finalizing or self._is_stale_turn(turn_token):
                return
            if not self._has_active_commit():
                self._pending_terminals.clear()
                return
            if self._item_confirmed or not self._pending_terminals:
                # Recheck under the transport owner: an ACK may have confirmed
                # while this timer waited to acquire the lock.
                return
            logger.error(
                "Commit ACK did not arrive within %.1fs of an early terminal; "
                "handling ack_grace_timeout before the %.0fs stop timeout",
                self._config.transcription_ack_grace_sec,
                self._config.transcription_finalize_timeout_sec,
            )
            self._pending_terminals.clear()
            self._consecutive_transcription_failures = max(
                self._consecutive_transcription_failures,
                self._config.transcription_failure_rotate_threshold,
            )
            if await self._retry_finalize_result_locked("ack_grace_timeout"):
                return
            if self._segment_rollover_pending:
                await self._fail_segment_rollover_locked("ack_grace_timeout")
            else:
                await self._finalize_turn_locked("ack_grace_timeout")

    async def _reset_client_display(self):
        """Reset the active item view without erasing finalized segment text."""
        async with self._output_lock:
            if (
                self._finalized
                or self._finalizing
                or self._websocket.client_state != WebSocketState.CONNECTED
            ):
                return
            payload = {
                "type": "text",
                "content": self._committed_transcript_text,
                "isNewResponse": True,
            }
            if self._active_turn_id is not None:
                payload["turn_id"] = self._active_turn_id
            await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))

    def _discard_tentative_stream(self):
        """Drop content accumulated under a superseded tentative binding."""
        self._provider_epoch_text = ""
        self._transcript_text = self._best_partial_text
        self._provider_completed_text = ""
        self._normalized_final_text = ""
        self._transcript_completed = False
        self._emitted_text = self._committed_transcript_text

    # ── provider event handlers ──────────────────────────────────────────────
    def _register_client_handlers(self, client: RealtimeClientBase):
        provider_generation = self._provider_generation

        async def _for_generation(name, handler, data):
            if (
                provider_generation != self._provider_generation
                or client is not self._client
            ):
                logger.info(
                    "Ignoring stale provider event type=%s generation=%d current=%d",
                    name, provider_generation, self._provider_generation,
                )
                return
            await handler(data)

        def guarded(name, handler, *, pass_epoch=False):
            async def _handler(data):
                if pass_epoch:
                    turn_generation = self._turn_generation

                    async def _with_epoch(payload):
                        await handler(
                            payload,
                            provider_generation=provider_generation,
                            provider_client=client,
                            turn_generation=turn_generation,
                        )

                    await _for_generation(name, _with_epoch, data)
                else:
                    await _for_generation(name, handler, data)

            return _handler

        def generic(name):
            return guarded(name, lambda data: self._on_generic_event(name, data))

        client.register_handler("session.updated", generic("session.updated"))
        client.register_handler("input_audio_buffer.speech_started", generic("input_audio_buffer.speech_started"))
        client.register_handler("input_audio_buffer.speech_stopped", generic("input_audio_buffer.speech_stopped"))
        client.register_handler("input_audio_buffer.cleared", generic("input_audio_buffer.cleared"))
        client.register_handler("conversation.item.created", generic("conversation.item.created"))
        client.register_handler("conversation.item.added", generic("conversation.item.added"))
        client.register_handler("conversation.item.done", generic("conversation.item.done"))
        client.register_handler("rate_limits.updated", generic("rate_limits.updated"))
        client.register_handler("ping", generic("ping"))
        client.register_handler(
            "input_audio_buffer.committed",
            guarded("input_audio_buffer.committed", self._on_input_buffer_committed),
        )
        client.register_handler(
            "conversation.item.input_audio_transcription.delta",
            guarded(
                "conversation.item.input_audio_transcription.delta",
                self._on_transcription_delta,
            ),
        )
        client.register_handler(
            "conversation.item.input_audio_transcription.completed",
            guarded(
                "conversation.item.input_audio_transcription.completed",
                self._on_transcription_completed,
            ),
        )
        client.register_handler(
            "conversation.item.input_audio_transcription.segment",
            generic("conversation.item.input_audio_transcription.segment"),
        )
        client.register_handler(
            "conversation.item.input_audio_transcription.failed",
            guarded(
                "conversation.item.input_audio_transcription.failed",
                self._on_transcription_failed,
            ),
        )
        client.register_handler(
            "error", guarded("error", self._on_error, pass_epoch=True)
        )

        async def _disconnect_for_generation():
            await self._on_provider_disconnect(provider_generation)

        client.set_on_disconnect(_disconnect_for_generation)

    async def _on_input_buffer_committed(self, data):
        # The commit ACK is the SOLE authority that CONFIRMS the item for this
        # turn. Fail-closed on a malformed (missing item_id) or retired ACK, and
        # never let a late ACK retire an already-confirmed current item — a
        # mismatched ACK may only supersede a still-*tentative* binding (N1).
        if self._finalized or self._finalizing:
            return
        ack_item = data.get("item_id")
        # Official schema: input_audio_buffer.committed carries item_id. A
        # missing one has no authority — fail-closed rather than promoting an
        # arbitrary tentative binding to confirmed.
        if not ack_item:
            logger.warning("Fail-closed: commit ACK missing item_id; item remains unconfirmed")
            return
        # A retired item id (past turn, or a superseded tentative binding) can
        # never be re-confirmed; dropping it protects an already-confirmed
        # current item from a late/stale ACK.
        if ack_item in self._retired_item_ids:
            logger.warning("Fail-closed: commit ACK for retired item_id=%s; ignoring", ack_item)
            return
        superseded: Optional[str] = None
        if self._current_item_id is not None and self._current_item_id != ack_item:
            if self._item_confirmed:
                logger.warning(
                    "Dropping commit ACK item_id=%s: current item_id=%s already confirmed",
                    ack_item, self._current_item_id,
                )
                return
            superseded = self._current_item_id
        # Commit the internal real-id transition ATOMICALLY (retire the superseded
        # tentative binding, discard its stream, rebind + confirm) BEFORE the
        # best-effort display clear, so a clear failure can never leave an "active
        # but retired / unconfirmed" identity (N1).
        if superseded is not None:
            self._retire_item(superseded)
            self._discard_tentative_stream()
        self._current_item_id = ack_item
        self._item_confirmed = True
        if self._commit_key == self._current_commit_key():
            self._transport_state = TransportState.COMMIT_ACKED
        if superseded is not None:
            logger.warning(
                "Commit ACK item_id=%s overrides tentative binding=%s; discarded tentative content",
                ack_item, superseded,
            )
            try:
                await self._reset_client_display()
            except Exception as e:
                logger.error(
                    "Best-effort display clear failed during rebind to item_id=%s: %s",
                    ack_item, e, exc_info=True,
                )
        else:
            logger.info("Confirmed item_id=%s (input_audio_buffer.committed)", ack_item)
        # A terminal that raced ahead of this ACK was stashed pending; replay it
        # now that the item is confirmed (H1 residual — avoids the 120s stop
        # timeout for the ACK-after-completed ordering).
        await self._consume_pending_terminal(ack_item)
        await self._on_generic_event("input_audio_buffer.committed", data)

    async def _on_transcription_delta(self, data):
        # Once finalize has begun, no further provider text may reach the client
        # (S1 / H2): drop late deltas rather than emitting text after idle.
        if self._finalized or self._finalizing:
            return
        try:
            async with self._output_lock:
                if self._finalized or self._finalizing:
                    return
                if not self._match_item(data, "transcription.delta"):
                    return
                delta = self._extract_delta(data)
                if not delta:
                    return
                self._provider_epoch_text = append_transcript_delta(
                    self._provider_epoch_text, delta
                )
                self._transcript_text = self._select_partial_candidate(
                    self._provider_epoch_text, self._best_partial_text
                )
                if self._websocket.client_state == WebSocketState.CONNECTED:
                    await self._emit_text_delta(delta)
        except Exception as e:
            logger.error("Error in transcription delta handler: %s", e, exc_info=True)

    async def _on_transcription_completed(self, data):
        # Terminal authority: only a CONFIRMED item may finalize (H1). A
        # foreign/stale/missing-id completed is dropped; a completed that raced
        # ahead of its own commit ACK is stashed pending (H1 residual) rather
        # than dropped, so the ACK can replay it instead of hanging to timeout.
        if self._finalized or self._finalizing:
            return
        decision = self._classify_terminal(data, "transcription.completed")
        if decision == "drop":
            return
        if decision == "pending":
            self._stash_pending_terminal("transcription.completed", data)
            return
        self._process_completed(data)

    def _process_completed(self, data: dict) -> None:
        """Apply a confirmed completed transcript and schedule finalize."""
        try:
            completed = self._extract_completed(data)
            self._consecutive_transcription_failures = 0
            # Keep provider completed, streamed partial, and later W6 policy output
            # separate; finalize defensively chooses the longer nonblank raw view.
            self._provider_completed_text = completed
            raw_final = self._select_completed_authority(
                completed, self._transcript_text
            )
            if not completed.strip() and self._transcript_text.strip():
                logger.info(
                    "Blank completed; retaining nonblank streamed partial (len=%d)",
                    len(self._transcript_text),
                )
            elif not completed.strip():
                logger.info("Blank completed with no partial; finalizing empty transcript")
            if not raw_final.strip() and self._can_retry_finalize_result():
                logger.warning(
                    "Empty finalize result; preserving turn cache for one retry"
                )
                self._transcript_completed = False
                self._schedule_finalize_retry("empty_transcript")
                return
            self._transcript_completed = True
            logger.info("Transcription completed (raw_len=%d)", len(raw_final))
        except Exception as e:
            logger.error("Error in transcription completed handler: %s", e, exc_info=True)
        # A memory-boundary commit finalizes only this segment. The surrounding
        # client turn remains live; normal stop commits still use the S1 barrier.
        if self._segment_rollover_pending:
            if self._transcript_completed:
                self._cancel_segment_rollover_timeout_task()
                self._schedule_segment_rollover()
            # If processing itself failed, leave the segment watchdog armed; it
            # owns bounded retry/finalization instead of silently wedging here.
            return
        # Finalize only for our confirmed item (reached only past the gate).
        self._schedule_finalize("transcription.completed")

    def _schedule_segment_rollover(self) -> None:
        if self._finalized or self._finalizing or not self._segment_rollover_pending:
            return
        existing = self._segment_rollover_task
        if existing is not None and not existing.done():
            return
        self._segment_rollover_task = asyncio.create_task(
            self._finish_segment_rollover(self._active_turn_id)
        )

    async def _finish_segment_rollover(self, turn_token) -> None:
        """Land one segment, promote its tail, then rebuild/replay without idle."""

        try:
            async with self._transport_lock:
                if (
                    self._finalized
                    or self._finalizing
                    or self._is_stale_turn(turn_token)
                    or not self._segment_rollover_pending
                    or not self._transcript_completed
                ):
                    return
                self._cancel_segment_rollover_timeout_task()

                # `isNewResponse` replaces the client's whole turn, so an
                # intermediate final must contain the prior finalized prefix too.
                async with self._output_lock:
                    await self._apply_completed_replacement(
                        "segment_rollover", turn_token
                    )
                    self._committed_transcript_text = self._normalized_final_text

                completed_item = self._current_item_id
                completed_bytes = self._turn_audio_bytes
                self._reset_active_segment_state_locked()
                self._promote_rollover_tail_locked()
                self._segment_rollover_pending = False
                self._openai_ready.clear()
                if self._stop_requested:
                    self._transport_state = TransportState.STOP_REQUESTED
                    self._is_recording = False
                else:
                    self._transport_state = TransportState.RECOVERING
                    self._is_recording = True

                logger.info(
                    "Segment finalized without ending turn "
                    "(turn=%s item_id=%s completed_bytes=%d next_bytes=%d "
                    "committed_text_len=%d stop_requested=%s)",
                    self._active_turn_id, completed_item, completed_bytes,
                    self._turn_audio_bytes, len(self._committed_transcript_text),
                    self._stop_requested,
                )
                recovered = await self._recover_transport_locked(
                    self._turn_generation,
                    planned_rollover=True,
                )
                if not recovered and not self._finalized:
                    await self._fail_segment_rollover_locked(
                        "segment_rollover_reconnect_exhausted"
                    )
                self._segment_rollover_done.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Segment rollover failed (%s)", type(exc).__name__, exc_info=True
            )
            if not self._finalized and not self._is_stale_turn(turn_token):
                await self._fail_segment_rollover("segment_rollover_error")
        finally:
            if self._segment_rollover_task is asyncio.current_task():
                self._segment_rollover_task = None

    async def _on_transcription_failed(self, data):
        if self._finalized or self._finalizing:
            return
        decision = self._classify_terminal(data, "transcription.failed")
        if decision == "drop":
            return
        if decision == "pending":
            self._stash_pending_terminal("transcription.failed", data)
            return
        self._process_failed(data)

    def _process_failed(self, data: dict) -> None:
        error_info = data.get("error", {}) if isinstance(data, dict) else {}
        self._consecutive_transcription_failures += 1
        logger.error(
            "Input audio transcription FAILED (type=%s code=%s consecutive=%d/%d)",
            error_info.get("type", "unknown"),
            error_info.get("code", "unknown"),
            self._consecutive_transcription_failures,
            self._config.transcription_failure_rotate_threshold,
        )
        if self._can_retry_finalize_result():
            self._schedule_finalize_retry("transcription.failed")
        else:
            self._schedule_finalize("transcription.failed")

    async def _on_error(
        self,
        data,
        *,
        provider_generation: Optional[int] = None,
        provider_client: Optional[RealtimeClientBase] = None,
        turn_generation: Optional[int] = None,
    ):
        error_info = data.get("error", {}) if isinstance(data, dict) else {}
        error_msg = error_info.get("message", "Unknown error")
        in_finalize = self._finalizing or self._transport_state in {
            TransportState.STOP_REQUESTED,
            TransportState.COMMITTING,
            TransportState.COMMIT_ACKED,
        }
        if self._is_recoverable_buffer_too_small(data) and not in_finalize:
            logger.info("Ignoring transient provider buffer-too-small before finalize")
            return
        logger.error(
            "Provider error (type=%s code=%s message_len=%d)",
            error_info.get("type", "unknown"),
            error_info.get("code", "unknown"),
            len(str(error_msg)),
        )
        payload = {"type": "error", "content": error_msg}
        if self._active_turn_id is not None:
            payload["turn_id"] = self._active_turn_id
        async with self._transport_lock:
            if self._finalized or self._finalizing:
                logger.info("Ignoring provider error after finalize barrier started")
                return
            if not self._provider_event_is_current(
                provider_generation, provider_client, turn_generation
            ):
                logger.info(
                    "Ignoring stale provider error generation=%s current=%s",
                    provider_generation, self._provider_generation,
                )
                return
            if self._transcript_completed:
                owner = (
                    "segment rollover coordinator"
                    if self._segment_rollover_pending
                    else "finalize barrier"
                )
                logger.info(
                    "Ignoring provider error after completed authority; deferring to %s",
                    owner,
                )
                return
            if in_finalize and self._can_retry_finalize_result():
                if await self._retry_finalize_result_locked("provider_error"):
                    return
            try:
                await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
            except Exception:
                pass
            await self._finalize_turn_locked("error")

    async def _on_generic_event(self, event_type, data):
        if VERBOSE_SERVER_LOG:
            keys = sorted(data.keys()) if isinstance(data, dict) else []
            body_len = len(json.dumps(data, ensure_ascii=False)) if data else 0
            logger.info("Handled %s (keys=%s body_len=%d)", event_type, keys, body_len)
        else:
            logger.debug("Handled %s", event_type)

    # ── finalize (single barrier, replacement strictly before idle: S1) ───────
    def _can_retry_finalize_result(self) -> bool:
        return 0 < self._finalize_commit_attempts < self._FINALIZE_MAX_ATTEMPTS

    def _schedule_finalize_retry(self, reason: str) -> None:
        if self._finalized or self._finalizing:
            return
        existing = self._finalize_retry_task
        if existing is not None and not existing.done():
            return
        self._finalize_retry_task = asyncio.create_task(
            self._retry_finalize_after_failure(
                reason, self._active_turn_id, self._finalize_commit_attempts
            )
        )

    async def _retry_finalize_after_failure(
        self, reason: str, turn_token, source_attempt: int
    ) -> None:
        try:
            if self._finalized or self._is_stale_turn(turn_token):
                return
            if await self._retry_finalize_result(
                reason, expected_attempt=source_attempt
            ):
                return
            if not self._finalized and not self._is_stale_turn(turn_token):
                if self._segment_rollover_pending:
                    await self._fail_segment_rollover(reason)
                else:
                    await self._finalize_turn(reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Finalize result retry failed (%s): %s",
                reason, type(exc).__name__, exc_info=True,
            )
            if not self._finalized and not self._is_stale_turn(turn_token):
                if self._segment_rollover_pending:
                    await self._fail_segment_rollover("finalize_retry_error")
                else:
                    await self._finalize_turn("finalize_retry_error")

    async def _retry_finalize_result(
        self, reason: str, *, expected_attempt: Optional[int] = None
    ) -> bool:
        async with self._transport_lock:
            if (
                expected_attempt is not None
                and self._finalize_commit_attempts != expected_attempt
            ):
                advanced = self._finalize_commit_attempts > expected_attempt
                logger.info(
                    "Skipping stale finalize retry after %s "
                    "(source_attempt=%d current_attempt=%d advanced=%s)",
                    reason, expected_attempt, self._finalize_commit_attempts, advanced,
                )
                return advanced
            return await self._retry_finalize_result_locked(reason)

    async def _retry_finalize_result_locked(self, reason: str) -> bool:
        self._assert_transport_locked()
        if self._finalized or not self._can_retry_finalize_result():
            return False
        logger.warning(
            "Retrying finalize result after %s (completed_attempts=%d max=%d)",
            reason, self._finalize_commit_attempts, self._FINALIZE_MAX_ATTEMPTS,
        )
        self._transport_state = TransportState.STOP_REQUESTED
        if not self._segment_rollover_pending:
            self._is_recording = False
        self._openai_ready.clear()
        return await self._recover_transport_locked(self._turn_generation)

    def _schedule_finalize(self, reason: str):
        if self._finalized or self._finalizing:
            return
        existing = self._finalize_task
        if existing is not None and not existing.done():
            return
        self._finalize_task = asyncio.create_task(
            self._finalize_after_barrier(reason, self._active_turn_id)
        )

    async def _finalize_after_barrier(self, reason: str, turn_token):
        try:
            if self._finalized or self._finalizing or self._is_stale_turn(turn_token):
                return
            if self._segment_rollover_pending:
                await self._fail_segment_rollover(reason)
                return
            # From here the turn is committed to finalizing: handlers drop any
            # further provider text so nothing lands after idle (S1 / H2).
            self._finalizing = True
            await self._finalize_turn(reason)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Error in finalize barrier (%s): %s", reason, e, exc_info=True)
            # A replacement/internal failure must still TERMINATE the turn: fall
            # through to the finalize barrier with a failure reason so done/idle/
            # provider-rotate best-effort cleanup runs, rather than wedging at
            # _finalizing=True/_finalized=False until the 120s stop timeout (N2).
            if not self._finalized and not self._is_stale_turn(turn_token):
                try:
                    if self._segment_rollover_pending:
                        await self._fail_segment_rollover(
                            "finalize_barrier_error"
                        )
                    else:
                        await self._finalize_turn("finalize_barrier_error")
                except Exception as e2:
                    logger.error(
                        "Error finalizing turn after barrier failure (%s): %s",
                        reason, e2, exc_info=True,
                    )

    async def _apply_completed_replacement(self, reason: str, turn_token):
        """If the resolved partial/completed transcript diverges from what we
        streamed, replace the whole on-screen text via isNewResponse=True. This
        runs strictly before status:idle so the client never sees text land after
        idle (S1)."""
        if not self._transcript_completed:
            return
        raw_final = self._select_completed_authority(
            self._provider_completed_text, self._transcript_text
        )
        policy_result = normalize_final_transcript(raw_final)
        final = self._aggregate_segment_text(policy_result.text)
        self._normalized_final_text = final
        logger.info(
            "Final transcript policy applied (rule_ids=%s count=%d)",
            ",".join(policy_result.applied_rule_ids) or "none",
            policy_result.applied_count,
        )
        current_view = self._current_emitted_view()
        if final == current_view:
            return  # deltas already reconstructed the completed transcript
        if self._finalized or self._is_stale_turn(turn_token):
            return
        payload = {"type": "text", "content": final, "isNewResponse": True}
        if self._active_turn_id is not None:
            payload["turn_id"] = self._active_turn_id
        await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
        self._emitted_text = final
        logger.info(
            "Applied completed-transcript replacement before finalize "
            "(final_len=%d streamed_len=%d)", len(final), len(current_view),
        )

    async def _finalize_turn(self, reason: str):
        async with self._transport_lock:
            await self._finalize_turn_locked(reason)

    async def _reconcile_partial_epoch_before_finalize_locked(self, reason: str) -> None:
        """Collapse cross-provider streamed duplication on non-completed exits."""

        self._assert_transport_locked()
        if self._transcript_completed:
            return  # completed authority is reconciled by the finalize barrier
        candidate = self._aggregate_segment_text(self._transcript_text)
        current_view = self._current_emitted_view()
        if candidate == current_view:
            return
        if self._websocket.client_state != WebSocketState.CONNECTED:
            return
        payload = {"type": "text", "content": candidate, "isNewResponse": True}
        if self._active_turn_id is not None:
            payload["turn_id"] = self._active_turn_id
        try:
            await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            logger.error(
                "Failed partial-epoch reconciliation before %s (%s)",
                reason, type(exc).__name__,
            )
            return
        self._emitted_text = candidate
        logger.info(
            "Reconciled provider-epoch partial before finalize "
            "(reason=%s candidate_len=%d streamed_len=%d)",
            reason, len(candidate), len(current_view),
        )

    async def _seal_client_output_locked(self, reason: str) -> str:
        """Reconcile text, latch final, and emit idle as one output barrier.

        The transport owner latches ``_finalizing`` before entering this helper.
        Holding ``_output_lock`` continuously then drains any delta whose client
        write was already in flight and prevents a queued delta from observing an
        intermediate replacement-before-idle state. Once released, every queued
        delta rechecks ``_finalizing`` / ``_finalized`` and drops (S1).
        """

        self._assert_transport_locked()
        async with self._output_lock:
            if self._transcript_completed:
                try:
                    await self._apply_completed_replacement(
                        reason, self._active_turn_id
                    )
                except Exception as exc:
                    logger.error(
                        "Failed completed reconciliation before %s (%s)",
                        reason, type(exc).__name__, exc_info=True,
                    )
                    reason = "finalize_barrier_error"
            else:
                await self._reconcile_partial_epoch_before_finalize_locked(reason)

            # Final text replacement and idle are one indivisible client-output
            # transition. No provider delta may be emitted between or after them.
            self._finalized = True
            self._transport_state = TransportState.FINALIZED
            try:
                payload = {"type": "status", "status": "idle"}
                if self._active_turn_id is not None:
                    payload["turn_id"] = self._active_turn_id
                await self._websocket.send_text(
                    json.dumps(payload, ensure_ascii=False)
                )
            except Exception as exc:
                logger.error(
                    "Error sending status after %s: %s",
                    reason, exc, exc_info=True,
                )
        return reason

    async def _finalize_turn_locked(self, reason: str):
        self._assert_transport_locked()
        done = self._turn_done
        if self._finalized:
            return
        # Latch before any awaited replacement/reconciliation so provider text
        # handlers cannot pass their entry guard and land after idle (S1).
        self._finalizing = True
        reason = await self._seal_client_output_locked(reason)
        # Single finalize barrier for every exit path (completed / failed /
        # timeout / commit error / provider error / disconnect): the output
        # barrier above sends idle exactly once and later provider text drops.
        self._is_recording = False
        self._openai_ready.clear()
        self._turn_audio_chunks.clear()
        self._turn_audio_bytes = 0
        self._provider_audio_bytes = 0
        self._rollover_tail_chunks.clear()
        self._rollover_tail_bytes = 0
        self._segment_rollover_pending = False
        self._segment_rollover_done.set()
        self._cancel_force_reconnect_task()
        self._cancel_finalize_retry_task()
        self._cancel_segment_rollover_task()
        self._cancel_segment_rollover_timeout_task()
        # The turn is terminating: any ACK-grace timer / stashed early terminal is
        # now moot (its own path either reached here or is superseded).
        self._cancel_ack_grace_timer()
        self._pending_terminals.clear()
        if done:
            done.set()
        try:
            self._audio_cache.enqueue_turn(
                turn_id=self._active_turn_id,
                outcome=reason,
                sample_rate=self._audio_processor.target_sample_rate,
            )
        except Exception as e:
            logger.error("Error enqueueing audio cache on finalize (%s): %s", reason, e, exc_info=True)
        emitted_len = len(self._emitted_text) if self._emitted_text else 0
        if (
            emitted_len == 0
            and reason in self._SUCCESSFUL_FINALIZE_REASONS
            and reason != "digital_silence"
        ):
            logger.warning("Empty transcription on finalize (%s)", reason)

        if not self._client:
            self._reset_session_bookkeeping()
            return

        can_keep_session = (
            self._config.keep_provider_session
            and reason in self._SUCCESSFUL_FINALIZE_REASONS
            and self._consecutive_transcription_failures < self._config.transcription_failure_rotate_threshold
        )
        if can_keep_session:
            logger.info("Finalizing turn (%s), keeping transcription session alive", reason)
            try:
                await self._client.clear_audio_buffer()
            except Exception as e:
                logger.warning("Failed to clear provider audio buffer on finalize (%s): %s", reason, e)
            return

        logger.info("Finalizing turn (%s), closing transcription session", reason)
        try:
            await self._client.close()
        except Exception as e:
            logger.error("Error closing client after %s: %s", reason, e, exc_info=True)
        self._client = None
        self._reset_session_bookkeeping()

    def _reset_session_bookkeeping(self):
        self._active_model = None
        self._provider_session_turns = 0
        self._provider_session_started_at = None
        self._openai_ready.clear()

    # ── provider session lifecycle ───────────────────────────────────────────
    async def _create_client(self) -> RealtimeClientBase:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        logger.info("Creating transcription client (model: %s)", WHISPER_TRANSCRIBE_MODEL)
        return WhisperRealtimeTranscriptionClient(OPENAI_API_KEY)

    async def _init_or_reuse_client(
        self,
        turn_id: Optional[int] = None,
        *,
        publish_ready: bool = True,
        force_rebuild: bool = False,
        emit_failure_error: bool = True,
        max_attempts_override: Optional[int] = None,
    ) -> bool:
        self._assert_transport_locked()
        requested_model = WHISPER_TRANSCRIBE_MODEL
        max_attempts = max_attempts_override or self._config.provider_init_max_attempts
        retry_delay_sec = self._config.provider_init_retry_delay_sec

        can_reuse = (
            not force_rebuild
            and
            self._config.keep_provider_session
            and self._client
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
        failure_limit_reached = (
            self._consecutive_transcription_failures
            >= self._config.transcription_failure_rotate_threshold
        )

        if can_reuse and turn_limit_reached:
            logger.info(
                "Transcription session reached max turns (%d/%d), rotating",
                self._provider_session_turns, self._config.provider_session_max_turns,
            )
        if can_reuse and age_limit_reached:
            logger.info(
                "Transcription session reached max age (%.0fs/%.0f), rotating (60min cap guard)",
                session_age_sec or 0.0, float(self._config.provider_session_max_age_sec),
            )
        if can_reuse and failure_limit_reached:
            logger.warning(
                "Transcription failed %d consecutive times (threshold=%d), rotating",
                self._consecutive_transcription_failures,
                self._config.transcription_failure_rotate_threshold,
            )
            self._consecutive_transcription_failures = 0

        if can_reuse and not turn_limit_reached and not age_limit_reached and not failure_limit_reached:
            # The transcription bias prompt is provider-session state. A later
            # session.update that omits prompt clears it, while resending it per
            # turn violates the per-session contract. Reuse the already-open,
            # already-configured provider session as-is.
            logger.info("Reusing existing transcription session configuration")
            if publish_ready:
                self._openai_ready.set()
                await self._send_status("connected", turn_id)
            return True

        self._openai_ready.clear()

        for attempt in range(1, max_attempts + 1):
            try:
                if self._client:
                    logger.warning("Previous client still exists during re-init, closing it")
                    try:
                        await self._client.close()
                    except Exception as e:
                        logger.error("Error closing stale client: %s", e)
                    self._client = None
                    self._reset_session_bookkeeping()

                self._client = await self._create_client()
                await self._client.connect()
                logger.info("Connected transcription client (attempt %d/%d)", attempt, max_attempts)
                self._provider_generation += 1
                self._provider_audio_bytes = 0
                self._begin_provider_epoch_locked()
                self._register_client_handlers(self._client)

                self._active_model = requested_model
                self._provider_session_turns = 0
                self._provider_session_started_at = time.time()
                if publish_ready:
                    self._openai_ready.set()
                    await self._send_status("connected", turn_id)
                return True
            except Exception as e:
                logger.error("Failed to connect transcription client (attempt %d/%d): %s", attempt, max_attempts, e)
                self._openai_ready.clear()
                if self._client:
                    try:
                        await self._client.close()
                    except Exception as close_err:
                        logger.error("Error closing failed client: %s", close_err)
                    self._client = None
                    self._reset_session_bookkeeping()
                if attempt < max_attempts:
                    await asyncio.sleep(retry_delay_sec)
                    continue
                if emit_failure_error:
                    payload = {"type": "error", "content": "Failed to initialize transcription session"}
                    if turn_id is not None:
                        payload["turn_id"] = turn_id
                    await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
                return False

    async def _send_status(self, status: str, turn_id: Optional[int]):
        payload = {"type": "status", "status": status}
        if turn_id is not None:
            payload["turn_id"] = turn_id
        await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))

    async def _fail_segment_rollover_locked(self, reason: str) -> None:
        """Expose a real rollover failure before ending a still-recording turn."""

        self._assert_transport_locked()
        if self._finalized:
            return
        payload = {
            "type": "error",
            "content": "Transcription segment recovery failed",
        }
        if self._active_turn_id is not None:
            payload["turn_id"] = self._active_turn_id
        try:
            if self._websocket.client_state == WebSocketState.CONNECTED:
                await self._websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            logger.error(
                "Failed to send rollover error before %s (%s)",
                reason, type(exc).__name__,
            )
        await self._finalize_turn_locked(reason)

    async def _fail_segment_rollover(self, reason: str) -> None:
        async with self._transport_lock:
            await self._fail_segment_rollover_locked(reason)

    async def _send_recovery_event(self, event_type: str, *, attempts: Optional[int] = None):
        payload = {"type": event_type, "turn_id": self._active_turn_id}
        if attempts is not None:
            payload["attempts"] = attempts
        logger.info(
            "Recovery event type=%s turn=%s attempts=%s",
            event_type, self._active_turn_id, attempts if attempts is not None else "-",
        )
        try:
            if self._websocket.client_state == WebSocketState.CONNECTED:
                await self._websocket.send_text(
                    json.dumps(payload, ensure_ascii=False)
                )
        except Exception as exc:
            logger.warning(
                "Failed to send recovery event type=%s (%s)",
                event_type,
                type(exc).__name__,
            )

    async def _recover_transport_locked(
        self,
        turn_generation: int,
        *,
        planned_rollover: bool = False,
    ) -> bool:
        """Rebuild and replay-all with grapeot's five-attempt backoff policy."""

        self._assert_transport_locked()
        self._openai_ready.clear()
        if self._transport_state not in {TransportState.STOP_REQUESTED, TransportState.FINALIZED}:
            self._transport_state = TransportState.RECOVERING
        if not planned_rollover:
            await self._send_recovery_event("recovery_started")
        if self._client:
            try:
                await asyncio.wait_for(self._client.close(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self._client = None
            self._reset_session_bookkeeping()
        for attempt in range(1, self._RECONNECT_MAX_ATTEMPTS + 1):
            if attempt > 1:
                backoff = self._RECONNECT_BACKOFF_BASE_SEC * (2 ** (attempt - 2))
                await asyncio.sleep(backoff)
            if self._finalized or turn_generation != self._turn_generation:
                return False
            logger.info("Recovering provider transport (attempt %d/%d, turn_generation=%d)",
                        attempt, self._RECONNECT_MAX_ATTEMPTS, turn_generation)
            ok = await self._init_or_reuse_client(
                turn_id=self._active_turn_id,
                publish_ready=False,
                force_rebuild=True,
                emit_failure_error=False,
                max_attempts_override=1,
            )
            self._openai_ready.clear()
            if not ok or self._client is None:
                continue
            if not await self._replay_turn_audio_locked():
                continue

            if self._transport_state == TransportState.STOP_REQUESTED:
                if self._turn_audio_bytes < self._MIN_COMMIT_AUDIO_BYTES:
                    await self._finalize_turn_locked("insufficient_audio")
                    return True
                if self._turn_is_digital_silence_locked():
                    await self._finalize_turn_locked("digital_silence")
                    return True
                committed = await self._commit_transport_locked()
                if committed or self._finalized:
                    return True
                if self._finalize_commit_attempts >= self._FINALIZE_MAX_ATTEMPTS:
                    break
                continue
            await self._publish_transport_ready(
                publish_status=not planned_rollover
            )
            return True

        await self._send_recovery_event("recovery_failed", attempts=self._RECONNECT_MAX_ATTEMPTS)
        return False

    async def _on_provider_disconnect(self, provider_generation: Optional[int] = None):
        async with self._transport_lock:
            if self._finalized:
                return
            # A callback can queue behind a direct send/finalize recovery. Check
            # its epoch only after acquiring the sole transport owner so an old
            # generation cannot close and replace the new one (ABA).
            if (
                provider_generation is not None
                and provider_generation != self._provider_generation
            ):
                logger.info(
                    "Ignoring stale provider disconnect generation=%s current=%s",
                    provider_generation, self._provider_generation,
                )
                return
            logger.warning("Provider WebSocket disconnected unexpectedly")
            self._openai_ready.clear()
            if self._transcript_completed:
                # A confirmed completed payload is already authoritative. Its
                # scheduled owner is either the intermediate rollover coordinator
                # or the final S1 barrier; recovery here could erase that result.
                owner = (
                    "segment rollover coordinator"
                    if self._segment_rollover_pending
                    else "finalize barrier"
                )
                logger.info(
                    "Provider disconnected after completed authority; deferring to %s",
                    owner,
                )
                return
            finalize_in_flight = self._transport_state in {
                TransportState.STOP_REQUESTED,
                TransportState.COMMITTING,
                TransportState.COMMIT_ACKED,
            }
            if (
                finalize_in_flight
                and self._finalize_commit_attempts >= self._FINALIZE_MAX_ATTEMPTS
            ):
                if self._segment_rollover_pending:
                    await self._fail_segment_rollover_locked(
                        "provider_disconnect_after_finalize_retry"
                    )
                else:
                    await self._finalize_turn_locked(
                        "provider_disconnect_after_finalize_retry"
                    )
                return
            if finalize_in_flight:
                # A commit-send/ACK is not the completed transcript. Keep the
                # cache and make the replacement replay + recommit this turn.
                self._transport_state = TransportState.STOP_REQUESTED
                if not self._segment_rollover_pending:
                    self._is_recording = False
            else:
                self._transport_state = TransportState.RECOVERING
            turn_generation = self._turn_generation
            recovered = await self._recover_transport_locked(turn_generation)
            if recovered or self._finalized or turn_generation != self._turn_generation:
                return
            if self._segment_rollover_pending:
                await self._fail_segment_rollover_locked("reconnect_exhausted")
            else:
                await self._finalize_turn_locked("reconnect_exhausted")

    # ── client message handlers ──────────────────────────────────────────────
    async def _handle_start_recording(self, msg: dict):
        logger.info("Processing start_recording request")
        requested_turn_id = self._normalize_turn_id(msg.get("turn_id"))
        if msg.get("turn_id") is not None and requested_turn_id is None:
            logger.warning("Ignoring invalid turn_id from client: %r", msg.get("turn_id"))
        same_turn_retry = (
            requested_turn_id is not None
            and self._active_turn_id is not None
            and requested_turn_id == self._active_turn_id
            and not self._finalized
        )

        # `provider` / `model` / `prompt_mode` are accepted for protocol parity
        # with MicWave but ignored: WhisperWave always drives the server-selected
        # transcription model and vocabulary-bias prompt.
        legacy_provider = msg.get("provider")
        if legacy_provider is not None and not self._legacy_provider_warned:
            self._legacy_provider_warned = True
            logger.info("Ignoring legacy 'provider' field (=%r); transcription is provider-fixed", legacy_provider)

        input_sample_rate = msg.get("input_sample_rate")
        if input_sample_rate:
            try:
                self._audio_processor.set_source_sample_rate(int(input_sample_rate))
                logger.info("Using input sample rate: %sHz", self._audio_processor.source_sample_rate)
            except (TypeError, ValueError):
                logger.warning("Invalid input_sample_rate %r, using default", input_sample_rate)
                self._audio_processor.set_source_sample_rate(self._config.default_source_sample_rate)
        else:
            self._audio_processor.set_source_sample_rate(self._config.default_source_sample_rate)

        async with self._transport_lock:
            if not same_turn_retry:
                self._reset_turn_state(requested_turn_id)
                self._transport_state = TransportState.CONNECTING
            await self._send_status("connecting", self._active_turn_id)

            if same_turn_retry:
                logger.info(
                    "Same-turn start_recording retry for turn %s; preserving cache_chunks=%d bytes=%d",
                    requested_turn_id, len(self._turn_audio_chunks), self._turn_audio_bytes,
                )
                if self._transport_state in {
                    TransportState.STOP_REQUESTED,
                    TransportState.COMMITTING,
                    TransportState.COMMIT_ACKED,
                    TransportState.FINALIZED,
                }:
                    return
                if self._client is None or not self._client._is_ws_open():
                    self._transport_state = TransportState.RECOVERING
                    if not await self._recover_transport_locked(self._turn_generation):
                        await self._finalize_turn_locked("reconnect_exhausted")
                        return
                else:
                    self._openai_ready.clear()
                    synced = self._provider_audio_bytes == self._turn_audio_bytes
                    if not synced and self._provider_audio_bytes == 0:
                        synced = await self._replay_turn_audio_locked()
                    if not synced:
                        if not await self._recover_transport_locked(self._turn_generation):
                            await self._finalize_turn_locked("reconnect_exhausted")
                            return
                    else:
                        await self._publish_transport_ready()
                return

            self._openai_ready.clear()
            if not await self._init_or_reuse_client(
                turn_id=self._active_turn_id,
                publish_ready=False,
            ):
                self._transport_state = TransportState.IDLE
                return
            self._provider_session_turns += 1
            logger.info("Transcription session turn started: %d", self._provider_session_turns)

            # Clear the client transcript through the same output barrier used by
            # deltas/finalize, so an old in-flight delta cannot resume after the
            # new-turn reset.
            await self._reset_client_display()
            self._is_recording = True
            self._audio_cache.start_turn(self._active_turn_id)
            await self._publish_transport_ready()

        if self._config.force_reconnect_after_ms > 0:
            self._force_reconnect_task = asyncio.create_task(
                self._force_reconnect_after_delay(self._active_turn_id)
            )

    async def _handle_stop_recording(self, msg: dict):
        requested_turn_id = self._normalize_turn_id(msg.get("turn_id"))
        if (
            requested_turn_id is not None
            and self._active_turn_id is not None
            and requested_turn_id != self._active_turn_id
        ):
            logger.warning(
                "Ignoring stop_recording for stale turn_id=%s (active=%s)",
                requested_turn_id, self._active_turn_id,
            )
            return

        done_event = self._turn_done
        if done_event is None:
            logger.error("Turn has no completion Event, forcing finalization")
            await self._finalize_turn("missing_turn_done_event")
            return
        if self._finalized:
            return

        # User stop is distinct from the internal STOP_REQUESTED state used to
        # recommit a bounded prefix. Latch it even when a rollover commit already
        # owns `_commit_key`; the rollover coordinator will commit the tail next.
        self._stop_requested = True
        self._is_recording = False

        key = self._current_commit_key()
        if self._commit_key != key:
            # Latch before waiting for recovery/replay.  The current transport
            # owner observes this state and commits after its drains complete.
            previous_state = self._transport_state
            self._transport_state = TransportState.STOP_REQUESTED
            self._is_recording = False
            self._openai_ready.clear()
            logger.info(
                "Stop latched (turn=%s generation=%d previous_state=%s)",
                self._active_turn_id, self._turn_generation, previous_state.value,
            )
            async with self._transport_lock:
                if self._commit_key != self._current_commit_key() and not self._finalized:
                    if self._turn_audio_bytes < self._MIN_COMMIT_AUDIO_BYTES:
                        await self._finalize_turn_locked("insufficient_audio")
                    elif self._turn_is_digital_silence_locked():
                        await self._finalize_turn_locked("digital_silence")
                    else:
                        needs_resync = (
                            self._client is None
                            or not self._client._is_ws_open()
                            or self._provider_audio_bytes != self._turn_audio_bytes
                        )
                        if needs_resync:
                            logger.info(
                                "Finalize re-sync required (sent_bytes=%d cached_bytes=%d)",
                                self._provider_audio_bytes, self._turn_audio_bytes,
                            )
                            recovered = await self._recover_transport_locked(
                                self._turn_generation
                            )
                            if not recovered and not self._finalized:
                                await self._finalize_turn_locked("provider_unavailable")
                        else:
                            committed = await self._commit_transport_locked()
                            if not committed and not self._finalized:
                                recovered = await self._recover_transport_locked(
                                    self._turn_generation
                                )
                                if not recovered and not self._finalized:
                                    await self._finalize_turn_locked("provider_unavailable")

        while not self._finalized and not done_event.is_set():
            waited_attempt = self._finalize_commit_attempts
            try:
                await asyncio.wait_for(
                    done_event.wait(),
                    timeout=self._config.transcription_finalize_timeout_sec,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Transcription timed out after %.1fs (finalize_attempt=%d/%d)",
                    self._config.transcription_finalize_timeout_sec,
                    waited_attempt, self._FINALIZE_MAX_ATTEMPTS,
                )
                async with self._transport_lock:
                    if self._finalized or done_event.is_set():
                        return
                    # Another duplicate stop may already have advanced the
                    # provider attempt while this waiter timed out.
                    if self._finalize_commit_attempts != waited_attempt:
                        continue
                    if await self._retry_finalize_result_locked("timeout"):
                        continue
                    await self._finalize_turn_locked("timeout")
                    return

    async def _handle_audio_bytes(self, raw: bytes):
        processed_audio = self._audio_processor.process_audio_chunk(raw)
        segment_limit = max(1, self._config.max_turn_audio_bytes)
        for offset in range(0, len(processed_audio), segment_limit):
            await self._handle_processed_audio_chunk(
                processed_audio[offset:offset + segment_limit]
            )

    async def _handle_processed_audio_chunk(self, processed_audio: bytes) -> None:
        """Conserve one bounded chunk, waiting outside the owner on backpressure."""

        backpressure_deadline: Optional[float] = None
        while True:
            rollover_done: Optional[asyncio.Event] = None
            async with self._transport_lock:
                state = self._transport_state
                accepts_audio = (
                    not self._finalized
                    and self._active_turn_id is not None
                    and (
                        state in {TransportState.CONNECTING, TransportState.RECOVERING}
                        or (state == TransportState.RECORDING and self._is_recording)
                        or (self._segment_rollover_pending and self._is_recording)
                    )
                )
                if not accepts_audio:
                    logger.debug(
                        "Ignoring audio outside active recording window (state=%s)",
                        state.value,
                    )
                    return

                if self._segment_rollover_pending:
                    # Direction 2 intentionally does not upload this tail into the
                    # buffer whose prefix is awaiting its terminal. Once this next
                    # segment is full, stop consuming client frames until the
                    # prefix completes; TCP/WebSocket flow control supplies the
                    # bounded backpressure instead of growing replay memory.
                    if not self._append_rollover_tail_locked(processed_audio):
                        rollover_done = self._segment_rollover_done
                    else:
                        if self._is_recording:
                            self._processed_audio_bytes += len(processed_audio)
                            self._audio_cache.accumulate(processed_audio)
                        return
                else:
                    # The active segment cache mutates before send under the same
                    # serial owner.
                    if not self._append_turn_audio_locked(processed_audio):
                        # The crossing chunk belongs to the next segment. Input
                        # chunks are split to <=limit above, so an empty tail must
                        # accept it without exceeding the memory fence.
                        if not self._append_rollover_tail_locked(processed_audio):
                            raise RuntimeError(
                                "empty rollover tail rejected bounded audio chunk"
                            )
                        if self._is_recording:
                            self._processed_audio_bytes += len(processed_audio)
                            self._audio_cache.accumulate(processed_audio)
                        await self._begin_segment_rollover_locked()
                        return

                    if self._is_recording:
                        self._processed_audio_bytes += len(processed_audio)
                        self._audio_cache.accumulate(processed_audio)

                    can_send = (
                        state == TransportState.RECORDING
                        and self._openai_ready.is_set()
                        and self._client is not None
                        and self._is_recording
                    )
                    if not can_send:
                        logger.debug("Provider not ready; audio retained in turn cache")
                        return
                    if not await self._send_provider_audio_locked(processed_audio):
                        logger.warning(
                            "Provider send failed; full turn remains cached for replay"
                        )
                        self._mark_transport_sync_failed_locked()
                        turn_generation = self._turn_generation
                        recovered = await self._recover_transport_locked(turn_generation)
                        if (
                            not recovered
                            and not self._finalized
                            and turn_generation == self._turn_generation
                        ):
                            await self._finalize_turn_locked("reconnect_exhausted")
                    return

            if rollover_done is not None:
                if backpressure_deadline is None:
                    backpressure_deadline = (
                        asyncio.get_running_loop().time()
                        + self._segment_rollover_timeout_budget_sec()
                    )
                remaining = (
                    backpressure_deadline - asyncio.get_running_loop().time()
                )
                try:
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    await asyncio.wait_for(
                        rollover_done.wait(),
                        timeout=remaining,
                    )
                    # A watchdog always signals on exit. If its normal exit raced
                    # the rollover coordinator, yield once so the coordinator can
                    # promote the tail before this waiter rechecks state.
                    await asyncio.sleep(0)
                except asyncio.TimeoutError:
                    logger.error(
                        "Segment rollover receive backpressure timed out after %.1fs "
                        "(tail_bytes=%d)",
                        self._segment_rollover_timeout_budget_sec(),
                        self._rollover_tail_bytes,
                    )
                    self._cancel_segment_rollover_timeout_task()
                    self._cancel_segment_rollover_task()
                    async with self._transport_lock:
                        if self._finalized:
                            return
                        rollover_completed = (
                            not self._segment_rollover_pending
                            and self._transport_state == TransportState.RECORDING
                            and self._openai_ready.is_set()
                        )
                        if rollover_completed:
                            continue
                        await self._fail_segment_rollover_locked(
                            "segment_rollover_backpressure_timeout"
                        )
                    return

    async def run(self):
        logger.info("Transcription session run() started")
        try:
            await self._websocket.send_text(json.dumps({"type": "status", "status": "idle"}, ensure_ascii=False))
            while True:
                if self._websocket.client_state == WebSocketState.DISCONNECTED:
                    logger.info("WebSocket client disconnected")
                    self._openai_ready.clear()
                    break
                try:
                    data = await asyncio.wait_for(self._websocket.receive(), timeout=30.0)
                except asyncio.CancelledError:
                    logger.info("Receive messages task cancelled")
                    raise
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error("Error receiving message: %s", e, exc_info=True)
                    break

                if "bytes" in data:
                    await self._handle_audio_bytes(data["bytes"])
                elif "text" in data:
                    outer = json.loads(data["text"])
                    if outer.get("type") == "start_recording":
                        await self._handle_start_recording(outer)
                    elif outer.get("type") == "stop_recording":
                        await self._handle_stop_recording(outer)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Error in WebSocket connection: %s", e, exc_info=True)
        finally:
            logger.info("Receive messages loop ended")

    async def close(self):
        if self._closed:
            return
        self._closed = True
        self._segment_rollover_done.set()
        finalize_task = self._finalize_task
        self._finalize_task = None
        if finalize_task is not None and not finalize_task.done():
            finalize_task.cancel()
            try:
                await finalize_task
            except (asyncio.CancelledError, Exception):
                pass
        finalize_retry_task = self._finalize_retry_task
        self._finalize_retry_task = None
        if finalize_retry_task is not None and not finalize_retry_task.done():
            finalize_retry_task.cancel()
            try:
                await finalize_retry_task
            except (asyncio.CancelledError, Exception):
                pass
        ack_grace_task = self._ack_grace_task
        self._ack_grace_task = None
        if ack_grace_task is not None and not ack_grace_task.done():
            ack_grace_task.cancel()
            try:
                await ack_grace_task
            except (asyncio.CancelledError, Exception):
                pass
        force_reconnect_task = self._force_reconnect_task
        self._force_reconnect_task = None
        if force_reconnect_task is not None and not force_reconnect_task.done():
            force_reconnect_task.cancel()
            try:
                await force_reconnect_task
            except (asyncio.CancelledError, Exception):
                pass
        for task_attr in (
            "_segment_rollover_task",
            "_segment_rollover_timeout_task",
        ):
            task = getattr(self, task_attr)
            setattr(self, task_attr, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        async with self._transport_lock:
            if self._client:
                try:
                    await self._client.close()
                except Exception as e:
                    logger.error("Error closing client in close(): %s", e)
                self._client = None
        try:
            self._audio_cache.close()
        except Exception as e:
            logger.warning("Ignoring error closing audio_cache: %s", e)
        if self._websocket.client_state != WebSocketState.DISCONNECTED:
            try:
                await self._websocket.close()
            except RuntimeError as e:
                logger.warning("Ignoring error during websocket close: %s", e)
        logger.info("WebSocket connection closed for /api/v1/ws")


@app.websocket("/ws")
@app.websocket("/api/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    config = TranscriptionTurnSessionConfig.from_env()
    session = TranscriptionTurnSession(websocket, config)
    try:
        await session.run()
    finally:
        await session.close()


if __name__ == '__main__':
    # Default 23459 preserves the production port; WHISPERWAVE_PORT lets a probe /
    # test run an isolated instance (e.g. 23465) without colliding with the live app.
    port = int(os.getenv("WHISPERWAVE_PORT", "23459"))
    uvicorn.run(app, host="127.0.0.1", port=port)
