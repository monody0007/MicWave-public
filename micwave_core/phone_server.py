"""
MicWave Phone — Dedicated transcription backend for OpenCode iOS client.

Architecture (v2 — session-pooled):
  iPhone → POST /transcribe (m4a/pcm audio) → this server → OpenAI Realtime → text response

The server keeps a warm OpenAI Realtime WebSocket connection and reuses it
across requests, matching the session-reuse pattern of realtime_server.py.
Uploaded audio is decoded in memory by PyAV on a worker thread.

Port: 23458 (launchd, tslink registered)

Every completed turn's final text is appended (off the request path) to the
shared transcript store the Mac IME uses, transcripts.jsonl, with
source="phone" and the turn's WAV name; see projects/EchoWave/transcript_store.py.
"""
import asyncio
import functools
import io
import itertools
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

try:
    import av
except ImportError:  # pragma: no cover - optional [phone] extra
    av = None  # type: ignore[assignment]
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .openai_realtime_client import CommitNotAcknowledgedError, OpenAIRealtimeAudioTextClient
from .audio_gain import measure_speech_activity_pcm16, preprocess_speech_pcm16, split_pcm_on_pauses
from .audio_persistence import TurnAudioCache
from .prompts import get_realtime_prompt, get_optimize_prompt
from .config import OPENAI_REALTIME_MODEL
from .transcript_store import SOURCE_PHONE, append_record, history_file_path, now_ts
from .transcript_merge import join_segment_texts
from .realtime_text_utils import (
    INCOMPLETE_TURN_MARKER,
    SIMILARITY_HARD_CAP_CHARS,
    OutputTruncationAssessment,
    answer_guard_fingerprint,
    assess_output_truncation,
    emitted_novel_material_ratio,
    extract_text_after_marker,
    is_likely_truncated_input_transcript,
    is_runaway_repetition_transcript,
    parse_ratio_env,
    scaled_suspicious_output_chars,
    transcription_similarity_ratio,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("micwave_phone")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the API key cache off the event loop so the listener is up at once;
    # /health reports api_key_configured=false until the thread finishes. A
    # failed prewarm only logs; the next /transcribe retries the fetch.
    prewarm = threading.Thread(
        target=_prewarm_api_key,
        name="micwave-api-key-prewarm",
        daemon=True,
    )
    prewarm.start()
    app.state.api_key_prewarm_thread = prewarm
    try:
        yield
    finally:
        _drain_transcript_writes()
        _shutdown_audio_cache()

PORT = int(os.getenv("MICWAVE_PHONE_PORT", "23458"))
TARGET_SAMPLE_RATE = 24000
CHUNK_SIZE_BYTES = 24000
MAX_SESSION_TURNS = int(os.getenv("MICWAVE_PHONE_MAX_TURNS", "8"))
MAX_SESSION_AGE_SEC = int(os.getenv("MICWAVE_PHONE_MAX_AGE_SEC", "7200"))
PHONE_TRANSCRIPTION_MARKER = "下面是不改变语言的语音识别结果："
SUSPICIOUS_AUDIO_THRESHOLD_SEC = 20.0
SUSPICIOUS_OUTPUT_THRESHOLD_CHARS = 120
# Speech front-end before upload (task 1193): 100 Hz high-pass + gated dynamic
# AGC (projects/EchoWave/audio_gain.py). Recordings longer than the soft limit
# are split at real pauses, each segment at most the hard limit long.
def _positive_float_env(name: str, default: float, minimum: float) -> float:
    """Finite float env value >= minimum; anything else warns and returns default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < minimum:
        logging.getLogger("micwave_phone").warning(
            "%s=%r is not a finite number >= %s; using %s", name, raw, minimum, default
        )
        return default
    return value


INPUT_AGC_ENABLED = os.getenv("MICWAVE_PHONE_INPUT_AGC_ENABLED", "1") == "1"
PHONE_SOFT_SEGMENT_SEC = _positive_float_env("MICWAVE_PHONE_SOFT_SEGMENT_SEC", 180.0, 30.0)
PHONE_HARD_SEGMENT_SEC = _positive_float_env(
    "MICWAVE_PHONE_HARD_SEGMENT_SEC", max(240.0, PHONE_SOFT_SEGMENT_SEC), PHONE_SOFT_SEGMENT_SEC
)
INCOMPLETE_RESPONSE_STATUSES = ("incomplete", "failed", "cancelled")
# Grace for the completed input transcript after response.done, scaled by
# recording length (task 0809). The completed gpt-4o-transcribe event for a long
# recording lands well after response.done (465s audio -> 6.1s late on the Mac
# lane, 2026-09-24); a flat 3s grace turned such no-marker turns into a 503.
# Effective grace = min(max(MAX, BASE), BASE + PER_AUDIO * audio_sec), then
# clamped to what is left of the transcription budget.
COMPLETED_INPUT_TRANSCRIPT_GRACE_BASE_SEC = float(os.getenv("MICWAVE_PHONE_INPUT_TRANSCRIPT_GRACE_BASE_SEC", "3.0"))
COMPLETED_INPUT_TRANSCRIPT_GRACE_PER_AUDIO_SEC = float(os.getenv("MICWAVE_PHONE_INPUT_TRANSCRIPT_GRACE_PER_AUDIO_SEC", "0.03"))
COMPLETED_INPUT_TRANSCRIPT_GRACE_MAX_SEC = float(os.getenv("MICWAVE_PHONE_INPUT_TRANSCRIPT_GRACE_MAX_SEC", "60"))
MATERIAL_REPLACEMENT_RATIO = 1.75
MATERIAL_REPLACEMENT_DELTA_CHARS = 80
# Transcription budget, scaled by recording length. The flat 60s cap this
# replaced killed responses that were actively streaming: a 2348s recording
# (2026-08-13 T247) had OpenAI emitting text at 19s and was cancelled at
# exactly 60.0s with the partial discarded. Both prefill and generated text
# grow with the recording (prefill measured 1.2s at 272s of audio, 19s at
# 2348s), so the budget grows with it too.
# The 300s floor is deliberate: cutting off a working transcription costs a
# recording that cannot be redone, while the failure it guards against (a hung
# turn holding the single-turn lock) has happened twice in the service's
# lifetime. Waiting too long is the cheaper mistake here, so every recording
# gets at least five minutes and long ones ramp to ten.
TRANSCRIBE_TIMEOUT_BASE_SEC = float(os.getenv("MICWAVE_PHONE_TIMEOUT_BASE_SEC", "300"))
TRANSCRIBE_TIMEOUT_PER_AUDIO_SEC = float(os.getenv("MICWAVE_PHONE_TIMEOUT_PER_AUDIO_SEC", "0.25"))
TRANSCRIBE_TIMEOUT_MAX_SEC = float(os.getenv("MICWAVE_PHONE_TIMEOUT_MAX_SEC", "600"))
# Answer-similarity guard thresholds (task 0436 G1/M4). <=0 disables similarity.
# Shared [0,1] parser (task 0436 N1): a bogus/inf/out-of-range value warns and
# falls back to the default instead of crashing at import. Default 0.60 matches
# the streaming lane; the novel-material ratio is a second signal for
# restate-then-answer forms whose similarity stays above threshold.
ANSWER_GUARD_MIN_SIMILARITY = parse_ratio_env("BRAINWAVE_ANSWER_GUARD_MIN_SIMILARITY", 0.60)
ANSWER_GUARD_NOVEL_MATERIAL_RATIO = parse_ratio_env("BRAINWAVE_ANSWER_GUARD_NOVEL_MATERIAL_RATIO", 0.55)

_audio_cache = TurnAudioCache.from_env()
_turn_ids = itertools.count(1)
_audio_cache_lock = asyncio.Lock()

# Completed turns are appended to the shared transcript store (the Mac IME's
# transcripts.jsonl, see projects/EchoWave/transcript_store.py) by one worker
# thread, so the response never waits on disk and records keep turn order.
TRANSCRIPT_DRAIN_TIMEOUT_SEC = 5.0
_transcript_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="micwave-transcript-store")


def _write_transcript_record(record: dict) -> None:
    try:
        append_record(history_file_path(), record)
    except Exception as exc:
        # The text itself stays out of the log (private notes); audio_file
        # identifies the turn and the WAV still holds the audio.
        logger.error(
            "Transcript store append failed (audio_file=%s): %s",
            record.get("audio_file"),
            exc,
            exc_info=True,
        )


def _submit_transcript_record(text: str, duration_s: float, audio_file: Optional[str], turn_id: int) -> None:
    if not text.strip():
        return
    record = {
        "ts": now_ts(),
        "text": text,
        "mode": "phone",
        "source": SOURCE_PHONE,
        "backend": "micwave_phone",
        "provider": "openai",
        "model": OPENAI_REALTIME_MODEL,
        "duration_s": round(duration_s, 2),
        "audio_file": audio_file,
        "turn_id": turn_id,
    }
    try:
        _transcript_writer.submit(_write_transcript_record, record)
    except Exception as exc:
        logger.error("Transcript store submit failed (audio_file=%s): %s", audio_file, exc, exc_info=True)


def _drain_transcript_writes() -> None:
    """Wait for queued store appends (shutdown path); never raises."""
    try:
        _transcript_writer.submit(lambda: None).result(timeout=TRANSCRIPT_DRAIN_TIMEOUT_SEC)
    except Exception as exc:
        logger.error("Transcript store drain failed: %s", exc)


def transcription_timeout_sec(audio_duration_sec: float) -> float:
    """Wall-clock budget for one OpenAI Realtime turn, given the audio length."""
    budget = TRANSCRIBE_TIMEOUT_BASE_SEC + TRANSCRIBE_TIMEOUT_PER_AUDIO_SEC * audio_duration_sec
    return min(TRANSCRIBE_TIMEOUT_MAX_SEC, max(TRANSCRIBE_TIMEOUT_BASE_SEC, budget))


def completed_input_transcript_grace_sec(
    audio_duration_sec: float,
    remaining_budget_sec: Optional[float] = None,
) -> float:
    """How long to wait for the completed input transcript after response.done.

    Grows with the recording; never exceeds the remaining transcription budget.
    """
    base = COMPLETED_INPUT_TRANSCRIPT_GRACE_BASE_SEC
    grace = base + COMPLETED_INPUT_TRANSCRIPT_GRACE_PER_AUDIO_SEC * max(0.0, audio_duration_sec)
    grace = min(max(COMPLETED_INPUT_TRANSCRIPT_GRACE_MAX_SEC, base), grace)
    if remaining_budget_sec is not None:
        grace = min(grace, remaining_budget_sec)
    return max(0.0, grace)


def _shutdown_audio_cache() -> None:
    _audio_cache.close()
    # Consumer-side compensation; close() does not join, deferred to audio_persistence sync-matrix work.
    worker = getattr(_audio_cache, "_worker", None)
    if worker is not None:
        worker.join(timeout=3.0)


# API key cache (task 0607, invariant I4). Unless the host injects
# `OPENAI_API_KEY`, the key comes from the host-configured key command
# (`MICWAVE_OPENAI_KEY_COMMAND`, shell-split, run without a shell, stdout is the
# key), at most until one non-empty value is cached; /health only reads the
# cache. On a slow host a per-request command (10s budget) pushed /health past
# the watchdog deadline, which kickstarted a healthy service every 5 minutes.
KEY_COMMAND_ENV = "MICWAVE_OPENAI_KEY_COMMAND"
_api_key_cache = ""
_api_key_lock = threading.Lock()


def _fetch_api_key_from_command() -> str:
    """Run the key command once; any failure is an empty fetch (not cached, retried next call).

    Only the exit status or exception type is logged: the command's output may be the secret.
    """
    command = os.getenv(KEY_COMMAND_ENV, "").strip()
    if not command:
        return ""
    try:
        result = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=10)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.warning("API key command failed: %s", type(exc).__name__)
        return ""
    if result.returncode != 0:
        logger.warning("API key command failed: exit %s", result.returncode)
        return ""
    return result.stdout.strip()


def load_api_key() -> str:
    """Return the API key; `OPENAI_API_KEY` wins, otherwise the key command once then cache.

    An empty fetch is not cached, so the next call retries. Callers on the event
    loop should run this via `asyncio.to_thread` because the first fetch blocks.
    """
    global _api_key_cache
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        return key
    with _api_key_lock:
        if _api_key_cache:
            return _api_key_cache
        key = _fetch_api_key_from_command()
        if key:
            _api_key_cache = key
        return key


def cached_api_key() -> str:
    """Read-only view for /health: env or cache, never a subprocess, never the lock."""
    return os.getenv("OPENAI_API_KEY", "") or _api_key_cache


def reset_api_key_cache() -> None:
    """Test hook: forget the cached key so the next load_api_key() fetches again."""
    global _api_key_cache
    with _api_key_lock:
        _api_key_cache = ""


def _prewarm_api_key() -> None:
    try:
        if load_api_key():
            logger.info("API key cache warmed at startup")
        else:
            logger.warning("API key prewarm returned empty; /transcribe will retry the fetch")
    except Exception as exc:
        logger.warning("API key prewarm failed: %s", exc)


def convert_audio_to_pcm16(audio_bytes: bytes, source_format: str = "m4a") -> bytes:
    """Decode in memory to 24kHz s16le mono PCM; detect format from content.

    source_format is retained for callers that pass the upload extension.
    """
    if not audio_bytes:
        raise ValueError("Empty audio file")
    if av is None:
        raise RuntimeError(
            "PyAV ('av') is required for audio container decoding; install micwave-core[phone]"
        )
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)
    chunks = []
    with av.open(io.BytesIO(audio_bytes), mode="r") as container:
        stream = container.streams.audio[0]
        for packet in container.demux(stream):
            if stream.codec_context.name == "aac":
                # Preserve the final AAC block like the previous ffmpeg CLI.
                # Newer libav exports end padding; keep start/priming skips.
                skip = packet.get_sidedata("skip_samples")
                data = bytes(skip)
                if data and int.from_bytes(data[4:8], "little"):
                    skip.update(data[:4] + b"\x00" * 4 + data[8:])
                    packet.set_sidedata(skip)
            for frame in packet.decode():
                for output in resampler.resample(frame):
                    chunks.append(output.to_ndarray().astype("<i2", copy=False).tobytes())
        # Drain the delayed samples so the recording's tail is preserved.
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().astype("<i2", copy=False).tobytes())
    pcm_data = b"".join(chunks)
    if not pcm_data:
        raise ValueError("No audio samples decoded")
    return pcm_data


def _first_text_value(data: dict | None) -> str:
    if not isinstance(data, dict):
        return ""
    for key in ("transcript", "text", "delta"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _content_parts_text(parts) -> str:
    if not isinstance(parts, list):
        return ""
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        value = _first_text_value(part)
        if value:
            text_parts.append(value)
    return "".join(text_parts)


def extract_input_transcription(event: dict) -> str:
    """Extract input transcription text from a realtime transcription event."""
    item = event.get("item") if isinstance(event, dict) else None
    for container in (event, item):
        value = _first_text_value(container)
        if value:
            return value.strip()
    for parts in (
        event.get("content") if isinstance(event, dict) else None,
        item.get("content") if isinstance(item, dict) else None,
    ):
        value = _content_parts_text(parts)
        if value:
            return value.strip()
    return ""


def _completed_input_transcript_for_replacement(event: dict) -> tuple[str, bool]:
    completed_text = extract_input_transcription(event)
    return completed_text, bool(completed_text)


def _clean_phone_result_text(result_text: str) -> tuple[bool, str]:
    found, cleaned = extract_text_after_marker(result_text, PHONE_TRANSCRIPTION_MARKER)
    if not found and result_text.strip():
        cleaned = result_text.strip()
    return found, cleaned.strip()


def _has_material_input_replacement(input_transcript_text: str, cleaned_output_text: str) -> bool:
    input_len = len(input_transcript_text.strip())
    output_len = len(cleaned_output_text.strip())
    return (
        input_len - output_len >= MATERIAL_REPLACEMENT_DELTA_CHARS
        and input_len >= output_len * MATERIAL_REPLACEMENT_RATIO
    )


def _has_suspicious_marker_output(result_text: str, audio_duration_sec: float) -> bool:
    marker_found, cleaned_output = _clean_phone_result_text(result_text)
    return (
        marker_found
        and audio_duration_sec >= SUSPICIOUS_AUDIO_THRESHOLD_SEC
        and len(cleaned_output)
        <= scaled_suspicious_output_chars(audio_duration_sec, SUSPICIOUS_OUTPUT_THRESHOLD_CHARS)
    )


def _remove_marker_literals(block: str) -> str:
    """Drop every marker literal inside a block, keeping all text around it."""
    pieces = [piece.strip() for piece in block.split(PHONE_TRANSCRIPTION_MARKER)]
    return "\n".join(piece for piece in pieces if piece)


def _assemble_response_output_parts(
    parts: Sequence[str],
    droppable_preamble: Optional[Sequence[bool]] = None,
) -> tuple[bool, str]:
    """``(marker_found, body)`` from the text of every output part, in order.

    Blocks before the first block that carries the marker are the model's
    preamble, as is text before the marker inside that block (the existing
    single-block contract of extract_text_after_marker). After that, every
    later block is body text: a marker literal it repeats (at its head or
    anywhere) is removed, and all text on both sides of it is kept (W0322 R2:
    stripping up to an inner marker deleted real speech). Blocks are joined on
    new lines. Without any marker, all blocks are joined as they are.

    ``droppable_preamble`` (aligned with ``parts``, default all True) says
    which blocks may be dropped as preamble at all; a block marked False keeps
    its whole text, marker literals removed, wherever it sits (W0322 R8: a
    part in conflict with the marker part is speech, not preamble).
    """
    if droppable_preamble is None:
        droppable_preamble = [True] * len(parts)
    if len(droppable_preamble) != len(parts):
        raise ValueError("droppable_preamble must align with parts")
    kept = [(p, drop) for p, drop in zip(parts, droppable_preamble) if p and p.strip()]
    for index, (block, droppable) in enumerate(kept):
        found, after = extract_text_after_marker(block, PHONE_TRANSCRIPTION_MARKER)
        if not found:
            continue
        preamble = [_remove_marker_literals(b) for b, drop in kept[:index] if not drop]
        head = after.strip() if droppable else _remove_marker_literals(block)
        bodies = preamble + [head] + [_remove_marker_literals(b) for b, _ in kept[index + 1:]]
        return True, "\n".join(body for body in bodies if body)
    return False, "\n".join(block.strip() for block, _ in kept)


@dataclass
class _OutputSlot:
    output_index: int
    content_index: int
    item_id: Optional[str]
    seq: int
    delta: str = ""
    done: Optional[str] = None
    # Position given by the provider: an explicit output_index on an event, or
    # a response.done output entry this slot was bound to (W0322 R4).
    positioned: bool = False
    # Which of the two aliased done event types completed this part (W0322 R5).
    done_types: set = field(default_factory=set)
    # One side of a conflict over a part's text (W0322 R8): it is speech that
    # disagrees with another part, never droppable as model preamble.
    conflicted: bool = False

    def text(self) -> str:
        return self.done if self.done is not None else self.delta

    @property
    def unpositioned(self) -> bool:
        """No item_id and no provider position: where it belongs is unknown."""
        return self.item_id is None and not self.positioned


def _event_index(data: dict, key: str) -> Optional[int]:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


# The provider may deliver one text part under both names; a done of each type
# with the same text is the only provable duplicate of an unpositioned part.
_ALIASED_DONE_TYPES = frozenset({"response.text.done", "response.output_text.done"})


def _output_body(text: str) -> str:
    """Text of a part without marker literals, for content matching."""
    return _remove_marker_literals(text or "")


class _ResponseOutputCollector:
    """Text of every output part of one response, ordered by provider position.

    A part is identified by ``(item_id, content_index)`` when the event names
    its item, else by an explicit ``(output_index, content_index)``. An event
    with neither is unpositioned: it is never assumed to be output 0 (W0322 R4:
    that silently overwrote the first item). Consecutive unpositioned events
    build one part until it is done. A part's done text replaces its streamed
    deltas. Only provable matches merge (W0322 R5): an unpositioned done equal
    to the open part's done merges only when it is the other of
    response.text.done / response.output_text.done (task 1193); any other
    later done starts a new part. response.done's output list is the
    authoritative snapshot and is reconciled part by part (W0322 R2): a part
    matches by ID, then by explicit position; an unpositioned part binds when
    its body equals exactly one snapshot part and that part is still unmatched,
    or, still streaming without a done, when it is the prefix of the
    response's only part. An unpositioned part nothing proves is never
    dropped: it is kept after every positioned part. Whenever text was kept
    without proof of its order or that it is not a duplicate, ``ambiguous`` is
    set so the caller can report the result as uncertain. Positioned text is
    ordered by ``(output_index, content_index, arrival)``, unpositioned text
    by arrival; so is text named only by item_id, without a position.
    A second done with different text for a done part never overwrites it.
    A snapshot part binds a slot only when consistent with it (W0322 R7):
    equal body for a done part, a body extending its deltas otherwise; on a
    conflict both texts are kept and ``ambiguous`` is set, as it is for any
    non-empty part the snapshot leaves unreconciled and for a delta arriving
    after its part's done.
    """

    def __init__(self):
        self._slots: list[_OutputSlot] = []
        self._ambiguous = False

    @property
    def ambiguous(self) -> bool:
        """Text was kept without proof of its place or identity (W0322 R6).

        Beyond what reconciliation flagged, two or more non-empty parts are
        ambiguous whenever one of them has no provider position (its order and
        whether it repeats another part are unproven), or two of them claim
        the same ``(output_index, content_index)``.
        """
        if self._ambiguous:
            return True
        non_empty = [slot for slot in self._slots if (slot.text() or "").strip()]
        if len(non_empty) < 2:
            return False
        if any(not slot.positioned for slot in non_empty):
            return True
        positions = {(slot.output_index, slot.content_index) for slot in non_empty}
        return len(positions) < len(non_empty)

    def _find(self, item_id: Optional[str], output_index: Optional[int], content_index: int):
        """The slot of a positioned part, or None. A named part matches its own
        ``(item_id, content_index)``, else an unnamed slot at the same explicit
        position; an unnamed part matches the one positioned slot there."""
        def unique(candidates):
            return candidates[0] if len(candidates) == 1 else None

        if item_id:
            # The latest slot of the part: a conflicting done or a late delta
            # opens a new slot for it, and later events continue that one.
            for slot in reversed(self._slots):
                if slot.item_id == item_id and slot.content_index == content_index:
                    return slot
        if output_index is None:
            return None
        at_position = [
            slot for slot in self._slots
            if slot.positioned
            and slot.output_index == output_index
            and slot.content_index == content_index
        ]
        if item_id:
            return unique([slot for slot in at_position if slot.item_id is None])
        return unique([slot for slot in at_position if slot.item_id is None]) or unique(at_position)

    def _open_unpositioned(self, content_index: int) -> Optional[_OutputSlot]:
        """The latest unpositioned slot of this content index, if any."""
        for slot in reversed(self._slots):
            if slot.unpositioned and slot.content_index == content_index:
                return slot
        return None

    def _new_slot(self, item_id, output_index, content_index) -> _OutputSlot:
        slot = _OutputSlot(
            output_index if output_index is not None else 0,
            content_index,
            item_id,
            len(self._slots),
            positioned=output_index is not None,
        )
        self._slots.append(slot)
        return slot

    def _slot(self, data: dict, done_text: Optional[str] = None) -> _OutputSlot:
        item_id = data.get("item_id") or None
        output_index = _event_index(data, "output_index")
        content_index = _event_index(data, "content_index") or 0
        if item_id is None and output_index is None:
            open_slot = self._open_unpositioned(content_index)
            if open_slot is not None:
                if open_slot.done is None:
                    return open_slot
                if done_text is not None and open_slot.done == done_text:
                    event_type = data.get("type")
                    if (
                        event_type in _ALIASED_DONE_TYPES
                        and open_slot.done_types
                        and open_slot.done_types <= _ALIASED_DONE_TYPES
                        and event_type not in open_slot.done_types
                    ):
                        return open_slot
                    # A repeated callback or a second part with the same
                    # text: nothing tells them apart, so keep both and say so.
                    self._ambiguous = True
            return self._new_slot(None, None, content_index)
        slot = self._find(item_id, output_index, content_index)
        if slot is None and done_text is not None:
            # A done naming its part completes unpositioned deltas it extends.
            open_slot = self._open_unpositioned(content_index)
            if (
                open_slot is not None
                and open_slot.done is None
                and open_slot.delta
                and done_text.startswith(open_slot.delta)
            ):
                slot = open_slot
        if slot is None:
            return self._new_slot(item_id, output_index, content_index)
        if item_id and slot.item_id is None:
            slot.item_id = item_id
        if output_index is not None and not slot.positioned:
            slot.output_index, slot.positioned = output_index, True
        return slot

    def add_delta(self, data: dict) -> None:
        delta = data.get("delta") or ""
        slot = self._slot(data)
        if slot.done is not None and delta:
            # A delta after the part's done would vanish (text() reads done):
            # keep it in a slot of its own and say so (W0322 R7).
            slot = self._conflict_slot(slot)
        slot.delta += delta

    def _conflict_slot(self, slot: _OutputSlot) -> _OutputSlot:
        """A new slot at ``slot``'s place for text that conflicts with it;
        both sides are kept and marked, and the result is ambiguous."""
        self._ambiguous = True
        slot.conflicted = True
        new_slot = self._new_slot(
            slot.item_id,
            slot.output_index if slot.positioned else None,
            slot.content_index,
        )
        new_slot.conflicted = True
        return new_slot

    def add_done(self, data: dict) -> None:
        text = data.get("text")
        if not isinstance(text, str):
            return
        slot = self._slot(data, done_text=text)
        if slot.done is None and self._consistent(slot, text):
            # The done completes the streamed deltas, which may carry the
            # marker the done text lacks.
            if PHONE_TRANSCRIPTION_MARKER in slot.delta and PHONE_TRANSCRIPTION_MARKER not in text:
                slot.done = f"{PHONE_TRANSCRIPTION_MARKER}\n\n{text}"
            else:
                slot.done = text
        elif slot.done is not None and slot.done == text:
            pass
        else:
            # A done that is not provably the part's text (different from its
            # done (W0322 R6), or not extending its deltas (W0322 R8)): never
            # overwrite; keep both and say so. An empty done adds nothing.
            if not text.strip():
                self._ambiguous = True
                slot.conflicted = True
                return
            slot = self._conflict_slot(slot)
            slot.done = text
        event_type = data.get("type")
        if event_type in _ALIASED_DONE_TYPES:
            slot.done_types.add(event_type)

    @staticmethod
    def _consistent(slot: _OutputSlot, text: str) -> bool:
        """``text`` provably is this slot's part (W0322 R7/R8): equal bodies
        for a done part; for a part still streaming, ``text`` extends its raw
        deltas (a marker cut off mid-stream included), or its non-empty body
        extends theirs (deltas carrying the marker, ``text`` only the body).
        Anything else, one side empty included, is a conflict. Deltas of only
        whitespace carry nothing to conflict with."""
        if slot.done is not None:
            return _output_body(slot.done) == _output_body(text)
        if not slot.delta.strip() or text.startswith(slot.delta):
            return True
        delta_body = _output_body(slot.delta)
        return bool(delta_body) and _output_body(text).startswith(delta_body)

    def _snapshot_candidates(self, item_id, output_index, content_index, reconciled):
        """Every slot not yet reconciled that may be this snapshot part: its
        own ``(item_id, content_index)``, or the same explicit position with
        no other item_id. A conflict leaves several slots at one place."""
        return [
            slot for slot in self._slots
            if id(slot) not in reconciled
            and (
                (item_id and slot.item_id == item_id and slot.content_index == content_index)
                or (
                    slot.positioned
                    and slot.output_index == output_index
                    and slot.content_index == content_index
                    and (slot.item_id is None or not item_id or slot.item_id == item_id)
                )
            )
        ]

    @staticmethod
    def _bind(slot: _OutputSlot, item_id, output_index, content_index, text) -> None:
        """Give a consistent slot the snapshot's place and complete its text,
        keeping a marker either side carries."""
        slot.output_index, slot.content_index, slot.positioned = output_index, content_index, True
        if item_id and slot.item_id is None:
            slot.item_id = item_id
        marker_in_text = PHONE_TRANSCRIPTION_MARKER in text
        if slot.done is not None:
            if marker_in_text and PHONE_TRANSCRIPTION_MARKER not in slot.done:
                slot.done = text
            return
        # Deltas without a done event may stop short of the part.
        if PHONE_TRANSCRIPTION_MARKER in slot.delta and not marker_in_text:
            slot.done = f"{PHONE_TRANSCRIPTION_MARKER}\n\n{text}"
        else:
            slot.done = text

    def add_response_output(self, response: dict) -> None:
        """Reconcile with response.done's output list, part by part."""
        output = response.get("output")
        if not isinstance(output, list):
            return
        snapshot = []
        for output_index, item in enumerate(output):
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_index, part in enumerate(content):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    snapshot.append((item.get("id") or None, output_index, content_index, part["text"]))
        before = [slot for slot in self._slots if (slot.text() or "").strip()]
        reconciled = set()
        unmatched = []
        conflicting = []
        for entry in snapshot:
            slot = self._find(*entry[:3])
            if slot is None or id(slot) in reconciled or not self._consistent(slot, entry[3]):
                # Several slots at one place (a conflict split it, W0322 R8):
                # bind the only one the snapshot text is consistent with.
                candidates = self._snapshot_candidates(*entry[:3], reconciled)
                consistent = [c for c in candidates if self._consistent(c, entry[3])]
                slot = consistent[0] if len(consistent) == 1 else None
                if slot is None and candidates:
                    # Event and snapshot disagree on one part: neither text is
                    # provably the other, so keep both and say so (W0322 R7).
                    self._ambiguous = True
                    conflicting.append(entry)
            if slot is not None:
                self._bind(slot, *entry)
                reconciled.add(id(slot))
                continue
            unmatched.append(entry)
        loose = [slot for slot in self._slots if slot.unpositioned]
        for slot in loose:
            body = _output_body(slot.text())
            match = None
            if body:
                # Equal to exactly one part of the whole snapshot, and that
                # part has no slot yet: provably the same part.
                equal = [e for e in snapshot if _output_body(e[3]) == body]
                if len(equal) == 1 and equal[0] in unmatched:
                    match = equal[0]
            if (
                match is None
                and slot.done is None
                and len(snapshot) == 1
                and len(loose) == 1
                and unmatched
            ):
                # Deltas that never got a done, under the response's only
                # part: a prefix of it (or an empty part) is that part.
                part_body = _output_body(unmatched[0][3])
                if part_body.startswith(body):
                    match = unmatched[0]
            if match is not None and self._consistent(slot, match[3]):
                unmatched.remove(match)
                self._bind(slot, *match)
                reconciled.add(id(slot))
        for entry in unmatched:
            item_id, output_index, content_index, text = entry
            if text.strip():
                new_slot = self._new_slot(item_id, output_index, content_index)
                new_slot.done = text
                new_slot.conflicted = entry in conflicting
        for slot in before:
            if id(slot) not in reconciled:
                # Text the snapshot does not account for (an empty output list,
                # other item_ids, a conflicting part): kept, never dropped,
                # never taken for preamble (W0322 R8).
                slot.conflicted = True
                self._ambiguous = True
        if snapshot and any(
            slot.unpositioned and _output_body(slot.text()) for slot in self._slots
        ):
            # Kept, never dropped: it may be a duplicate of a snapshot part or
            # a part the snapshot lacks, and its place is unknown.
            self._ambiguous = True

    def _ordered(self) -> list[_OutputSlot]:
        def order(slot):
            if not slot.positioned:
                return (1, slot.seq, 0, 0)
            return (0, slot.output_index, slot.content_index, slot.seq)

        return sorted(self._slots, key=order)

    def parts(self) -> list[str]:
        return [slot.text() for slot in self._ordered()]

    def text(self) -> str:
        """The response as one string: a found marker is rendered at the head
        so _clean_phone_result_text sees the same text as a single item.

        Only a part provably before the first marker part, and in no conflict,
        is dropped as preamble: both placed, at a smaller position, not
        conflicted (W0322 R8). A conflicted, same-position or unplaced part is
        kept, and a conflicted marker part keeps its text before the marker.
        """
        ordered = [slot for slot in self._ordered() if (slot.text() or "").strip()]
        first = next(
            (slot for slot in ordered if PHONE_TRANSCRIPTION_MARKER in slot.text()), None,
        )
        droppable = []
        for slot in ordered:
            if first is None or slot is first:
                droppable.append(first is None or not first.conflicted)
                continue
            droppable.append(
                slot.positioned
                and first.positioned
                and not slot.conflicted
                and (slot.output_index, slot.content_index)
                < (first.output_index, first.content_index)
            )
        marker_found, body = _assemble_response_output_parts(
            [slot.text() for slot in ordered], droppable,
        )
        if marker_found:
            return f"{PHONE_TRANSCRIPTION_MARKER}\n\n{body}"
        return body


# Decisions without a trustworthy text; the pool answers them with a 503.
FAIL_CLOSED_DECISION_REASONS = frozenset({
    "no_marker_unavailable",
    "no_marker_input_repetition",
    "realtime_repetition_unavailable",
})


@dataclass
class PhoneTranscriptionDecision:
    """Outcome of choosing the final phone transcription text (task 0436 G4)."""
    final_text: str
    marker_found: bool
    replaced: bool
    reason: str
    realtime_len: int
    input_len: int
    similarity: Optional[float]


def decide_final_phone_transcription(
    result_text: str,
    input_transcript_text: str,
    input_transcript_completed: bool,
    audio_duration_sec: float,
    response_status: Optional[str] = None,
    *,
    active_speech_sec: Optional[float] = None,
    tail_active_speech_sec: Optional[float] = None,
    output_ambiguous: bool = False,
) -> PhoneTranscriptionDecision:
    """Choose the final text; see _decide_by_content for the precedence.

    ``active_speech_sec`` / ``tail_active_speech_sec`` are the acoustic
    evidence for the truncated-transcript check (None = text-only rules).

    When the provider ended the response as incomplete/failed/cancelled
    (task 1193) a longer completed, non-looping input transcript is adopted,
    and whatever text is returned gets INCOMPLETE_TURN_MARKER on its own line:
    a completed transcript event does not prove it covers the recording (T80
    completed after two sentences), so an abnormal end is always shown
    (W0322 R2). An empty result raises 502.

    ``output_ambiguous`` (the collector kept output parts it could not place
    or dedupe with proof, W0322 R5): a final text still taken from the model
    output gets INCOMPLETE_TURN_MARKER too; one replaced by the input
    transcript does not depend on the collector.
    """
    decision = _decide_by_content(
        result_text, input_transcript_text, input_transcript_completed, audio_duration_sec,
        active_speech_sec=active_speech_sec, tail_active_speech_sec=tail_active_speech_sec,
    )
    if response_status in INCOMPLETE_RESPONSE_STATUSES:
        decision = _mark_incomplete_response(
            decision, response_status, input_transcript_text,
            input_transcript_completed, audio_duration_sec,
        )
    if output_ambiguous and not decision.replaced and decision.final_text:
        logger.warning(
            "[phone-decision] output parts ambiguous (reason=%s realtime_len=%d)",
            decision.reason, decision.realtime_len,
        )
        final_text = decision.final_text
        if INCOMPLETE_TURN_MARKER not in final_text:
            final_text = f"{final_text}\n{INCOMPLETE_TURN_MARKER}"
        decision = replace(
            decision, final_text=final_text,
            reason=f"{decision.reason}+ambiguous_output_marked",
        )
    return decision


def _mark_incomplete_response(
    decision: PhoneTranscriptionDecision,
    response_status: str,
    input_transcript_text: str,
    input_transcript_completed: bool,
    audio_duration_sec: float,
) -> PhoneTranscriptionDecision:
    logger.warning(
        "[phone-decision] upstream response status=%s (reason=%s realtime_len=%d input_len=%d)",
        response_status, decision.reason, decision.realtime_len, decision.input_len,
    )
    input_stripped = (input_transcript_text or "").strip()
    if (
        input_transcript_completed
        and len(input_stripped) > len(decision.final_text)
        and not is_runaway_repetition_transcript(input_stripped, audio_duration_sec)
    ):
        decision = replace(
            decision, final_text=input_stripped, replaced=True,
            reason="incomplete_response_rescued",
        )
    if not decision.final_text:
        raise HTTPException(status_code=502, detail=f"Upstream response {response_status}")
    if INCOMPLETE_TURN_MARKER in decision.final_text:
        return decision
    return replace(
        decision,
        final_text=f"{decision.final_text}\n{INCOMPLETE_TURN_MARKER}",
        reason=f"{decision.reason}+incomplete_marked",
    )


def _decide_by_content(
    result_text: str,
    input_transcript_text: str,
    input_transcript_completed: bool,
    audio_duration_sec: float,
    *,
    active_speech_sec: Optional[float] = None,
    tail_active_speech_sec: Optional[float] = None,
) -> PhoneTranscriptionDecision:
    """Decide the final iOS text and record why (for observability, task 0436).

    Precedence:
      * No marker (G2/M3): the model did not follow the transcription format.
        The phone is a batch lane, so it fails closed — replace only with a
        COMPLETED faithful transcript. Without one (empty / failed / timed-out /
        partial-delta-only) return neither the model answer nor a partial; the
        caller raises a controlled 503. A completed transcript that is a repeat
        loop is no faithful transcript either: 503 (W0322 R2).
      * Marker present, the output itself is a repeat loop: replace it with a
        completed, non-looping transcript, else 503 (W0322 R2).
      * Marker present, completed transcript is a repeat loop while the output
        is not (task 1193): keep the output.
      * Marker present, answer guard (G1/M4): completed transcript is non-empty
        and either the emitted output diverges below the similarity threshold or
        it carries too much material unaligned with the transcript — replace,
        unless the transcript looks truncated against a complete output
        (task 1193, T80: 566 realtime chars vs a 97-char transcript; the
        acoustic evidence decides what "long dictation" means) — keep.
      * Marker present, long-audio/short-output suspicious drift: replace when
        the completed transcript is materially longer.
      * Otherwise keep the (cleaned) realtime output.
    """
    marker_found, cleaned_output = _clean_phone_result_text(result_text)
    input_stripped = (input_transcript_text or "").strip()
    realtime_len = len(cleaned_output)
    input_len = len(input_stripped)

    input_loops = bool(input_stripped) and is_runaway_repetition_transcript(
        input_stripped, audio_duration_sec
    )
    usable_input = input_transcript_completed and bool(input_stripped) and not input_loops

    if not marker_found:
        if usable_input:
            return PhoneTranscriptionDecision(
                input_stripped, marker_found, True, "no_marker_replace",
                realtime_len, input_len, None,
            )
        reason = (
            "no_marker_input_repetition"
            if input_transcript_completed and input_loops
            else "no_marker_unavailable"
        )
        return PhoneTranscriptionDecision(
            "", marker_found, False, reason, realtime_len, input_len, None,
        )

    if cleaned_output and is_runaway_repetition_transcript(cleaned_output, audio_duration_sec):
        if usable_input:
            return PhoneTranscriptionDecision(
                input_stripped, marker_found, True, "realtime_repetition_rescued",
                realtime_len, input_len, None,
            )
        return PhoneTranscriptionDecision(
            "", marker_found, False, "realtime_repetition_unavailable",
            realtime_len, input_len, None,
        )

    if (
        input_transcript_completed
        and input_stripped
        and cleaned_output
        and input_loops
    ):
        return PhoneTranscriptionDecision(
            cleaned_output, marker_found, False, "keep_realtime_input_repetition",
            realtime_len, input_len, None,
        )

    similarity: Optional[float] = None
    if (
        ANSWER_GUARD_MIN_SIMILARITY > 0
        and input_transcript_completed
        and input_stripped
    ):
        novel_ratio: Optional[float] = None
        try:
            similarity = transcription_similarity_ratio(
                cleaned_output, input_stripped, SIMILARITY_HARD_CAP_CHARS
            )
        except Exception:
            similarity = None
        try:
            novel_ratio = emitted_novel_material_ratio(
                cleaned_output, input_stripped, SIMILARITY_HARD_CAP_CHARS
            )
        except Exception:
            novel_ratio = None
        low_similarity = similarity is not None and similarity < ANSWER_GUARD_MIN_SIMILARITY
        high_novelty = (
            novel_ratio is not None
            and ANSWER_GUARD_NOVEL_MATERIAL_RATIO > 0
            and novel_ratio > ANSWER_GUARD_NOVEL_MATERIAL_RATIO
        )
        if low_similarity or high_novelty:
            if is_likely_truncated_input_transcript(
                cleaned_output, input_stripped, audio_duration_sec,
                active_speech_sec=active_speech_sec,
                tail_active_speech_sec=tail_active_speech_sec,
            ):
                return PhoneTranscriptionDecision(
                    cleaned_output, marker_found, False, "keep_realtime_input_truncated",
                    realtime_len, input_len, similarity,
                )
            return PhoneTranscriptionDecision(
                input_stripped, marker_found, True, "answer_guard",
                realtime_len, input_len, similarity,
            )

    if (
        input_transcript_completed
        and _has_suspicious_marker_output(result_text, audio_duration_sec)
        and _has_material_input_replacement(input_stripped, cleaned_output)
    ):
        return PhoneTranscriptionDecision(
            input_stripped, marker_found, True, "suspicious_material",
            realtime_len, input_len, similarity,
        )

    return PhoneTranscriptionDecision(
        cleaned_output, marker_found, False, "keep_realtime",
        realtime_len, input_len, similarity,
    )


def choose_final_phone_transcription_text(
    result_text: str,
    input_transcript_text: str,
    input_transcript_completed: bool,
    audio_duration_sec: float,
) -> str:
    return decide_final_phone_transcription(
        result_text,
        input_transcript_text,
        input_transcript_completed,
        audio_duration_sec,
    ).final_text


@dataclass(frozen=True)
class PhoneTranscriptionEnvelope:
    """Final text plus the truncation verdict iOS (W0273) retries on.

    ``active_speech_sec`` / ``tail_active_speech_sec`` are the coverage
    measurement: per component max of raw and front-end enhanced PCM.
    """
    text: str
    suspected_truncated: bool
    truncation_reason: Optional[str]
    decision_reason: str
    audio_duration_sec: float
    active_speech_sec: float
    tail_active_speech_sec: float
    norm_char_count: int
    chars_per_audio_sec: float
    chars_per_active_sec: Optional[float]
    segment_count: int
    failed_segment_count: int

    @property
    def no_speech(self) -> bool:
        """0 chars with no truncation verdict (active speech < 0.6 s). When
        ``norm_char_count == 0`` exactly one of this and
        ``suspected_truncated`` holds; with text it is always False."""
        return self.norm_char_count == 0 and not self.suspected_truncated


@dataclass(frozen=True)
class _SegmentOutcome:
    text: str
    decision_reason: str
    upstream_incomplete: bool
    # The final text still depends on output parts kept without proof.
    output_ambiguous: bool = False


def _max_activity(
    raw: tuple[float, float], enhanced: Optional[tuple[float, float]]
) -> tuple[float, float]:
    if enhanced is None:
        return (float(raw[0]), float(raw[1]))
    return (max(raw[0], enhanced[0]), max(raw[1], enhanced[1]))


def _envelope(
    text: str,
    assessment: OutputTruncationAssessment,
    *,
    decision_reason: str,
    audio_duration_sec: float,
    coverage_activity: tuple[float, float],
    segment_count: int,
    failed_segment_count: int,
) -> PhoneTranscriptionEnvelope:
    return PhoneTranscriptionEnvelope(
        text=text,
        suspected_truncated=assessment.suspected_truncated,
        truncation_reason=assessment.truncation_reason,
        decision_reason=decision_reason,
        audio_duration_sec=audio_duration_sec,
        active_speech_sec=coverage_activity[0],
        tail_active_speech_sec=coverage_activity[1],
        norm_char_count=assessment.norm_char_count,
        chars_per_audio_sec=assessment.chars_per_audio_sec,
        chars_per_active_sec=assessment.chars_per_active_sec,
        segment_count=segment_count,
        failed_segment_count=failed_segment_count,
    )


def envelope_for_plain_text(text: str, pcm_data: bytes) -> PhoneTranscriptionEnvelope:
    """Default verdict for a plain-text result (a stubbed pool.transcribe):
    raw PCM activity, one segment, ``keep_realtime``."""
    audio_sec = len(pcm_data) / (TARGET_SAMPLE_RATE * 2)
    activity = measure_speech_activity_pcm16(pcm_data, TARGET_SAMPLE_RATE)
    assessment = assess_output_truncation(
        text, audio_sec, active_speech_sec=activity[0], tail_active_speech_sec=activity[1],
    )
    return _envelope(
        text, assessment, decision_reason="keep_realtime", audio_duration_sec=audio_sec,
        coverage_activity=_max_activity(activity, None), segment_count=1, failed_segment_count=0,
    )


class SessionPool:
    """Manages a single warm OpenAI Realtime WebSocket connection."""

    def __init__(self):
        self._client: OpenAIRealtimeAudioTextClient | None = None
        self._lock = asyncio.Lock()
        self._turn_count = 0
        self._created_at = 0.0
        self._api_key = ""

    def _should_rotate(self) -> bool:
        if self._client is None:
            return True
        if not self._client._is_ws_open():
            return True
        if self._turn_count >= MAX_SESSION_TURNS:
            return True
        if time.time() - self._created_at > MAX_SESSION_AGE_SEC:
            return True
        return False

    async def _close_client(self):
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
            self._turn_count = 0

    async def _ensure_client(self, api_key: str) -> OpenAIRealtimeAudioTextClient:
        if self._should_rotate():
            await self._close_client()
            client = OpenAIRealtimeAudioTextClient(api_key=api_key, model=OPENAI_REALTIME_MODEL)
            await client.connect(
                modalities=["text"],
                instructions=get_optimize_prompt(),
            )
            self._client = client
            self._turn_count = 0
            self._created_at = time.time()
            self._api_key = api_key
            logger.info("Session pool: new connection created")
        elif self._api_key != api_key:
            await self._close_client()
            return await self._ensure_client(api_key)
        else:
            try:
                await self._client.refresh_session(
                    modalities=["text"],
                    instructions=get_optimize_prompt(),
                )
                logger.info("Session pool: reused connection (refresh)")
            except Exception as e:
                logger.warning(f"Session refresh failed, rotating: {e}")
                await self._close_client()
                return await self._ensure_client(api_key)
        return self._client

    async def transcribe(self, pcm_data: bytes, api_key: str) -> str:
        """Transcribe one recording; text only (see transcribe_with_verdict)."""
        return (await self.transcribe_with_verdict(pcm_data, api_key)).text

    async def transcribe_with_verdict(
        self, pcm_data: bytes, api_key: str
    ) -> PhoneTranscriptionEnvelope:
        """Transcribe one recording and judge whether the text looks cut short.

        The PCM passes the speech front-end (high-pass + gated AGC) and, when
        longer than PHONE_SOFT_SEGMENT_SEC, is split at real pauses. Segments
        run in order under the pool lock; a failed segment becomes an explicit
        ``[片段N转录失败]`` line between the others, and only a recording whose
        every segment failed raises (the first segment's error).

        The truncation verdict (assess_output_truncation) uses the coverage
        activity: per component max of raw and enhanced PCM, because the gated
        AGC recovers far-field syllables the raw gate misses (T80: 19.0 s ->
        31.3 s) while steady noise and rumble stay near 0. The input-transcript
        guard keeps the raw measurement. With several segments a segment whose
        own verdict is truncated marks the whole recording, so a long complete
        segment cannot hide a cut-short one.
        """
        enhanced_pcm = pcm_data
        whole_activity: Optional[tuple[float, float]] = None
        if INPUT_AGC_ENABLED:
            enhanced_pcm, agc_stats = await asyncio.to_thread(
                preprocess_speech_pcm16, pcm_data, TARGET_SAMPLE_RATE
            )
            whole_activity = (agc_stats.active_speech_sec, agc_stats.tail_active_speech_sec)
            if agc_stats.applied:
                logger.info(
                    "[phone-agc] applied=True snr_p95=%.2f median_gain=%.2f max_gain=%.2f "
                    "out_rms=%.1f active_sec=%.1f tail_active_sec=%.1f",
                    agc_stats.snr_p95,
                    agc_stats.median_gain,
                    agc_stats.max_gain,
                    agc_stats.output_rms,
                    agc_stats.active_speech_sec,
                    agc_stats.tail_active_speech_sec,
                )
        segments = await asyncio.to_thread(
            split_pcm_on_pauses,
            enhanced_pcm,
            TARGET_SAMPLE_RATE,
            soft_max_sec=PHONE_SOFT_SEGMENT_SEC,
            hard_max_sec=PHONE_HARD_SEGMENT_SEC,
        )
        # Speech activity (truncated-transcript evidence) is measured on the
        # raw PCM of each segment; segments keep byte offsets, so the raw
        # slices line up with the enhanced ones.
        raw_segments: list[bytes] = []
        offset = 0
        for segment in segments:
            raw_segments.append(pcm_data[offset:offset + len(segment)])
            offset += len(segment)
        if len(segments) <= 1 and whole_activity is not None:
            activities = [whole_activity]
        else:
            activities = [
                await asyncio.to_thread(measure_speech_activity_pcm16, raw, TARGET_SAMPLE_RATE)
                for raw in raw_segments
            ]
        total_sec = len(pcm_data) / (TARGET_SAMPLE_RATE * 2)
        if len(segments) <= 1:
            whole_raw = activities[0]
        elif whole_activity is not None:
            whole_raw = whole_activity
        else:
            whole_raw = await asyncio.to_thread(
                measure_speech_activity_pcm16, pcm_data, TARGET_SAMPLE_RATE
            )
        whole_enh = (
            await asyncio.to_thread(measure_speech_activity_pcm16, enhanced_pcm, TARGET_SAMPLE_RATE)
            if INPUT_AGC_ENABLED else None
        )
        whole_cov = _max_activity(whole_raw, whole_enh)
        async with self._lock:
            if len(segments) <= 1:
                single = await self._transcribe_single_segment(
                    enhanced_pcm, api_key, activities[0]
                )
                assessment = assess_output_truncation(
                    single.text, total_sec,
                    active_speech_sec=whole_cov[0], tail_active_speech_sec=whole_cov[1],
                    upstream_incomplete=single.upstream_incomplete,
                    ambiguous_output=single.output_ambiguous,
                )
                return _envelope(
                    single.text, assessment, decision_reason=single.decision_reason,
                    audio_duration_sec=total_sec, coverage_activity=whole_cov,
                    segment_count=1, failed_segment_count=0,
                )
            logger.info(
                "[phone-segments] %d segments: %s",
                len(segments),
                ", ".join(f"{len(seg) / (TARGET_SAMPLE_RATE * 2):.1f}s" for seg in segments),
            )
            outputs: list[str] = []
            first_error: Optional[BaseException] = None
            succeeded = 0
            failed_count = 0
            any_upstream_incomplete = False
            any_ambiguous = False
            seg_reasons: list[str] = []
            first_bad_seg: Optional[OutputTruncationAssessment] = None
            for idx, (segment, raw_seg, activity) in enumerate(
                zip(segments, raw_segments, activities), start=1
            ):
                try:
                    seg = await self._transcribe_single_segment(segment, api_key, activity)
                    outputs.append(seg.text)
                    succeeded += 1
                    any_upstream_incomplete = any_upstream_incomplete or seg.upstream_incomplete
                    any_ambiguous = any_ambiguous or seg.output_ambiguous
                    seg_reasons.append(seg.decision_reason)
                    seg_enh = (
                        await asyncio.to_thread(
                            measure_speech_activity_pcm16, segment, TARGET_SAMPLE_RATE
                        )
                        if INPUT_AGC_ENABLED else None
                    )
                    seg_cov = _max_activity(activity, seg_enh)
                    seg_assessment = assess_output_truncation(
                        seg.text, len(segment) / (TARGET_SAMPLE_RATE * 2),
                        active_speech_sec=seg_cov[0], tail_active_speech_sec=seg_cov[1],
                        upstream_incomplete=seg.upstream_incomplete,
                        ambiguous_output=seg.output_ambiguous,
                    )
                    if seg_assessment.suspected_truncated and first_bad_seg is None:
                        first_bad_seg = seg_assessment
                        logger.warning(
                            "[phone-verdict] segment %d/%d suspected truncated: %s "
                            "(%d chars, %.3f chars/active s)",
                            idx, len(segments), seg_assessment.truncation_reason,
                            seg_assessment.norm_char_count,
                            seg_assessment.chars_per_active_sec or 0.0,
                        )
                except Exception as exc:
                    first_error = first_error or exc
                    logger.error(
                        "[phone-segments] segment %d/%d failed: %s",
                        idx, len(segments), getattr(exc, "detail", exc),
                    )
                    outputs.append(f"[片段{idx}转录失败]")
                    failed_count += 1
            if succeeded == 0:
                raise first_error
            joined = join_segment_texts(outputs)
            whole_assessment = assess_output_truncation(
                joined, total_sec,
                active_speech_sec=whole_cov[0], tail_active_speech_sec=whole_cov[1],
                upstream_incomplete=any_upstream_incomplete,
                failed_segment_count=failed_count,
                ambiguous_output=any_ambiguous,
            )
            if not whole_assessment.suspected_truncated and first_bad_seg is not None:
                whole_assessment = replace(
                    whole_assessment,
                    suspected_truncated=True,
                    truncation_reason=first_bad_seg.truncation_reason,
                )
            if failed_count > 0:
                decision_reason = "multi_segment_partial_failure"
            else:
                decision_reason = next(
                    (r for r in seg_reasons if r != "keep_realtime"), "keep_realtime"
                )
            return _envelope(
                joined, whole_assessment, decision_reason=decision_reason,
                audio_duration_sec=total_sec, coverage_activity=whole_cov,
                segment_count=len(segments), failed_segment_count=failed_count,
            )

    async def _transcribe_single_segment(
        self,
        pcm_data: bytes,
        api_key: str,
        speech_activity: Optional[tuple[float, float]] = None,
    ) -> _SegmentOutcome:
        """One provider turn; the caller holds self._lock. ``speech_activity``
        is ``(active_speech_sec, tail_active_speech_sec)`` of this audio."""
        client = await self._ensure_client(api_key)
        self._turn_count += 1

        output = _ResponseOutputCollector()
        response_status: Optional[str] = None
        input_transcript_delta_text = ""
        completed_input_transcript_text = ""
        input_transcript_completed = False
        response_complete = asyncio.Event()
        input_transcript_done = asyncio.Event()
        error_holder = [None]
        audio_duration_sec = len(pcm_data) / (TARGET_SAMPLE_RATE * 2)
        timeout_sec = transcription_timeout_sec(audio_duration_sec)

        async def on_text_delta(data):
            output.add_delta(data)

        async def on_text_done(data):
            output.add_done(data)

        async def on_response_done(data):
            nonlocal response_status
            response = data.get("response") if isinstance(data, dict) else None
            if isinstance(response, dict):
                response_status = response.get("status")
                output.add_response_output(response)
            response_complete.set()

        def is_this_turn_item(data) -> bool:
            # A reused session can still deliver input transcription for an
            # earlier request's item; only this turn's committed item counts.
            # The client binds committed_item_id when the ack is received,
            # before any event for that item is dispatched.
            item_id = data.get("item_id")
            if item_id is not None and item_id == client.committed_item_id:
                return True
            logger.debug("Ignoring input transcription event for another item")
            return False

        async def on_input_transcription_delta(data):
            nonlocal input_transcript_delta_text
            if not is_this_turn_item(data):
                return
            input_transcript_delta_text += extract_input_transcription(data)

        async def on_input_transcription_completed(data):
            nonlocal completed_input_transcript_text, input_transcript_completed
            if not is_this_turn_item(data):
                return
            completed_text, completed = _completed_input_transcript_for_replacement(data)
            if completed:
                completed_input_transcript_text = completed_text
                input_transcript_completed = True
            elif input_transcript_delta_text:
                logger.debug(
                    "Input transcription completed without extractable text after %d delta chars",
                    len(input_transcript_delta_text),
                )
            input_transcript_done.set()

        async def on_input_transcription_failed(data):
            if not is_this_turn_item(data):
                return
            if input_transcript_delta_text:
                logger.debug(
                    "Input transcription failed after %d delta chars",
                    len(input_transcript_delta_text),
                )
            input_transcript_done.set()

        async def on_error(data):
            error_msg = data.get("error", {}).get("message", "Unknown OpenAI error")
            error_holder[0] = error_msg
            response_complete.set()

        client.register_handler("response.text.delta", on_text_delta)
        client.register_handler("response.output_text.delta", on_text_delta)
        client.register_handler("response.text.done", on_text_done)
        client.register_handler("response.output_text.done", on_text_done)
        client.register_handler("response.done", on_response_done)
        client.register_handler(
            "conversation.item.input_audio_transcription.delta",
            on_input_transcription_delta,
        )
        client.register_handler(
            "conversation.item.input_audio_transcription.completed",
            on_input_transcription_completed,
        )
        client.register_handler(
            "conversation.item.input_audio_transcription.failed",
            on_input_transcription_failed,
        )
        client.register_handler("error", on_error)

        try:
            for i in range(0, len(pcm_data), CHUNK_SIZE_BYTES):
                chunk = pcm_data[i:i + CHUNK_SIZE_BYTES]
                await client.send_audio(chunk)

            await client.commit_audio()
            await client.start_response(instructions=get_optimize_prompt())

            logger.info(
                "Awaiting response: audio %.1fs, budget %.0fs",
                audio_duration_sec,
                timeout_sec,
            )
            loop = asyncio.get_running_loop()
            budget_started_at = loop.time()
            await asyncio.wait_for(response_complete.wait(), timeout=timeout_sec)
            # Wait briefly for the faithful input transcript when it may be
            # needed: no marker (G2 no-marker replacement), long-audio /
            # short-output drift, or the answer-similarity guard (G1). The
            # phone is batch mode, so a bounded wait has no streaming cost;
            # is_set() already-completed short-circuits the common case.
            result_text = output.text()
            marker_found_now, _ = _clean_phone_result_text(result_text)
            need_input_transcript = (
                not marker_found_now
                or _has_suspicious_marker_output(result_text, audio_duration_sec)
                or ANSWER_GUARD_MIN_SIMILARITY > 0
            )
            if need_input_transcript and not input_transcript_done.is_set():
                remaining_budget_sec = timeout_sec - (loop.time() - budget_started_at)
                grace_sec = completed_input_transcript_grace_sec(
                    audio_duration_sec, remaining_budget_sec
                )
                try:
                    await asyncio.wait_for(
                        input_transcript_done.wait(),
                        timeout=grace_sec,
                    )
                except asyncio.TimeoutError:
                    logger.info(
                        "Completed input transcript not received within %.1fs grace "
                        "(audio %.1fs, remaining budget %.1fs, %d delta chars)",
                        grace_sec,
                        audio_duration_sec,
                        remaining_budget_sec,
                        len(input_transcript_delta_text),
                    )

        except asyncio.TimeoutError:
            await self._close_client()
            logger.error(
                "Transcription timed out after %.0fs (audio %.1fs, %d chars streamed before cutoff)",
                timeout_sec,
                audio_duration_sec,
                len(output.text()),
            )
            raise HTTPException(
                status_code=504,
                detail=f"Transcription timed out ({timeout_sec:.0f}s)",
            )
        except CommitNotAcknowledgedError:
            # The provider never confirmed the commit: an upstream stall,
            # reported like the response timeout. The session is unusable
            # after a failed commit, so it is torn down (task 1082).
            await self._close_client()
            logger.error(
                "Commit not acknowledged (audio %.1fs); session closed",
                audio_duration_sec,
            )
            raise HTTPException(
                status_code=504,
                detail="Transcription timed out (commit not acknowledged)",
            )
        except Exception as e:
            await self._close_client()
            if error_holder[0]:
                # Provider rejected a turn event (e.g. the commit): keep the
                # 502 contract instead of surfacing it as a 500.
                raise HTTPException(
                    status_code=502, detail=f"OpenAI error: {error_holder[0]}"
                ) from e
            raise

        if error_holder[0]:
            await self._close_client()
            raise HTTPException(status_code=502, detail=f"OpenAI error: {error_holder[0]}")

        try:
            await client.clear_audio_buffer()
        except Exception:
            pass

        # Only a COMPLETED transcript may replace the output (task 0436 M3);
        # partial deltas are never used as the final text. The bounded diffs
        # run off the event loop (task 0436 M5) so a large input can't stall
        # the phone backend's other requests.
        result_text = output.text()
        try:
            decision = await asyncio.to_thread(
                functools.partial(
                    decide_final_phone_transcription,
                    result_text,
                    completed_input_transcript_text,
                    input_transcript_completed,
                    audio_duration_sec,
                    response_status,
                    active_speech_sec=speech_activity[0] if speech_activity else None,
                    tail_active_speech_sec=speech_activity[1] if speech_activity else None,
                    output_ambiguous=output.ambiguous,
                )
            )
        except HTTPException:
            await self._close_client()
            raise
        _, final_sha1 = answer_guard_fingerprint(decision.final_text)
        logger.info(
            "[phone-decision] marker_found=%s replaced=%s reason=%s "
            "realtime_len=%d input_len=%d similarity=%s response_status=%s final_sha1=%s",
            decision.marker_found,
            decision.replaced,
            decision.reason,
            decision.realtime_len,
            decision.input_len,
            f"{decision.similarity:.3f}" if decision.similarity is not None else "n/a",
            response_status or "n/a",
            final_sha1,
        )
        if decision.reason in FAIL_CLOSED_DECISION_REASONS:
            # Batch lane fails closed (task 0436 M3): the model did not
            # transcribe (or looped) and there is no faithful transcript to
            # fall back to (none, partial only, or itself a loop: W0322 R2).
            # Return a controlled, user-facing error the iOS client can show
            # verbatim rather than a wrong answer or a truncated partial.
            await self._close_client()
            raise HTTPException(
                status_code=503,
                detail="转录暂时不可用，请重试",
            )
        return _SegmentOutcome(
            text=decision.final_text,
            decision_reason=decision.reason,
            upstream_incomplete=response_status in INCOMPLETE_RESPONSE_STATUSES,
            output_ambiguous=output.ambiguous and not decision.replaced,
        )


_pool = SessionPool()


async def health():
    has_key = bool(cached_api_key())
    pool_active = _pool._client is not None and _pool._client._is_ws_open()
    return {
        "status": "ok",
        "service": "micwave_phone",
        "port": PORT,
        "api_key_configured": has_key,
        "session_active": pool_active,
        "turn_count": _pool._turn_count,
        "audio_cache_enabled": bool(getattr(_audio_cache, "_enabled", False)),
    }


async def transcribe(file: UploadFile = File(...)):
    t0 = time.time()
    # Off the event loop: a cold fetch must not stall /health (invariant I4).
    api_key = await asyncio.to_thread(load_api_key)
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio conversion failed: Empty audio file")

    filename = file.filename or "audio.m4a"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "m4a"
    logger.info(f"Transcribe: {filename} ({len(audio_bytes)} bytes)")

    if ext in ("pcm", "raw"):
        pcm_data = audio_bytes
    else:
        try:
            pcm_data = await asyncio.to_thread(convert_audio_to_pcm16, audio_bytes, source_format=ext)
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise HTTPException(status_code=400, detail=f"Audio conversion failed: {e}")

    if not pcm_data:
        raise HTTPException(status_code=400, detail="Audio conversion produced empty output")

    duration_s = len(pcm_data) / (TARGET_SAMPLE_RATE * 2)
    logger.info(f"PCM ready: {len(pcm_data)} bytes ({duration_s:.1f}s)")

    tid = next(_turn_ids)
    outcome = "error"
    audio_file = None
    async with _audio_cache_lock:
        _audio_cache.start_turn(tid)
        _audio_cache.accumulate(pcm_data)
        try:
            if "transcribe" in vars(_pool):
                # pool.transcribe replaced on the instance (tests): it may
                # return plain text, so derive the default verdict here.
                stub_out = await _pool.transcribe(pcm_data, api_key)
                if isinstance(stub_out, PhoneTranscriptionEnvelope):
                    envelope = stub_out
                else:
                    envelope = await asyncio.to_thread(envelope_for_plain_text, stub_out, pcm_data)
            else:
                envelope = await _pool.transcribe_with_verdict(pcm_data, api_key)
            text = envelope.text
            outcome = "completed"
        except HTTPException as exc:
            if exc.status_code == 504:
                outcome = "timeout"
            elif exc.status_code == 502:
                outcome = "openai_error"
            elif exc.status_code == 503:
                outcome = "unavailable"
            raise
        except Exception:
            raise
        finally:
            try:
                audio_file = _audio_cache.enqueue_turn_file(tid, outcome, TARGET_SAMPLE_RATE, channels=1)
            except Exception as e:
                logger.error(f"Audio cache enqueue failed: {e}", exc_info=True)

    _submit_transcript_record(text, duration_s, audio_file, tid)
    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        "Done: %d chars in %.0fms suspected_truncated=%s truncation_reason=%s",
        len(text), elapsed_ms, envelope.suspected_truncated, envelope.truncation_reason,
    )
    return JSONResponse({
        "text": envelope.text,
        "status": "ok",
        "suspected_truncated": envelope.suspected_truncated,
        "truncation_reason": envelope.truncation_reason,
        "decision_reason": envelope.decision_reason,
        "audio_duration_sec": round(envelope.audio_duration_sec, 3),
        "active_speech_sec": round(envelope.active_speech_sec, 3),
        "tail_active_speech_sec": round(envelope.tail_active_speech_sec, 3),
        "norm_char_count": envelope.norm_char_count,
        "chars_per_audio_sec": envelope.chars_per_audio_sec,
        "chars_per_active_sec": envelope.chars_per_active_sec,
        "segment_count": envelope.segment_count,
        "failed_segment_count": envelope.failed_segment_count,
        "truncated": envelope.suspected_truncated,
        "no_speech": envelope.no_speech,
    })


def create_phone_app() -> FastAPI:
    """Create a FastAPI session-pooled phone batch transcription application."""
    phone_app = FastAPI(
        title="MicWave Phone",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    phone_app.add_api_route("/health", health, methods=["GET"])
    phone_app.add_api_route("/transcribe", transcribe, methods=["POST"])
    return phone_app


app = create_phone_app()


def main() -> None:
    host = os.getenv("MICWAVE_PHONE_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
