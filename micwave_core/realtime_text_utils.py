"""Utilities for parsing transcript content emitted by realtime providers."""

import difflib
import functools
import hashlib
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple

from . import vocabulary

logger = logging.getLogger(__name__)


def parse_ratio_env(name: str, default: float) -> float:
    """Parse a 0..1 ratio env var with a shared, safe policy (task 0436 N1).

    Reject a non-numeric, non-finite, or out-of-[0,1] value with a warning and
    fall back to ``default``. Never raises at import time. Used by both the
    streaming server and the phone backend so answer-guard thresholds share one
    parsing semantic.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r (not a number); using %s", name, raw, default)
        return default
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        logger.warning(
            "Invalid %s=%r (must be a finite ratio in [0,1]); using %s",
            name, raw, default,
        )
        return default
    return value


def _parse_positive_int_env(name: str, default: int) -> int:
    """Parse a positive-int env var; fall back to default on bad input."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r (not an integer); using %s", name, raw, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r (must be > 0); using %s", name, raw, default)
        return default
    return value


# Hard cap on the normalized length fed to SequenceMatcher so the answer guard
# never runs an unbounded near-O(n^2) diff on the event loop (task 0436 M5).
# Inputs longer than the cap are truncated to the cap before comparison.
SIMILARITY_HARD_CAP_CHARS = _parse_positive_int_env(
    "BRAINWAVE_ANSWER_GUARD_SIMILARITY_MAX_CHARS", 6000
)


NO_SPEECH_PLACEHOLDER = "（没有识别到输入）"
NO_SPEECH_PLACEHOLDER_MAX_INNER_CHARS = 20
_NO_SPEECH_MAX_LEADING_WHITESPACE = 20
_NO_SPEECH_OPEN_PARENS = frozenset("（(")
_NO_SPEECH_CLOSE_PARENS = frozenset("）)")
_NO_SPEECH_TRAILING_PERIODS = frozenset("。．.")
_NO_SPEECH_PLACEHOLDER_BODIES = frozenset(
    f"{prefix}{any_word}{effective}{voice}输入"
    for prefix in ("没有识别到", "未识别到")
    for any_word in ("", "任何")
    for effective in ("", "有效")
    for voice in ("", "语音")
)
_NO_SPEECH_BODY_RE = "|".join(
    re.escape(body)
    for body in sorted(_NO_SPEECH_PLACEHOLDER_BODIES, key=len, reverse=True)
)
NO_SPEECH_PLACEHOLDER_PATTERN = re.compile(
    rf"^\s*[（(](?P<body>{_NO_SPEECH_BODY_RE})[）)]"
)


def strip_no_speech_placeholder_prefix(text: str) -> Tuple[bool, str]:
    """Strip one diagnostic no-speech prefix and its trailing period/space.

    The match is deliberately anchored at the transcript body's first
    non-whitespace character. It never scans or rewrites later transcript text.
    """
    if not text:
        return False, text
    match = NO_SPEECH_PLACEHOLDER_PATTERN.match(text)
    if match is None:
        return False, text
    end = match.end()
    while end < len(text):
        char = text[end]
        if char.isspace() or char in _NO_SPEECH_TRAILING_PERIODS:
            end += 1
            continue
        break
    return True, text[end:]


def is_no_speech_placeholder_only(text: str) -> bool:
    """Return True when text contains only a no-speech diagnostic variant."""
    matched, remaining = strip_no_speech_placeholder_prefix(text)
    return matched and not remaining.strip()


class StreamingNoSpeechPrefixGuard:
    """Bounded, one-shot filter for a no-speech prefix split across deltas.

    The marker owns the response's leading formatting, so extra whitespace at
    the start of the transcript body is discarded. Normal text passes through
    from its first non-whitespace character. Only a leading parenthesis enters
    the bounded screening buffer. Once a diagnostic prefix matches, the prefix
    plus following periods/whitespace are discarded; all later body text passes
    through unchanged.
    """

    def __init__(
        self,
        max_inner_chars: int = NO_SPEECH_PLACEHOLDER_MAX_INNER_CHARS,
        max_leading_whitespace: int = _NO_SPEECH_MAX_LEADING_WHITESPACE,
    ):
        self._max_inner_chars = max_inner_chars
        self._max_leading_whitespace = max_leading_whitespace
        self.reset()

    def reset(self) -> None:
        self._buffer = ""
        self._done = False
        self._discard_trailing = False

    def _release_buffer(self) -> str:
        released = self._buffer.lstrip()
        self._buffer = ""
        self._done = True
        return released

    def _consume_matched_suffix(self, text: str) -> str:
        index = 0
        while index < len(text):
            char = text[index]
            if char.isspace() or char in _NO_SPEECH_TRAILING_PERIODS:
                index += 1
                continue
            break
        if index == len(text):
            return ""
        self._discard_trailing = False
        self._done = True
        return text[index:]

    def push(self, delta: str) -> str:
        if not delta:
            return ""
        if self._done:
            return delta
        if self._discard_trailing:
            return self._consume_matched_suffix(delta)

        self._buffer += delta
        first_content = next(
            (index for index, char in enumerate(self._buffer) if not char.isspace()),
            None,
        )
        if first_content is None:
            if len(self._buffer) > self._max_leading_whitespace:
                self._buffer = ""
            return ""
        if first_content > self._max_leading_whitespace:
            return self._release_buffer()
        if self._buffer[first_content] not in _NO_SPEECH_OPEN_PARENS:
            return self._release_buffer()

        inner_start = first_content + 1
        close_index = next(
            (
                index
                for index in range(inner_start, len(self._buffer))
                if self._buffer[index] in _NO_SPEECH_CLOSE_PARENS
            ),
            None,
        )
        if close_index is None:
            inner = self._buffer[inner_start:]
            if (
                len(inner) > self._max_inner_chars
                or not any(body.startswith(inner) for body in _NO_SPEECH_PLACEHOLDER_BODIES)
            ):
                return self._release_buffer()
            return ""

        inner = self._buffer[inner_start:close_index]
        if (
            len(inner) > self._max_inner_chars
            or inner not in _NO_SPEECH_PLACEHOLDER_BODIES
        ):
            return self._release_buffer()

        remainder = self._buffer[close_index + 1 :]
        self._buffer = ""
        self._discard_trailing = True
        return self._consume_matched_suffix(remainder)

    def flush(self) -> str:
        """Finish screening at response end without dropping partial text."""
        if self._done:
            return ""
        if self._discard_trailing:
            self._discard_trailing = False
            self._done = True
            return ""
        if not self._buffer.strip():
            self._buffer = ""
            self._done = True
            return ""
        return self._release_buffer()


_HOMONYM_RE = re.compile(
    r'(?:克劳德|克劳得|克劳特|克勞德|克勞得|克勞特|云|cloud|cloude|clouds|cloudy|claud|claude)'
    r'[\s\-]*'
    r'code',
    re.IGNORECASE,
)

# Hold back this many trailing chars in streaming so a pattern split across
# chunks ("Cloud" then " Code") still gets matched. Longest pattern is
# "克勞特 cod" (9 chars) or "cloude code" (11), 16 leaves margin.
_HOMONYM_HOLD_TAIL = 16


def apply_homonym_correction(text: str, substitutions=vocabulary.STREAMING_SUBSTITUTIONS) -> str:
    """Replace 'cloud/云/克劳德 + code' homonyms with 'Claude Code' (case-insensitive), then speaker vocabulary."""
    if not text:
        return text
    text = _HOMONYM_RE.sub('Claude Code', text)
    for pattern, replacement in substitutions:
        text = pattern.sub(replacement, text)
    return text


class StreamingHomonymCorrector:
    """Streaming-aware homonym corrector.

    Buffers a short tail across deltas so cross-chunk patterns still match.
    Caller pushes raw delta and gets back the safe-to-emit corrected portion.
    Call flush() at turn end to drain remaining buffer.
    """

    def __init__(self, hold_tail: int = _HOMONYM_HOLD_TAIL):
        self._buffer = ""
        self._emitted = ""
        self._hold_tail = hold_tail

    def push(self, delta: str) -> str:
        if not delta:
            return ""
        self._buffer += delta
        corrected = apply_homonym_correction(self._buffer)
        if len(corrected) <= self._hold_tail:
            self._buffer = corrected
            return ""
        emit = corrected[: -self._hold_tail]
        self._buffer = corrected[-self._hold_tail :]
        self._emitted += emit
        return emit

    def flush(self) -> str:
        emit = apply_homonym_correction(self._buffer)
        self._buffer = ""
        self._emitted += emit
        return emit

    def peek_full_text(self) -> str:
        """Read-only view of the full corrected text produced so far.

        Returns everything already emitted plus the corrected held tail that has
        not been sent yet, without consuming the buffer. The answer guard uses
        this so a short sentence still sitting in the hold-tail buffer is
        compared at its true length instead of the artificially truncated
        emitted-only text (task 0436 M6).
        """
        return self._emitted + apply_homonym_correction(self._buffer)

    def reset(self) -> None:
        # Drop the held tail and emitted history without emitting. Used when the
        # consumer is about to overwrite the entire stream (e.g. fallback emit
        # with isNewResponse=True), so a later flush() can't append leftover
        # characters and peek_full_text() no longer reflects the replaced text.
        self._buffer = ""
        self._emitted = ""


def extract_text_after_marker(text: str, marker_prefix: str) -> Tuple[bool, str]:
    """Return (found, content_after_marker) for both strict and relaxed prefix matches."""
    if not text or not marker_prefix:
        return False, ""

    marker_prefix_no_newline = marker_prefix.rstrip("\n")

    if text.startswith(marker_prefix):
        return True, text[len(marker_prefix):]

    if text.startswith(marker_prefix_no_newline):
        return True, text[len(marker_prefix_no_newline):].lstrip("\n")

    marker_index = text.find(marker_prefix)
    if marker_index != -1:
        return True, text[marker_index + len(marker_prefix):]

    marker_index = text.find(marker_prefix_no_newline)
    if marker_index != -1:
        return True, text[marker_index + len(marker_prefix_no_newline):].lstrip("\n")

    return False, ""


# ── Answer-similarity guard (task 0436 G1) ──────────────────────────────────
# The realtime "response" channel is a conversational Q&A channel; even with a
# marker prefix the model can answer instead of transcribe. gpt-4o-transcribe's
# input transcription is a faithful ground truth for the same audio. When the
# emitted (marker-following) text diverges materially from that transcript, the
# model answered rather than transcribed. We measure divergence with a
# normalized SequenceMatcher ratio that is friendly to CJK/English code-mixing
# and forgives light punctuation/whitespace polishing.


# Technical punctuation kept even though it is Unicode P* (task 0436 M4). ``#``
# is category Po, so dropping all P* folded ``C#``/``F#`` onto ``C``/``F`` and
# hid a distinct token. Keep the minimal set that carries technical meaning.
_KEPT_TECHNICAL_PUNCTUATION = frozenset("#")


def normalize_for_similarity(text: str) -> str:
    """NFKC-normalize, drop whitespace + punctuation, keep symbols, lowercase.

    Light transcription polish (punctuation, casing, spacing) collapses away so
    a faithful transcript vs its lightly-edited realtime echo scores near 1.0,
    while a genuine answer scores low. Unicode symbol (``S*``) characters are
    kept so technical tokens like ``C++``, ``$``, and math/currency symbols are
    NOT folded together (task 0436 M4). A minimal set of technical punctuation
    (``#`` — Unicode Po) is also kept so ``C#``/``F#`` stay distinct from
    ``C``/``F``; otherwise only whitespace and ``P*`` punctuation are dropped.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    kept = []
    for ch in normalized:
        if ch.isspace():
            continue
        category = unicodedata.category(ch)
        if (
            category
            and category[0] == "P"
            and ch not in _KEPT_TECHNICAL_PUNCTUATION
        ):  # punctuation only; keep symbols and technical punctuation
            continue
        kept.append(ch.lower())
    return "".join(kept)


def _capped(text: str, max_chars: int, label: str) -> str:
    """Bound a normalized string to max_chars by sampling its head and tail.

    A plain prefix truncation (task 0436 M5) hid a long answer appended after a
    faithful prefix: the shared prefix alone scored a perfect match while the
    divergent suffix was never compared (task 0436 M3-R4). Sampling head cap/2 +
    tail cap/2 keeps the same runtime bound (the diff still sees <= max_chars
    chars) while exposing suffix divergence. A normal long transcript, whose
    head and tail both align with the transcript, still scores near 1.0.
    """
    if max_chars and len(text) > max_chars:
        logger.warning(
            "[answer-guard] %s normalized length %d exceeds cap %d; sampling "
            "head+tail for bounded similarity", label, len(text), max_chars,
        )
        head = max_chars // 2
        tail = max_chars - head
        return text[:head] + text[-tail:]
    return text


def transcription_similarity_ratio(
    emitted_text: str,
    transcript_text: str,
    max_chars: int = SIMILARITY_HARD_CAP_CHARS,
) -> float:
    """Return a 0..1 similarity between emitted output and faithful transcript.

    Both sides are normalized first. Two empty strings are treated as identical
    (1.0); one-sided empty is fully dissimilar (0.0). Normalized inputs longer
    than ``max_chars`` are truncated before the SequenceMatcher runs so the diff
    stays bounded and never blocks the event loop (task 0436 M5).
    """
    left = normalize_for_similarity(emitted_text)
    right = normalize_for_similarity(transcript_text)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    left = _capped(left, max_chars, "emitted")
    right = _capped(right, max_chars, "transcript")
    return difflib.SequenceMatcher(None, left, right).ratio()


def emitted_novel_material_ratio(
    emitted_text: str,
    transcript_text: str,
    max_chars: int = SIMILARITY_HARD_CAP_CHARS,
) -> float:
    """Fraction of the emitted text NOT aligned with the faithful transcript.

    A secondary answer-guard signal (task 0436 M4): when the model restates the
    question and then answers it, similarity can stay above threshold, but the
    emitted text carries a lot of material that does not align with the input
    transcript. Returns 0.0 when the emitted side is empty (nothing novel) and
    is bounded by the same length cap as the similarity ratio.
    """
    left = normalize_for_similarity(emitted_text)
    right = normalize_for_similarity(transcript_text)
    if not left:
        return 0.0
    if not right:
        return 1.0
    left = _capped(left, max_chars, "emitted")
    right = _capped(right, max_chars, "transcript")
    matcher = difflib.SequenceMatcher(None, left, right)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return max(0.0, 1.0 - matched / len(left))


def answer_guard_fingerprint(text: str) -> Tuple[int, str]:
    """Return (char_length, sha1[:8]) for incident logging without plaintext."""
    material = text or ""
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:8]
    return len(material), digest


# ── Truncated / runaway input-transcript guards (task 1193, W0322) ──────────
# The answer guard trusts the completed input transcript as ground truth. On a
# quiet far-field phone recording with long pauses (T80: 91.6 s) that
# transcript stalled after two sentences (97 chars) while the realtime output
# carried the whole recording (566 chars); similarity 0.218 then replaced the
# complete text with the truncated one. The helpers below let both lanes
# recognise a transcript that is truncated or looping instead of faithful.

TRUNCATED_INPUT_MIN_AUDIO_SEC = 20.0
TRUNCATED_INPUT_MAX_CHARS_PER_SEC = 1.8
TRUNCATED_EMITTED_MIN_RATIO = 2.2
TRUNCATED_EMITTED_MIN_EXTRA_CHARS = 60
TRUNCATED_EMITTED_MAX_CHARS_PER_SEC = 12.0
TRUNCATED_EMITTED_MAX_CHARS_FLOOR = 240
TRUNCATED_PREFIX_MIN_SIMILARITY = 0.45
TRUNCATED_INPUT_MAX_NOVEL_RATIO = 0.38
# Acoustic evidence (W0322 R2). A 30 s recording that holds a 3 s question and
# 27 s of silence must not count as 30 s of dictation, or a restate-then-answer
# reply looks like a complete transcript of a truncated one.
# Only as gates: frame-level activity under-counts far-field speech (T80
# measured active_ratio 0.229, about 21 s of 91.6 s), so rates per active
# second would reject the very case this guard exists for.
TRUNCATED_MIN_ACTIVE_SPEECH_SEC = 15.0
TRUNCATED_MIN_TAIL_ACTIVE_SPEECH_SEC = 4.0
# Without acoustic evidence the text alone must look like a long dictation
# that stalled after at least one full sentence.
TRUNCATED_TEXT_ONLY_MIN_AUDIO_SEC = 30.0
TRUNCATED_TEXT_ONLY_MIN_INPUT_CHARS = 24
RUNAWAY_MAX_CHARS_PER_SEC = 18.0
RUNAWAY_MAX_CHARS_FLOOR = 400
RUNAWAY_NGRAM = 6
RUNAWAY_MIN_CHARS_FOR_NGRAM = 120
RUNAWAY_MIN_UNIQUE_NGRAM_RATIO = 0.20


def is_likely_truncated_input_transcript(
    emitted_text: str,
    input_transcript: str,
    audio_duration_sec: float,
    *,
    active_speech_sec: Optional[float] = None,
    tail_active_speech_sec: Optional[float] = None,
) -> bool:
    """Whether the input transcript looks cut short rather than faithful.

    All must hold: the recording is long (>= 20 s); there is evidence of long
    speech (below); the transcript is sparse (< 1.8 normalized chars per
    second); the emitted text is much longer (>= max(2.2x, +60 chars)) yet
    within human speech rate (<= 12 chars/s, floor 240), which rules out a
    synthetic long answer; and the emitted text covers the same speech: its
    head matches the transcript (similarity >= 0.45) or the transcript is
    mostly contained in it (novel ratio <= 0.38).

    Evidence of long speech (W0322 R2): with ``active_speech_sec`` (frames
    above the speech gate) at least 15 s of it and, when given,
    ``tail_active_speech_sec`` >= 4 s in the recording's second half, so a
    short question before or after a long silence never qualifies. Without
    acoustic evidence the recording must be >= 30 s and the transcript >= 24
    normalized chars (a stalled dictation has at least one full sentence; a
    one-line question has not).
    """
    if audio_duration_sec < TRUNCATED_INPUT_MIN_AUDIO_SEC:
        return False
    in_norm = normalize_for_similarity(input_transcript)
    em_norm = normalize_for_similarity(emitted_text)
    if not in_norm or not em_norm:
        return False
    if active_speech_sec is not None:
        if active_speech_sec < TRUNCATED_MIN_ACTIVE_SPEECH_SEC:
            return False
        if (
            tail_active_speech_sec is not None
            and tail_active_speech_sec < TRUNCATED_MIN_TAIL_ACTIVE_SPEECH_SEC
        ):
            return False
    elif (
        audio_duration_sec < TRUNCATED_TEXT_ONLY_MIN_AUDIO_SEC
        or len(in_norm) < TRUNCATED_TEXT_ONLY_MIN_INPUT_CHARS
    ):
        return False
    if len(in_norm) >= audio_duration_sec * TRUNCATED_INPUT_MAX_CHARS_PER_SEC:
        return False
    min_emitted = max(
        int(len(in_norm) * TRUNCATED_EMITTED_MIN_RATIO),
        len(in_norm) + TRUNCATED_EMITTED_MIN_EXTRA_CHARS,
    )
    if len(em_norm) < min_emitted:
        return False
    max_emitted = max(
        TRUNCATED_EMITTED_MAX_CHARS_FLOOR,
        int(audio_duration_sec * TRUNCATED_EMITTED_MAX_CHARS_PER_SEC),
    )
    if len(em_norm) > max_emitted:
        return False
    in_len = len(input_transcript)
    prefix_window = emitted_text[: max(in_len + 40, int(in_len * 1.5))]
    if transcription_similarity_ratio(prefix_window, input_transcript) >= TRUNCATED_PREFIX_MIN_SIMILARITY:
        return True
    return emitted_novel_material_ratio(input_transcript, emitted_text) <= TRUNCATED_INPUT_MAX_NOVEL_RATIO


def is_runaway_repetition_transcript(text: str, audio_duration_sec: float) -> bool:
    """Whether a transcript is a repeat loop (gpt-4o-transcribe on long noise).

    True when it is longer than any speech rate allows (> 18 normalized chars
    per second, floor 400) or, from 120 chars on, fewer than 20 % of its
    6-grams are distinct.
    """
    norm = normalize_for_similarity(text)
    limit = max(RUNAWAY_MAX_CHARS_FLOOR, int(max(audio_duration_sec, 1.0) * RUNAWAY_MAX_CHARS_PER_SEC))
    if len(norm) > limit:
        return True
    if len(norm) < RUNAWAY_MIN_CHARS_FOR_NGRAM:
        return False
    ngrams = [norm[i:i + RUNAWAY_NGRAM] for i in range(len(norm) - RUNAWAY_NGRAM + 1)]
    return len(set(ngrams)) / len(ngrams) < RUNAWAY_MIN_UNIQUE_NGRAM_RATIO


def scaled_suspicious_output_chars(
    audio_duration_sec: float,
    base_chars: int = 120,
    chars_per_sec: float = 2.2,
) -> int:
    """Short-output threshold for the long-audio drift check, grown with the
    recording: 20 s -> 120, 91.6 s -> 201, 200 s -> 440 (defaults)."""
    return max(base_chars, int(max(0.0, audio_duration_sec) * chars_per_sec))


# Appended on its own line when the provider ends a response as
# incomplete/failed/cancelled and nothing complete could replace it, so a
# cut-short transcript is never shown as the whole recording (task 1193).
INCOMPLETE_TURN_MARKER = "[转录未完整]"


# ── Output truncation verdict for client retry (task 1193, W0322 m7) ────────
# iOS (W0273) retries a recording automatically when the backend says the
# transcript looks cut short, and reuses this verdict instead of a second
# client-side heuristic. Diagnostic markers never count as transcript text.

_SEGMENT_FAILED_MARKER_RE = re.compile(r"\[片段\d+转录失败\]")

TRUNC_EMPTY_MIN_ACTIVE_SEC = 0.6
TRUNC_EMPTY_TEXT_ONLY_MIN_AUDIO_SEC = 30.0
# Long recording, sparse coverage. Calibrated on T80 before the fix (97 chars,
# 91.6 s, 31.3 s active, 11.5 s tail: 0.98 chars/s, 2.87 chars/active s) and
# W0273-m14 (15 s speech + 120 s silence + 11.5 s speech, second half lost:
# 0.51 / 3.23 -> truncated; complete 137 chars: 0.94 / 5.89 -> not).
TRUNC_COVERAGE_MIN_AUDIO_SEC = 20.0
TRUNC_COVERAGE_MIN_ACTIVE_SEC = 12.0
TRUNC_COVERAGE_MIN_TAIL_ACTIVE_SEC = 4.0
TRUNC_COVERAGE_MAX_CHARS_PER_SEC = 1.8
TRUNC_COVERAGE_MAX_CHARS_PER_ACTIVE_SEC = 3.4
# 10-20 s continuous dictation that lost its second half.
TRUNC_ACTIVE_MIN_AUDIO_SEC = 10.0
TRUNC_ACTIVE_MIN_ACTIVE_SEC = 8.0
TRUNC_ACTIVE_MIN_TAIL_ACTIVE_SEC = 3.0
TRUNC_ACTIVE_MAX_CHARS_PER_ACTIVE_SEC = 2.2


def strip_diagnostic_markers(text: str) -> str:
    """Remove INCOMPLETE_TURN_MARKER and ``[片段N转录失败]`` placeholders."""
    if not text:
        return ""
    return _SEGMENT_FAILED_MARKER_RE.sub("", text.replace(INCOMPLETE_TURN_MARKER, "")).strip()


@dataclass(frozen=True)
class OutputTruncationAssessment:
    suspected_truncated: bool
    truncation_reason: Optional[str]
    norm_char_count: int
    chars_per_audio_sec: float
    chars_per_active_sec: Optional[float]


def _clean_activity_sec(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return max(0.0, value)


def _truncation_verdict(
    norm_chars: int,
    audio_sec: float,
    active_sec: Optional[float],
    tail_sec: Optional[float],
    upstream_incomplete: bool,
    failed_segment_count: int,
    ambiguous_output: bool = False,
) -> Tuple[bool, Optional[str]]:
    if failed_segment_count > 0:
        return True, "segment_failed"
    if upstream_incomplete:
        return True, "upstream_incomplete"
    if ambiguous_output:
        return True, "ambiguous_output"
    if active_sec is not None:
        if norm_chars == 0:
            if active_sec >= TRUNC_EMPTY_MIN_ACTIVE_SEC:
                return True, "empty_output_for_active_speech"
            # Silence, steady noise or handling rumble: never retry.
            return False, None
        if (
            audio_sec >= TRUNC_COVERAGE_MIN_AUDIO_SEC
            and active_sec >= TRUNC_COVERAGE_MIN_ACTIVE_SEC
            and (tail_sec is None or tail_sec >= TRUNC_COVERAGE_MIN_TAIL_ACTIVE_SEC)
            and norm_chars / audio_sec < TRUNC_COVERAGE_MAX_CHARS_PER_SEC
            and norm_chars / active_sec < TRUNC_COVERAGE_MAX_CHARS_PER_ACTIVE_SEC
        ):
            return True, "sparse_output_for_speech_coverage"
        if (
            audio_sec >= TRUNC_ACTIVE_MIN_AUDIO_SEC
            and active_sec >= TRUNC_ACTIVE_MIN_ACTIVE_SEC
            and (tail_sec is None or tail_sec >= TRUNC_ACTIVE_MIN_TAIL_ACTIVE_SEC)
            and norm_chars / active_sec < TRUNC_ACTIVE_MAX_CHARS_PER_ACTIVE_SEC
        ):
            return True, "sparse_output_for_active_speech"
        return False, None
    if audio_sec >= TRUNC_EMPTY_TEXT_ONLY_MIN_AUDIO_SEC and norm_chars == 0:
        return True, "empty_output_long_audio"
    return False, None


def assess_output_truncation(
    text: str,
    audio_duration_sec: float,
    *,
    active_speech_sec: Optional[float] = None,
    tail_active_speech_sec: Optional[float] = None,
    upstream_incomplete: bool = False,
    failed_segment_count: int = 0,
    ambiguous_output: bool = False,
) -> OutputTruncationAssessment:
    """Whether a final transcript looks cut short relative to its recording.

    Precedence: a failed segment, an abnormal upstream end, output whose part
    order or duplication could not be proven (W0322 R5), then (with
    acoustic evidence) speech with no text (>= 0.6 s active), sparse coverage
    of a long recording, sparse text per active second; without acoustic
    evidence only an empty transcript of >= 30 s audio. Diagnostic markers are
    stripped before counting normalized chars.
    """
    try:
        audio_sec = float(audio_duration_sec)
    except (TypeError, ValueError):
        audio_sec = 0.0
    if not math.isfinite(audio_sec) or audio_sec < 0:
        audio_sec = 0.0
    active_sec = _clean_activity_sec(active_speech_sec)
    tail_sec = _clean_activity_sec(tail_active_speech_sec)
    norm_chars = len(normalize_for_similarity(strip_diagnostic_markers(text or "")))
    suspected, reason = _truncation_verdict(
        norm_chars, audio_sec, active_sec, tail_sec,
        bool(upstream_incomplete), int(failed_segment_count or 0),
        bool(ambiguous_output),
    )
    return OutputTruncationAssessment(
        suspected_truncated=suspected,
        truncation_reason=reason,
        norm_char_count=norm_chars,
        chars_per_audio_sec=round(norm_chars / audio_sec, 3) if audio_sec > 0 else 0.0,
        chars_per_active_sec=(
            round(norm_chars / active_sec, 3)
            if active_sec is not None and active_sec > 0 else None
        ),
    )


# ── Deterministic final-transcript normalization policy (WhisperWave) ──
from typing import Callable, Match, Pattern

RULE_NUMBER_ORDINAL = "number.ordinal"
RULE_NUMBER_VERSION = "number.version_decimal"
RULE_NUMBER_IDENTIFIER = "number.identifier"
RULE_NUMBER_COUNT = "number.count"
RULE_ACRONYM_SSOT = "domain.acronym.ssot"
RULE_CLAUDE_CODE = "domain.claude_code"


_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUMERAL_CHARS = "零〇一二两三四五六七八九十百千万"
_CANONICAL_DIGIT_CHARS = "零〇一二三四五六七八九"
_NUMERAL_TOKEN = rf"[{_NUMERAL_CHARS}]+"
_VERSION_COMPONENT = rf"[{_NUMERAL_CHARS}0-9]+"
_IDENTIFIER_COMPONENT = rf"[{_CANONICAL_DIGIT_CHARS}0-9]+"
_TERMINATING_PUNCTUATION = frozenset(
    "，。！？、；：,!?;:）)]】》」』\"'”’"
)

_ZERO_CHARS = "零〇"
_NONZERO_DIGITS = "一二三四五六七八九"
_UNIT_MULTIPLIER_DIGITS = "一二两三四五六七八九"
_ONES_GRAMMAR = rf"[{_UNIT_MULTIPLIER_DIGITS}]"
_TENS_WITH_LEADING_GRAMMAR = (
    rf"[{_NONZERO_DIGITS}]十(?:[{_NONZERO_DIGITS}])?"
)
_TENS_GRAMMAR = (
    rf"(?:十(?:[{_NONZERO_DIGITS}])?|{_TENS_WITH_LEADING_GRAMMAR})"
)
_HUNDREDS_GRAMMAR = (
    rf"[{_UNIT_MULTIPLIER_DIGITS}]百"
    rf"(?:[{_ZERO_CHARS}][{_NONZERO_DIGITS}]|"
    rf"{_TENS_WITH_LEADING_GRAMMAR})?"
)
_THOUSANDS_GRAMMAR = (
    rf"[{_UNIT_MULTIPLIER_DIGITS}]千"
    rf"(?:[{_ZERO_CHARS}](?:[{_NONZERO_DIGITS}]|"
    rf"{_TENS_WITH_LEADING_GRAMMAR})|{_HUNDREDS_GRAMMAR})?"
)
_CANONICAL_SECTION_RE = re.compile(
    rf"(?:[{_ZERO_CHARS}]|{_ONES_GRAMMAR}|{_TENS_GRAMMAR}|"
    rf"{_HUNDREDS_GRAMMAR}|{_THOUSANDS_GRAMMAR})\Z"
)


_ORDINAL_RE = re.compile(
    rf"第(?P<number>{_NUMERAL_TOKEN})(?P<unit>次|轮|版|章|节|页|期|名|个|条|项|组|步|段|回|届|季|集|关|天|年|月|周|号)"
)
_VERSION_RE = re.compile(
    rf"(?P<label>版本(?:号)?)(?P<spacing>[ \t]*)"
    rf"(?P<body>{_VERSION_COMPONENT}[ \t]*点[ \t]*{_VERSION_COMPONENT}"
    rf"(?:[ \t]*[点.．\-－/／][ \t]*\w*)*)"
)
_VERSION_SEPARATOR_RE = re.compile(r"[ \t]*(?:点|[.．])[ \t]*")
_IDENTIFIER_RE = re.compile(
    rf"(?P<label>端口(?:号)?|编号|验证码|工号|序号)(?P<spacing>[ \t]*)"
    rf"(?P<body>{_IDENTIFIER_COMPONENT}"
    rf"(?:[ \t]*[-－/／.．][ \t]*\w*)*)"
)
_IDENTIFIER_SEPARATOR_RE = re.compile(r"[ \t]*([-－/／.．])[ \t]*")
_COUNT_RE = re.compile(
    rf"(?<![{_NUMERAL_CHARS}])(?P<number>{_NUMERAL_TOKEN})"
    rf"(?P<context>个(?:字母|样本|字符|文件|参数|步骤|任务|项目|测试|请求|响应|模型|规则|字段|条目))"
)

_SSOT_ALIAS_RE = re.compile(
    r"(?P<context>(?:(?:这个|该|其|本)?(?:缩写|简称)[ \t]*"
    r"(?:(?:是|为|叫(?:作)?)|[:：])?[ \t]*|"
    r"(?:它|这个|这个词)[ \t]*(?:是|为)[ \t]*|这是[ \t]*))"
    r"(?P<alias>S[ \t]+S[ \t]+O[ \t]+T|SOT)"
    r"(?=$|[ \t\r\n，。！？、；：,!?;:）)\]】》」』\"”’])"
)

_CLAUDE_CODE_CANDIDATE_RE = re.compile(
    r"(?:(?P<phonetic>克劳德|克劳得|克劳特|克勞德|克勞得|克勞特)[ \t]*|"
    r"(?P<latin>cloude|claude|claud)[ \t]+)"
    r"code",
    re.IGNORECASE,
)
_CLAUDE_CODE_ZH_CUE = r"(?:使用|用|通过|打开|调用|运行|让|请)"
_CLAUDE_CODE_CUE_RE = re.compile(
    rf"(?:{_CLAUDE_CODE_ZH_CUE}|"
    r"(?<![A-Za-z0-9_])(?:use|using|with))[ \t]*\Z",
    re.IGNORECASE,
)
_CLAUDE_CODE_ATTACHED_CUE_RE = re.compile(rf"{_CLAUDE_CODE_ZH_CUE}\Z")
_CLAUDE_CODE_SENTENCE_END = frozenset("。！？.!?")



@dataclass(frozen=True)
class TranscriptPolicyResult:
    """Normalized final text plus privacy-safe rule application metadata."""

    text: str
    applied_rule_ids: Tuple[str, ...]
    applied_count: int


Replacement = Callable[[Match[str]], str]


def _replace_changed(
    pattern: Pattern[str], text: str, replacement: Replacement
) -> Tuple[str, int]:
    """Apply a regex and count only replacements that actually change text."""

    changed = 0

    def replace(match: Match[str]) -> str:
        nonlocal changed
        value = replacement(match)
        if value != match.group(0):
            changed += 1
        return value

    return pattern.sub(replace, text), changed


def _parse_under_ten_thousand(token: str) -> Optional[int]:
    """Parse one section only when it matches the canonical numeral grammar."""

    if _CANONICAL_SECTION_RE.fullmatch(token) is None:
        return None

    total = 0
    pending_digit: Optional[int] = None
    last_unit = 10_000

    for char in token:
        if char in _CHINESE_DIGITS:
            digit = _CHINESE_DIGITS[char]
            if pending_digit is None or pending_digit == 0:
                pending_digit = digit
                continue
            # Consecutive non-zero digits are a spoken digit sequence, not a
            # formal cardinal. Identifier/version contexts handle those safely.
            return None

        unit = _SMALL_UNITS.get(char)
        if unit is None or unit >= last_unit:
            return None
        multiplier = 1 if pending_digit is None else pending_digit
        if multiplier == 0:
            return None
        total += multiplier * unit
        pending_digit = None
        last_unit = unit

    if pending_digit is not None:
        total += pending_digit
    return total


def _parse_cardinal(token: str) -> Optional[int]:
    """Parse a conservative Chinese cardinal used only in numeric contexts."""

    if not token or any(char not in _NUMERAL_CHARS for char in token):
        return None

    if all(char in _CHINESE_DIGITS for char in token):
        # A multi-digit unitless sequence such as 两三 is often an approximation
        # ("two or three"), so generic count/ordinal rules leave it untouched.
        return _CHINESE_DIGITS[token] if len(token) == 1 else None

    if token.count("万") > 1:
        return None
    if "万" in token:
        high, low = token.split("万", 1)
        if not high:
            return None
        high_value = _parse_under_ten_thousand(high)
        if high_value is None or high_value == 0:
            return None
        if not low:
            return high_value * 10_000

        if low[0] in _ZERO_CHARS:
            low = low[1:]
            if not low or low[0] in _ZERO_CHARS:
                return None
            low_value = _parse_under_ten_thousand(low)
            if low_value is None or low_value >= 1000:
                return None
        else:
            low_value = _parse_under_ten_thousand(low)
            if low_value is None or low_value < 1000:
                return None
        return high_value * 10_000 + low_value

    return _parse_under_ten_thousand(token)


def _decimal_component(token: str) -> Optional[str]:
    """Render a version component, preserving spoken digit sequences/zeroes."""

    if token.isascii() and token.isdigit():
        return token
    if "两" in token:
        return None
    if token and all(char in _CANONICAL_DIGIT_CHARS for char in token):
        return "".join(str(_CHINESE_DIGITS[char]) for char in token)
    value = _parse_cardinal(token)
    return None if value is None else str(value)


def _identifier_component(token: str) -> Optional[str]:
    """Render one direct-digit identifier component without guessing units."""

    normalized = []
    for char in token:
        if char in _CANONICAL_DIGIT_CHARS:
            normalized.append(str(_CHINESE_DIGITS[char]))
        elif "0" <= char <= "9":
            normalized.append(char)
        else:
            return None
    return "".join(normalized) if normalized else None


def _candidate_has_strong_end(match: Match[str]) -> bool:
    """Require sentence/end authority after a captured numeric candidate.

    Horizontal whitespace is only formatting. If prose continues after it, the
    candidate remains ambiguous and is preserved. A line break, terminal
    punctuation, or the actual end of text is strong enough to normalize.
    """

    text = match.string
    index = match.end()
    if index >= len(text):
        return True
    if text[index] in _TERMINATING_PUNCTUATION or text[index] in "\r\n":
        return True
    if not text[index].isspace():
        return False

    while index < len(text) and text[index].isspace():
        if text[index] in "\r\n":
            return True
        index += 1
    return index >= len(text) or text[index] in _TERMINATING_PUNCTUATION


def _apply_ordinal(text: str) -> Tuple[str, int]:
    def replace(match: Match[str]) -> str:
        value = _parse_cardinal(match.group("number"))
        if value is None:
            return match.group(0)
        return f"第{value}{match.group('unit')}"

    return _replace_changed(_ORDINAL_RE, text, replace)


def _apply_version(text: str) -> Tuple[str, int]:
    def replace(match: Match[str]) -> str:
        if not _candidate_has_strong_end(match):
            return match.group(0)
        normalized_components = []
        for component in _VERSION_SEPARATOR_RE.split(match.group("body")):
            normalized = _decimal_component(component)
            if normalized is None:
                return match.group(0)
            normalized_components.append(normalized)
        return (
            f"{match.group('label')}{match.group('spacing')}"
            f"{'.'.join(normalized_components)}"
        )

    return _replace_changed(_VERSION_RE, text, replace)


def _apply_identifier(text: str) -> Tuple[str, int]:
    def replace(match: Match[str]) -> str:
        if not _candidate_has_strong_end(match):
            return match.group(0)
        normalized_parts = []
        digit_count = 0
        for index, part in enumerate(
            _IDENTIFIER_SEPARATOR_RE.split(match.group("body"))
        ):
            if index % 2:
                normalized_parts.append(part)
                continue
            normalized = _identifier_component(part)
            if normalized is None:
                return match.group(0)
            normalized_parts.append(normalized)
            digit_count += len(normalized)
        if digit_count < 2:
            return match.group(0)
        return (
            f"{match.group('label')}{match.group('spacing')}"
            f"{''.join(normalized_parts)}"
        )

    return _replace_changed(_IDENTIFIER_RE, text, replace)


def _apply_count(text: str) -> Tuple[str, int]:
    def replace(match: Match[str]) -> str:
        value = _parse_cardinal(match.group("number"))
        if value is None:
            return match.group(0)
        return f"{value}{match.group('context')}"

    return _replace_changed(_COUNT_RE, text, replace)


def _is_unicode_token_continuation(char: str) -> bool:
    """Return whether a code point continues a Unicode word-like token."""

    category = unicodedata.category(char)
    return category[0] in "LMN" or category == "Pc"


def _starts_with_single_unicode_letter_token(suffix: str) -> bool:
    """Detect one Unicode letter grapheme after horizontal whitespace."""

    index = 0
    while index < len(suffix) and suffix[index] in " \t":
        index += 1
    if index == 0 or index == len(suffix):
        return False
    if unicodedata.category(suffix[index])[0] != "L":
        return False

    index += 1
    while index < len(suffix) and unicodedata.category(suffix[index])[0] == "M":
        index += 1
    return index == len(suffix) or not _is_unicode_token_continuation(
        suffix[index]
    )


def _apply_ssot(text: str) -> Tuple[str, int]:
    def replace(match: Match[str]) -> str:
        alias = match.group("alias")
        if any(char in " \t" for char in alias):
            suffix = match.string[match.end() :]
            if _starts_with_single_unicode_letter_token(suffix):
                return match.group(0)
        return f"{match.group('context')}SSOT"

    return _replace_changed(
        _SSOT_ALIAS_RE,
        text,
        replace,
    )


def _has_natural_language_claude_code_suffix(
    suffix: str, *, allow_attached_prose: bool
) -> bool:
    """Accept only prose whose Unicode token has a natural-language end."""

    index = 0
    while index < len(suffix) and suffix[index] in " \t":
        index += 1
    if index == len(suffix):
        return True

    is_attached = index == 0
    next_char = suffix[index]
    if unicodedata.category(next_char)[0] == "L":
        if is_attached and not allow_attached_prose:
            return False

        letter_count = 0
        token_end = index
        while token_end < len(suffix):
            category = unicodedata.category(suffix[token_end])[0]
            if category not in "LM":
                break
            if category == "L":
                letter_count += 1
            token_end += 1

        if is_attached and letter_count < 2:
            return False
        if token_end == len(suffix) or suffix[token_end] in " \t\r\n":
            return True
        return (
            suffix[token_end] in _CLAUDE_CODE_SENTENCE_END
            and (
                token_end + 1 == len(suffix)
                or suffix[token_end + 1] in " \t\r\n"
            )
        )
    if _is_unicode_token_continuation(next_char):
        return False
    if next_char not in _CLAUDE_CODE_SENTENCE_END:
        return False
    following_index = index + 1
    return (
        following_index == len(suffix)
        or suffix[following_index] in " \t\r\n"
    )


def _is_free_standing_claude_code_candidate(match: Match[str]) -> bool:
    """Require positive natural-language structure around a candidate.

    Canonical Latin spelling requires an explicit language cue so a bare command
    remains byte-for-byte stable. A Latin misspelling or phonetic candidate may
    also occupy the whole horizontal-whitespace-trimmed transcript. Structural
    neighbors never become accepted merely because they were omitted from a
    denylist.
    """

    prefix = match.string[: match.start()]
    suffix = match.string[match.end() :]
    prefix_is_horizontal_space = not prefix.strip(" \t")
    suffix_is_horizontal_space = not suffix.strip(" \t")

    if (
        match.start()
        and _is_unicode_token_continuation(match.string[match.start() - 1])
        and _CLAUDE_CODE_ATTACHED_CUE_RE.search(prefix) is None
    ):
        return False

    if prefix_is_horizontal_space and suffix_is_horizontal_space:
        latin = match.group("latin")
        return latin is None or latin.casefold() != "claude"
    if not _has_natural_language_claude_code_suffix(
        suffix,
        allow_attached_prose=match.group("phonetic") is not None,
    ):
        return False
    if _CLAUDE_CODE_CUE_RE.search(prefix) is not None:
        return True
    return match.group("phonetic") is not None and prefix_is_horizontal_space


def _apply_claude_code(text: str) -> Tuple[str, int]:
    def replace(match: Match[str]) -> str:
        if not _is_free_standing_claude_code_candidate(match):
            return match.group(0)
        return "Claude Code"

    return _replace_changed(
        _CLAUDE_CODE_CANDIDATE_RE,
        text,
        replace,
    )


class TranscriptPolicy:
    """Final-only, deterministic normalization policy for completed text.

    Built-in rules run first, then the speaker vocabulary's final rules (`vocabulary.py`).
    """

    _BUILTIN_RULES = (
        (RULE_NUMBER_VERSION, _apply_version),
        (RULE_NUMBER_IDENTIFIER, _apply_identifier),
        (RULE_NUMBER_ORDINAL, _apply_ordinal),
        (RULE_NUMBER_COUNT, _apply_count),
        (RULE_ACRONYM_SSOT, _apply_ssot),
        (RULE_CLAUDE_CODE, _apply_claude_code),
    )

    def __init__(self, vocabulary_rules=vocabulary.FINAL_RULES) -> None:
        self._rules = self._BUILTIN_RULES + tuple(
            (rule_id, functools.partial(_replace_changed, pattern, replacement=replacement))
            for rule_id, pattern, replacement in vocabulary_rules
        )

    def normalize_final(self, text: str) -> TranscriptPolicyResult:
        if not text:
            return TranscriptPolicyResult(text=text, applied_rule_ids=(), applied_count=0)

        normalized = text
        rule_ids = []
        applied_count = 0
        for rule_id, apply_rule in self._rules:
            normalized, count = apply_rule(normalized)
            if count:
                rule_ids.append(rule_id)
                applied_count += count
        return TranscriptPolicyResult(
            text=normalized,
            applied_rule_ids=tuple(rule_ids),
            applied_count=applied_count,
        )


_DEFAULT_TRANSCRIPT_POLICY = TranscriptPolicy()


def normalize_final_transcript(text: str) -> TranscriptPolicyResult:
    """Normalize authoritative completed text and return rule metadata."""

    return _DEFAULT_TRANSCRIPT_POLICY.normalize_final(text)
