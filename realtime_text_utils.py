"""Utilities for parsing transcript content emitted by realtime providers."""

import difflib
import hashlib
import logging
import math
import os
import re
import unicodedata
from typing import Tuple

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


def apply_homonym_correction(text: str) -> str:
    """Replace 'cloud/云/克劳德 + code' homonyms with 'Claude Code' (case-insensitive)."""
    if not text:
        return text
    return _HOMONYM_RE.sub('Claude Code', text)


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
