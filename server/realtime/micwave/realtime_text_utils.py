"""Utilities for parsing transcript content emitted by realtime providers."""

import re
from typing import Tuple


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
        return emit

    def flush(self) -> str:
        emit = apply_homonym_correction(self._buffer)
        self._buffer = ""
        return emit


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
