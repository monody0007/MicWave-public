"""Helpers for assembling incremental transcript deltas and segment joins."""

from __future__ import annotations

import re


def _is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    codepoint = ord(ch)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x20000 <= codepoint <= 0x2A6DF
    )


def _is_all_cjk(text: str) -> bool:
    return bool(text) and all(_is_cjk_char(ch) for ch in text)


def _should_preserve_cjk_single_char_repeat(next_ch: str) -> bool:
    return next_ch in {"看", "下", "听", "想", "问", "说"}


def _digit_suffix_len(text: str) -> int:
    count = 0
    for ch in reversed(text):
        if ch.isdigit():
            count += 1
            continue
        break
    return count


def _digit_prefix_len(text: str) -> int:
    count = 0
    for ch in text:
        if ch.isdigit():
            count += 1
            continue
        break
    return count


def merge_incremental_text(current: str, delta: str, max_overlap: int = 32) -> str:
    """Merge streamed delta text into the current transcript."""
    if not delta:
        return current
    if not current:
        return delta

    overlap_cap = min(len(current), len(delta), max_overlap)
    for overlap in range(overlap_cap, 1, -1):
        overlap_text = delta[:overlap]
        if not current.endswith(overlap_text):
            continue

        if overlap == len(delta) and len(delta) <= 2 and _is_all_cjk(delta):
            continue

        if _is_all_cjk(overlap_text) and overlap < 3:
            continue

        return current + delta[overlap:]

    if (
        current[-1] == delta[0]
        and current[-1] == "0"
        and len(delta) >= 2
        and delta[1].isdigit()
    ):
        return current + delta[1:]

    if current[-1] == delta[0] and current[-1].isdigit():
        left_digits = _digit_suffix_len(current)
        right_digits = _digit_prefix_len(delta)
        if right_digits >= 2 and (left_digits + right_digits) >= 5:
            return current + delta[1:]

    if (
        current[-1] == delta[0]
        and _is_cjk_char(current[-1])
        and len(delta) > 1
    ):
        if _should_preserve_cjk_single_char_repeat(delta[1]):
            return current + delta
        return current + delta[1:]

    return current + delta


_SEGMENT_FAILURE_PLACEHOLDER = re.compile(r"^\[片段\d+转录失败[^\]]*\]$")
_SENTENCE_END = frozenset("。！？!?；;…，,、：:\n")


def join_segment_texts(segments) -> str:
    """Join transcripts of consecutive pause-cut audio segments into one text."""
    parts = [s.strip() for s in segments if isinstance(s, str) and s.strip()]
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        prev_failed = bool(_SEGMENT_FAILURE_PLACEHOLDER.match(out.rsplit("\n", 1)[-1]))
        if prev_failed or _SEGMENT_FAILURE_PLACEHOLDER.match(part):
            out = f"{out}\n{part}"
        elif out[-1] in _SENTENCE_END:
            out += part
        elif out[-1].isascii() and out[-1].isalnum() and part[0].isascii() and part[0].isalnum():
            out = f"{out} {part}"
        else:
            out = f"{out}，{part}"
    return out


# ── WhisperWave verbatim delta + segment helpers ──
def append_transcript_delta(current: str, delta: str) -> str:
    """Append one provider delta verbatim."""
    return current + delta


_UNSPACED_RANGES = (
    (0x0E00, 0x0EFF),
    (0x1000, 0x109F),
    (0x1780, 0x17FF),
    (0x2E80, 0x2FDF),
    (0x3040, 0x30FF),
    (0x3100, 0x312F),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF66, 0xFF9F),
    (0x20000, 0x3FFFF),
)
_WHISPER_SENTENCE_PUNCTUATION = ".,;:!?)"


def _is_spaced_word_char(char: str) -> bool:
    if not char.isalnum():
        return False
    code = ord(char)
    return not any(low <= code <= high for low, high in _UNSPACED_RANGES)


def join_whisper_segment_texts(prefix: str, segment: str) -> str:
    """Join finalized WhisperWave prefix with next independently decoded segment."""
    if not prefix or not segment:
        return prefix + segment
    left, right = prefix[-1], segment[0]
    if (
        _is_spaced_word_char(right)
        and (_is_spaced_word_char(left) or left in _WHISPER_SENTENCE_PUNCTUATION)
    ):
        return f"{prefix} {segment}"
    return prefix + segment


def _prefix_table(text: str) -> list[int]:
    prefix = [0] * len(text)
    for index in range(1, len(text)):
        matched = prefix[index - 1]
        while matched and text[index] != text[matched]:
            matched = prefix[matched - 1]
        if text[index] == text[matched]:
            matched += 1
        prefix[index] = matched
    return prefix


def find_pathological_repetition(text: str) -> tuple[int, int, int] | None:
    """Find an extreme exact paragraph loop (>=80 chars * >=12 copies, >=80% span)."""
    required_span = max(1_200, (len(text) * 80 + 99) // 100)
    if len(text) < required_span:
        return None
    start = 0
    last_start = len(text) - required_span
    while start <= last_start:
        anchor = text[start:start + 80]
        max_unit = (len(text) - start) // 12
        candidate = text.find(anchor, start + 80, start + max_unit + 80)
        if candidate == -1:
            start += 1
            continue
        periods = _prefix_table(text[start:start + max_unit])
        while candidate != -1:
            unit = text[start:candidate]
            period = len(unit) - periods[len(unit) - 1]
            if len(unit) % period:
                period = len(unit)
            end = candidate
            copies = 1
            while text.startswith(unit, end):
                copies += 1
                end += len(unit)
            if period < 80 and end - start >= max_unit + period:
                start = end - max_unit - period + 1
                break
            if copies >= 12 and end - start >= required_span:
                if period >= 80:
                    return start, len(unit), copies
            candidate = text.find(anchor, candidate + 1, start + max_unit + 80)
        else:
            start += 1
    return None
