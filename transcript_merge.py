"""Helpers for assembling incremental transcript deltas safely."""


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
    # Common colloquial constructions where a repeated first char is intentional,
    # e.g. "试试看", "想想看", "问问看", "听听看".
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
    """Merge streamed delta text into the current transcript.

    Strategy:
    - Prefer suffix/prefix overlap to avoid duplicate appends.
    - For Chinese text, keep overlap dedup conservative (>= 3 chars) so natural
      repeated phrases like "一点一点" are preserved.
    - Keep a narrowly-scoped numeric fix for "0" + "08" style duplication.
    """
    if not delta:
        return current
    if not current:
        return delta

    overlap_cap = min(len(current), len(delta), max_overlap)
    for overlap in range(overlap_cap, 1, -1):
        overlap_text = delta[:overlap]
        if not current.endswith(overlap_text):
            continue

        # Preserve intentionally repeated short Chinese phrases such as "一点一点".
        if overlap == len(delta) and len(delta) <= 2 and _is_all_cjk(delta):
            continue

        # For Chinese, 2-char overlap is often a valid repetition in natural speech.
        if _is_all_cjk(overlap_text) and overlap < 3:
            continue

        return current + delta[overlap:]

    # Keep this narrowly-scoped numeric fix for the "0 + 08 -> 08" class only.
    if (
        current[-1] == delta[0]
        and current[-1] == "0"
        and len(delta) >= 2
        and delta[1].isdigit()
    ):
        return current + delta[1:]

    # For long numeric runs (QQ/phone/account-like), single-char overlap is usually
    # a streaming boundary artifact and should be deduped conservatively.
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
