import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime_text_utils import (
    StreamingHomonymCorrector,
    apply_homonym_correction,
    extract_text_after_marker,
)


MARKER = "下面是不改变语言的语音识别结果：\n\n"


def test_extract_text_after_marker_exact_prefix():
    found, text = extract_text_after_marker(MARKER + "你好，世界", MARKER)
    assert found is True
    assert text == "你好，世界"


def test_extract_text_after_marker_prefix_without_trailing_newlines():
    found, text = extract_text_after_marker("下面是不改变语言的语音识别结果：\n内容", MARKER)
    assert found is True
    assert text == "内容"


def test_extract_text_after_marker_prefix_in_middle():
    raw = "noise\n下面是不改变语言的语音识别结果：\n\n正文"
    found, text = extract_text_after_marker(raw, MARKER)
    assert found is True
    assert text == "正文"


def test_extract_text_after_marker_not_found():
    found, text = extract_text_after_marker("no marker here", MARKER)
    assert found is False
    assert text == ""


def test_extract_text_after_marker_empty_inputs():
    assert extract_text_after_marker("", MARKER) == (False, "")
    assert extract_text_after_marker("abc", "") == (False, "")


# ─── apply_homonym_correction ──────────────────────────────────────────────


def test_apply_homonym_basic_english():
    assert apply_homonym_correction("I use cloud code") == "I use Claude Code"
    assert apply_homonym_correction("Cloud Code is great") == "Claude Code is great"
    assert apply_homonym_correction("CLAUD CODE") == "Claude Code"


def test_apply_homonym_chinese_variants():
    assert apply_homonym_correction("用克劳德code写程序") == "用Claude Code写程序"
    assert apply_homonym_correction("云 code 很好用") == "Claude Code 很好用"
    assert "Claude Code" in apply_homonym_correction("克劳得code")


def test_apply_homonym_no_match_passthrough():
    assert apply_homonym_correction("hello world") == "hello world"
    assert apply_homonym_correction("") == ""


def test_apply_homonym_separator_tolerant():
    # patterns separated by hyphen / multiple spaces still match
    assert apply_homonym_correction("cloud-code") == "Claude Code"
    assert apply_homonym_correction("cloud  code") == "Claude Code"


# ─── StreamingHomonymCorrector: basic push/flush ───────────────────────────


def test_corrector_short_push_holds_buffer():
    # push less than hold_tail (16 chars) -> nothing emitted, buffer retained
    c = StreamingHomonymCorrector()
    assert c.push("hello") == ""
    # flush drains the held content
    assert c.flush() == "hello"
    # buffer is empty after flush
    assert c.flush() == ""


def test_corrector_long_push_emits_prefix_holds_tail():
    c = StreamingHomonymCorrector()
    text = "a" * 32  # well past hold_tail
    emitted = c.push(text)
    # emit length = total - hold_tail
    assert len(emitted) == 32 - 16
    assert emitted == "a" * 16
    # remaining 16 chars come out on flush
    assert c.flush() == "a" * 16


def test_corrector_empty_push_is_noop():
    c = StreamingHomonymCorrector()
    assert c.push("") == ""
    assert c.flush() == ""


def test_corrector_homonym_correction_within_buffer():
    # When the homonym pattern fits within a single push, correction applies
    c = StreamingHomonymCorrector()
    out = c.push("I love cloud code very much, it is great") + c.flush()
    assert "Claude Code" in out
    assert "cloud code" not in out.lower() or "Claude Code" in out


def test_corrector_cross_chunk_pattern_match():
    # Pattern split across two pushes should still match thanks to hold_tail
    c = StreamingHomonymCorrector()
    out = c.push("I use Cloud") + c.push(" Code daily and like it a lot")
    out += c.flush()
    assert "Claude Code" in out, f"cross-chunk match failed: {out!r}"


# ─── reset(): regression test for the double-paste bug ─────────────────────


def test_corrector_reset_drops_buffer_without_emitting():
    # The fix: consumer can drop the held tail when it's about to overwrite
    # the entire stream (e.g. server fallback emitting isNewResponse=True).
    c = StreamingHomonymCorrector()
    c.push("hello world this is some long text exceeding hold tail")
    c.reset()
    # After reset, flush MUST emit nothing, otherwise the consumer would
    # append the leftover tail to the just-replaced text → double paste.
    assert c.flush() == ""


def test_corrector_reset_after_short_push():
    # Short push keeps everything in buffer; reset must still empty it.
    c = StreamingHomonymCorrector()
    c.push("几点了")  # 3 chars, all held
    c.reset()
    assert c.flush() == ""


def test_corrector_reset_then_reuse():
    # After reset, the corrector must still work correctly for next turn.
    c = StreamingHomonymCorrector()
    c.push("first turn data exceeding hold tail")
    c.reset()
    out = c.push("second turn cloud code text here") + c.flush()
    assert "Claude Code" in out
    assert "first turn" not in out


def test_corrector_reset_is_idempotent():
    c = StreamingHomonymCorrector()
    c.push("some content")
    c.reset()
    c.reset()  # second reset on empty buffer
    assert c.flush() == ""


# ─── Bug regression: simulate the exact failure scenario ───────────────────


def test_bug_regression_fallback_replace_then_finalize_no_double_emit():
    """
    Reproduces the 2026-04-26 double-paste bug:

    Scenario: short utterance, model didn't follow marker prompt → server
    fallback path replaces the streamed text via isNewResponse=True. The
    StreamingHomonymCorrector still held its hold_tail in buffer, and the
    post-fallback flush would append that tail → second paste with mixed
    content.

    Expected behavior (post-fix): consumer calls reset() before sending the
    fallback payload, so the subsequent flush emits nothing.
    """
    corrector = StreamingHomonymCorrector()

    # Simulate streaming: model emits some delta before response.done.
    # All within hold_tail, so nothing is emitted to client yet — buffered.
    streamed = corrector.push("几点了")
    assert streamed == ""  # nothing emitted; sits in buffer

    # Server decides to apply fallback (replace via isNewResponse=True).
    # The fix: reset corrector BEFORE sending the replace payload.
    corrector.reset()

    # finalize_turn's flush must now emit nothing — otherwise the client
    # would receive the replace payload AND the leftover tail = double paste.
    flushed = corrector.flush()
    assert flushed == "", (
        f"Bug regression: flush after reset emitted {flushed!r}, "
        "which would double-paste onto the just-replaced text."
    )
