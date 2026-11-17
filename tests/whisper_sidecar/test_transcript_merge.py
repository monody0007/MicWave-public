import pathlib
import sys

import pytest


from micwave_core.transcript_merge import (
    append_transcript_delta,
    find_pathological_repetition,
    join_whisper_segment_texts as join_segment_texts,
)


def test_append_with_empty_values():
    assert append_transcript_delta("", "") == ""
    assert append_transcript_delta("", "hello") == "hello"
    assert append_transcript_delta("hello", "") == "hello"


@pytest.mark.parametrize(
    "deltas",
    [
        ("SSOT", "SSOT"),
        ("你好啊", "你好啊"),
        ("abc", "abc"),
        ("S", "S", "O", "T"),
    ],
)
def test_r3_m4_provider_deltas_append_as_newly_available_chunks(deltas):
    transcript = ""
    for delta in deltas:
        transcript = append_transcript_delta(transcript, delta)
    assert transcript == "".join(deltas)


@pytest.mark.parametrize(
    "unit_chars,copies,extra_chars,suspect",
    [
        (180, 31, 179, True),
        (180, 12, 0, True),
        (180, 11, 0, False),
        (500, 3, 0, False),
        (79, 31, 0, False),
        (80, 15, 0, True),
        (80, 14, 0, False),
        (180, 12, 540, True),
        (180, 12, 541, False),
    ],
)
def test_repetition_guard_has_exact_boundaries_and_preserves_unique_material(
    unit_chars, copies, extra_chars, suspect
):
    unit = "".join(chr(0x4E00 + i) for i in range(unit_chars))
    unique = "".join(chr(0x6000 + i) for i in range(extra_chars))
    result = find_pathological_repetition(unit * copies + unique)
    assert (result is not None) is suspect
    if suspect:
        assert result == (0, unit_chars, copies)


def test_repetition_guard_does_not_treat_short_word_loops_as_paragraphs():
    assert find_pathological_repetition("yes " * 400) is None
    assert find_pathological_repetition("Please repeat this. " * 100) is None


def test_repetition_guard_finds_dominant_loop_after_unique_preface():
    unit = "".join(f"{index:03d} " for index in range(45))
    text = "Unique preface:" + unit * 31 + "Unique ending."
    assert find_pathological_repetition(text) == (15, 180, 31)


@pytest.mark.parametrize(
    "text",
    [
        "ha" * 3000,
        "x" * 6000,
        "Unique preface:" + "ha" * 3000 + "Unique suffix.",
        ("ha" * 499 + "!?") * 6,
        "ha" * 2370 + "".join(chr(0x6000 + i) for i in range(1260)),
        "ha" * 4740 + "".join(chr(0x6000 + i) for i in range(2520)),
        "Unique preface:" + "".join(f"{i:03d} " for i in range(140)) * 11 + "Unique suffix.",
    ],
)
def test_repetition_guard_does_not_block_the_event_loop_on_negative_controls(text):
    from time import perf_counter

    started = perf_counter()
    assert find_pathological_repetition(text) is None
    assert perf_counter() - started < 0.5


@pytest.mark.parametrize(
    ("prefix", "segment", "expected"),
    [
        # CJK and other unspaced scripts join verbatim, never with a stray space.
        ("你来好好设计这个", "Control Plane怎么样", "你来好好设计这个Control Plane怎么样"),
        ("前段内容。", "后段内容", "前段内容。后段内容"),
        ("前段内容", "后段内容", "前段内容后段内容"),
        ("这个是", "OK的", "这个是OK的"),
        ("設計を", "お願いします", "設計をお願いします"),
        ("Control", "了吗", "Control了吗"),
        ("Version", "2 is out", "Version 2 is out"),
        # Spaced scripts get exactly one space at the boundary.
        ("the quick", "brown fox", "the quick brown fox"),
        ("First part.", "Second part.", "First part. Second part."),
        ("Okay,", "so we go", "Okay, so we go"),
        ("设计 Control", "Plane 看板", "设计 Control Plane 看板"),
        ("안녕", "하세요", "안녕 하세요"),
        # Existing whitespace on either side is never doubled.
        ("the quick ", "brown", "the quick brown"),
        ("the quick", " brown", "the quick brown"),
        ("line one\n", "line two", "line one\nline two"),
        # Empty sides and non-word starts are verbatim.
        ("", "hello", "hello"),
        ("hello", "", "hello"),
        ("hello", "...", "hello..."),
        ("(note)", "(next)", "(note)(next)"),
    ],
)
def test_join_segment_texts_spaces_only_between_spaced_script_words(
    prefix, segment, expected
):
    assert join_segment_texts(prefix, segment) == expected


def test_join_segment_texts_is_stable_over_many_segments():
    parts = ["设计这个", "Control Plane", "怎么样。", "Next, ", "step two.", "好的"]
    joined = ""
    for part in parts:
        joined = join_segment_texts(joined, part)
    assert joined == "设计这个Control Plane怎么样。Next, step two.好的"
