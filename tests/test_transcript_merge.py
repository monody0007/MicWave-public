import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transcript_merge import merge_incremental_text


def test_merge_with_empty_values():
    assert merge_incremental_text("", "") == ""
    assert merge_incremental_text("", "hello") == "hello"
    assert merge_incremental_text("hello", "") == "hello"


def test_merge_multi_char_overlap():
    assert merge_incremental_text("hello wor", "world") == "hello world"
    assert merge_incremental_text("今天是周", "周日") == "今天是周日"


def test_merge_numeric_single_char_overlap():
    assert merge_incremental_text("0", "08") == "08"
    # Numeric single-char overlap is now intentionally looser to avoid over-dedup.
    assert merge_incremental_text("02", "2") == "022"
    assert merge_incremental_text("2", "22") == "222"


def test_merge_long_numeric_single_char_overlap_is_deduped():
    # Long digit runs (QQ/phone/account style) should dedupe 1-char overlap.
    assert merge_incremental_text("74777", "78175") == "747778175"
    assert merge_incremental_text("QQ号是74777", "78175") == "QQ号是747778175"


def test_single_char_non_digit_overlap_is_conservative():
    # Keep behavior conservative for regular text to avoid accidental character loss.
    assert merge_incremental_text("ab", "bcd") == "abbcd"


def test_single_char_cjk_overlap_requires_delta_gt_one_char():
    assert merge_incremental_text("哈", "哈") == "哈哈"
    assert merge_incremental_text("哈", "哈哈") == "哈哈"


def test_merge_when_delta_fully_repeated():
    assert merge_incremental_text("版本20", "版本20") == "版本20"


def test_preserve_common_short_chinese_repetition():
    assert merge_incremental_text("一点", "一点") == "一点一点"
    assert merge_incremental_text("一点一", "一点") == "一点一点"


def test_preserve_shishikan_style_phrase():
    assert merge_incremental_text("你可以试", "试看") == "你可以试试看"
