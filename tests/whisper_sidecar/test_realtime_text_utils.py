"""Behavior tests for completed-transcript normalization."""

import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_backend.realtime_text_utils import (
    RULE_ACRONYM_SSOT,
    RULE_CLAUDE_CODE,
    RULE_NUMBER_COUNT,
    RULE_NUMBER_IDENTIFIER,
    RULE_NUMBER_ORDINAL,
    RULE_NUMBER_VERSION,
    TranscriptPolicy,
    normalize_final_transcript,
)


@pytest.mark.parametrize(
    ("source", "expected", "rule_id"),
    [
        ("这里有四个字母", "这里有4个字母", RULE_NUMBER_COUNT),
        ("第十二次测试", "第12次测试", RULE_NUMBER_ORDINAL),
        ("版本二点一", "版本2.1", RULE_NUMBER_VERSION),
        ("当前是版本一点五。", "当前是版本1.5。", RULE_NUMBER_VERSION),
        ("版本二点一.三", "版本2.1.3", RULE_NUMBER_VERSION),
        ("版本二点一．三", "版本2.1.3", RULE_NUMBER_VERSION),
        ("端口二三四五九", "端口23459", RULE_NUMBER_IDENTIFIER),
        ("验证码一二-三四", "验证码12-34", RULE_NUMBER_IDENTIFIER),
        ("编号一二/三四", "编号12/34", RULE_NUMBER_IDENTIFIER),
        ("一百二十三个样本", "123个样本", RULE_NUMBER_COUNT),
        ("一百零二个样本", "102个样本", RULE_NUMBER_COUNT),
        ("一万零二个样本", "10002个样本", RULE_NUMBER_COUNT),
        ("两百个样本", "200个样本", RULE_NUMBER_COUNT),
        ("缩写是 S S O T", "缩写是 SSOT", RULE_ACRONYM_SSOT),
        ("缩写是 SOT", "缩写是 SSOT", RULE_ACRONYM_SSOT),
    ],
)
def test_final_policy_high_confidence_positive_cases(source, expected, rule_id):
    result = normalize_final_transcript(source)
    assert result.text == expected
    assert result.applied_rule_ids == (rule_id,)
    assert result.applied_count == 1


@pytest.mark.parametrize(
    "text",
    [
        "一点一点来",
        "第一性原理",
        "万一",
        "千方百计",
        "功利主义",
        "功力深厚",
        "这是一条功利",
        "这是一条功力",
        "功利原则",
        "Google Cloud Code",
        "SOT",
        "sot",
        "S-S-O-T",
        "两三个人",
    ],
)
def test_final_policy_preserves_non_numeric_or_ambiguous_language(text):
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    "text",
    [
        "这个版本一点一点来改",
        "这个版本一点五颜六色的元素都没有",
        "这个版本一点五彩斑斓的感觉都没有",
        "版本二点一.两",
        "版本二点一.三候选",
        "版本两三点四还没定",
        "端口两三个都能用",
        "验证码一二-两三",
        "验证码一二-三四个字符",
        "一百二个样本",
        "一万二个样本",
        "万一个文件坏了怎么办",
        "这是一条功利原则",
        "SOT_value",
        "SOT_1",
        "prefixSOTsuffix",
        "v2SOT3",
        "namespace.SOT_value",
        "X_SOT",
        "config/SOT.yaml",
        "README-SOT.md",
        "SOT＿value",
        "αSOTβ",
        "变量SOT值",
        "v2 的伦理框架强调这条功利。",
        "这是一条功利。下一段讨论体系设计。",
        "这是一条功利原则。另一句包含 V2。",
        "他的功力深厚。另一句包含 v2。",
        "第一性原理在前。这是一条功利原则。",
        "功利主义仍然有效。另一句包含 V2。",
        "版本两三点四还没定，端口两三个都能用。",
        "config.SOT_value 保留这个版本一点一点来改",
    ],
)
def test_final_policy_preserves_compositional_ambiguous_cases(text):
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    "text",
    [
        "这个版本一点五 颜六色的元素都没有",
        "这个版本一点五\t颜六色的元素都没有",
        "这个版本一点五   颜六色的元素都没有",
        "这个版本一点五\t \t彩斑斓的感觉都没有",
    ],
)
def test_r3_m1_horizontal_whitespace_does_not_end_bare_version_prose(text):
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("版本一点五。", "版本1.5。"),
        ("版本一点五\t", "版本1.5\t"),
        ("版本一点五\n下一句", "版本1.5\n下一句"),
    ],
)
def test_r3_m1_keeps_high_confidence_terminated_version_positives(source, expected):
    result = normalize_final_transcript(source)
    assert result.text == expected
    assert result.applied_rule_ids == (RULE_NUMBER_VERSION,)
    assert result.applied_count == 1


@pytest.mark.parametrize(
    "text",
    [
        "版本二点一 . alpha",
        "版本二点一 . 三A",
        "版本二点一 / 三",
        "版本二点一 .",
        "编号一二 - 两三",
        "编号一二 . alpha",
    ],
)
def test_r3_m2_spaced_compounds_are_preserved_whole_when_any_part_is_unsupported(text):
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    ("source", "expected", "rule_id"),
    [
        ("版本二点一 . 三", "版本2.1.3", RULE_NUMBER_VERSION),
        ("版本二点一 ． 三。", "版本2.1.3。", RULE_NUMBER_VERSION),
        ("编号一二 - 三四", "编号12-34", RULE_NUMBER_IDENTIFIER),
        ("编号一二 ／ 三四", "编号12／34", RULE_NUMBER_IDENTIFIER),
        ("验证码一二 . 三四。", "验证码12.34。", RULE_NUMBER_IDENTIFIER),
    ],
)
def test_r3_m2_spaced_compounds_normalize_only_when_every_component_is_supported(
    source, expected, rule_id
):
    result = normalize_final_transcript(source)
    assert result.text == expected
    assert result.applied_rule_ids == (rule_id,)
    assert result.applied_count == 1


@pytest.mark.parametrize(
    "text",
    [
        "const SOT = 1",
        "$SOT",
        "${SOT}",
        "SOT()",
        "SOT<T>",
        "[SOT]",
        "SOT―value",
        "SOT−value",
        "SOT﹒yaml",
        "SOT\u200dα",
        "config/SOT.yaml",
        "README-SOT.md",
        "SOT＿value",
        "αSOTβ",
        "SOT",
        "S S O T",
        "缩写是 SOT_value",
        "缩写是 SOT()",
        "缩写是 SOT<T>",
        "缩写是 SOT―value",
        "缩写是 SOT﹒yaml",
        "缩写是 SOT\u200dα",
    ],
)
def test_r3_m3_sot_is_preserved_outside_explicit_linguistic_alias_context(text):
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("缩写是 SOT", "缩写是 SSOT"),
        ("它是 SOT", "它是 SSOT"),
        ("这个缩写为 S S O T", "这个缩写为 SSOT"),
    ],
)
def test_r3_m3_sot_alias_requires_narrow_explicit_linguistic_context(source, expected):
    result = normalize_final_transcript(source)
    assert result.text == expected
    assert result.applied_rule_ids == (RULE_ACRONYM_SSOT,)
    assert result.applied_count == 1


@pytest.mark.parametrize(
    "text",
    [
        "cloud code",
        "Cloud Code",
        "云 code",
        "I use Cloud Code for Kubernetes.",
        "We run cloud code in Google Cloud.",
        "我们使用云 code 部署到 Google Cloud。",
    ],
)
def test_r3_m6_ambiguous_cloud_code_entities_are_preserved(text):
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    "text",
    [
        "npm install -g @anthropic-ai/claude-code",
        "cd claude-code",
        "claude-code.md",
        "~/.claude-code/config",
        "claudecode",
        "xclaude code",
        "克劳德code.md",
        "claude\ncode",
        "claude code --help",
        "claude code",
        "CLAUDE CODE",
        "claude code.md",
        "claude code/value",
        "claude code_1",
        "claude code2",
        "@claude code",
        "/claude code",
        ".claude code",
        "_claude code",
        "~claude code",
        "-claude code",
        "\\claude code",
        "claude code@",
        "claude code\\",
        "claude code~",
        "claude code-",
        "cd claude code",
        "docs/claude code/config",
        "使用 claude code:23459",
        "使用 claude code,config",
        "使用 claude code)",
        "使用 claude code::value",
    ],
)
def test_r4_f1_n2_claude_code_preserves_structural_and_nonlinguistic_tokens(text):
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("我使用 claude code 写程序", "我使用 Claude Code 写程序"),
        ("I use cloude code daily", "I use Claude Code daily"),
        ("使用 claude code。", "使用 Claude Code。"),
        ("请用克勞德code。", "请用Claude Code。"),
        ("克劳德code写程序", "Claude Code写程序"),
    ],
)
def test_r4_f1_keeps_narrow_natural_language_claude_code_positives(source, expected):
    result = normalize_final_transcript(source)
    assert result.text == expected
    assert result.applied_rule_ids == (RULE_CLAUDE_CODE,)
    assert result.applied_count == 1


@pytest.mark.parametrize(
    "text",
    [
        "缩写是 S S O T A",
        "这个缩写为 S S O T X",
    ],
)
def test_r4_n1_spaced_ssot_alias_preserves_longer_spelled_letter_sequences(text):
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


def test_r4_n1_spaced_ssot_alias_normalizes_complete_token_before_prose():
    result = normalize_final_transcript("缩写是 S S O T 的定义")
    assert result.text == "缩写是 SSOT 的定义"
    assert result.applied_rule_ids == (RULE_ACRONYM_SSOT,)
    assert result.applied_count == 1


@pytest.mark.parametrize(
    "text",
    [
        "使用 claude codeβ.py",
        "使用 claude codeβ/配置",
        "使用 claude code变量.py",
        "克劳德codeβ.py",
    ],
)
def test_r5_m1_claude_code_preserves_reviewer_unicode_structural_reproducers(text):
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    "continuation",
    [
        pytest.param("Ａ.py", id="unicode-letter"),
        pytest.param("\u0301.py", id="combining-mark"),
        pytest.param("١.py", id="unicode-number"),
        pytest.param("＿value", id="connector-punctuation"),
    ],
)
def test_r5_m1_claude_code_preserves_adjacent_unicode_token_matrix(continuation):
    text = f"使用 claude code{continuation}"
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    "continuation",
    [
        pytest.param("Ａ", id="fullwidth-latin"),
        pytest.param("β", id="greek"),
        pytest.param("Ж", id="cyrillic"),
        pytest.param("字", id="cjk"),
        pytest.param("A\u0301", id="decomposed-letter"),
    ],
)
def test_r5_m2_spaced_ssot_preserves_unicode_single_letter_continuations(
    continuation,
):
    text = f"这个缩写是 S S O T {continuation}"
    result = normalize_final_transcript(text)
    assert result.text == text
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("这个缩写是 S S O T", "这个缩写是 SSOT"),
        ("简称：SOT。", "简称：SSOT。"),
    ],
)
def test_r5_unicode_guards_keep_exact_ssot_alias_positives(source, expected):
    result = normalize_final_transcript(source)
    assert result.text == expected
    assert result.applied_rule_ids == (RULE_ACRONYM_SSOT,)
    assert result.applied_count == 1


@pytest.mark.parametrize(
    ("source", "expected", "rule_id"),
    [
        ("用克劳德 code 写程序", "用Claude Code 写程序", RULE_CLAUDE_CODE),
        ("用克劳得code写程序", "用Claude Code写程序", RULE_CLAUDE_CODE),
        ("CLAUD CODE", "Claude Code", RULE_CLAUDE_CODE),
    ],
)
def test_domain_aliases_require_clear_context(source, expected, rule_id):
    result = normalize_final_transcript(source)
    assert result.text == expected
    assert result.applied_rule_ids == (rule_id,)
    assert result.applied_count == 1


@pytest.mark.parametrize(
    "normalized",
    [
        "这里有4个字母",
        "第12次测试",
        "版本2.1",
        "端口23459",
        "123个样本",
        "SSOT",
        "Claude Code",
        "这是一条原理",
    ],
)
def test_final_policy_is_idempotent(normalized):
    result = normalize_final_transcript(normalized)
    assert result.text == normalized
    assert result.applied_rule_ids == ()
    assert result.applied_count == 0


def test_policy_reports_multiple_rules_without_transcript_metadata():
    result = TranscriptPolicy().normalize_final("版本二点一，缩写是 SOT")
    assert result.text == "版本2.1，缩写是 SSOT"
    assert result.applied_rule_ids == (RULE_NUMBER_VERSION, RULE_ACRONYM_SSOT)
    assert result.applied_count == 2


def test_ssot_canonicalization_is_local_and_compositional():
    source = "独立缩写是 SOT，路径 config/SOT.yaml，标识 αSOTβ，分字母缩写是 S S O T。"
    expected = "独立缩写是 SSOT，路径 config/SOT.yaml，标识 αSOTβ，分字母缩写是 SSOT。"

    result = normalize_final_transcript(source)

    assert result.text == expected
    assert result.applied_rule_ids == (RULE_ACRONYM_SSOT,)
    assert result.applied_count == 2
