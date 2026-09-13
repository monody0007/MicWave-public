"""Deterministic text policy for completed WhisperWave transcripts.

The active server streams provider deltas verbatim. Semantic normalization is
reserved for the authoritative ``transcription.completed`` text so a partial
never receives an irreversible edit. Ambiguous candidates retain provider text.
"""

from dataclasses import dataclass
import re
import unicodedata
from typing import Callable, Match, Optional, Pattern, Tuple


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
    """Final-only, deterministic normalization policy for completed text."""

    _RULES = (
        (RULE_NUMBER_VERSION, _apply_version),
        (RULE_NUMBER_IDENTIFIER, _apply_identifier),
        (RULE_NUMBER_ORDINAL, _apply_ordinal),
        (RULE_NUMBER_COUNT, _apply_count),
        (RULE_ACRONYM_SSOT, _apply_ssot),
        (RULE_CLAUDE_CODE, _apply_claude_code),
    )

    def normalize_final(self, text: str) -> TranscriptPolicyResult:
        if not text:
            return TranscriptPolicyResult(text=text, applied_rule_ids=(), applied_count=0)

        normalized = text
        rule_ids = []
        applied_count = 0
        for rule_id, apply_rule in self._RULES:
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
