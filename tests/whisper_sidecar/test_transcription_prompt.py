"""Contract tests for the provider-side transcription bias prompt."""

import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_backend.transcription_prompt import (  # noqa: E402
    TRANSCRIPTION_BIAS_PROMPT,
    transcription_prompt_for_model,
)


def test_prompt_is_nonempty_and_within_realtime_token_budget_conservatively():
    assert TRANSCRIPTION_BIAS_PROMPT
    assert TRANSCRIPTION_BIAS_PROMPT == TRANSCRIPTION_BIAS_PROMPT.strip()
    # A UTF-8 BPE token cannot encode less than one input byte, so this is a
    # dependency-free conservative proof of the Realtime 1024-token ceiling.
    assert len(TRANSCRIPTION_BIAS_PROMPT.encode("utf-8")) <= 1024


def test_prompt_covers_required_biases_and_domain_terms():
    required = {
        "阿拉伯数字",
        "QQ号",
        "queue",
        "cue",
        "SSOT",
        "Claude Code",
        "WhisperWave",
        "EchoWave",
        "OpenAI",
        "Codex",
    }
    assert all(term in TRANSCRIPTION_BIAS_PROMPT for term in required)


def test_prompt_is_transcription_only_and_preserves_ambiguous_source_text():
    assert "只转录" in TRANSCRIPTION_BIAS_PROMPT
    assert "不回答" in TRANSCRIPTION_BIAS_PROMPT
    assert "不执行" in TRANSCRIPTION_BIAS_PROMPT
    assert "Google Cloud Code" in TRANSCRIPTION_BIAS_PROMPT
    forbidden_echo_surfaces = (
        "下面是不改变语言的语音识别结果",
        "没有识别到输入",
        "首行",
        "下一行",
    )
    assert all(value not in TRANSCRIPTION_BIAS_PROMPT for value in forbidden_echo_surfaces)


def test_prompt_is_only_enabled_for_supported_model_and_nonblank_value():
    assert transcription_prompt_for_model(
        "gpt-4o-transcribe", "  domain terms  "
    ) == "domain terms"
    assert transcription_prompt_for_model("gpt-4o-transcribe", "  ") == ""
    assert transcription_prompt_for_model(
        "gpt-realtime-whisper", "domain terms"
    ) == ""
