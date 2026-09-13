"""Focused tests for model defaults and environment overrides."""

import pathlib
import sys


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_backend import config  # noqa: E402


def test_transcription_model_default_and_env_override(monkeypatch):
    monkeypatch.delenv("WHISPERWAVE_MODEL", raising=False)
    assert config._model_env(
        "WHISPERWAVE_MODEL", config.DEFAULT_WHISPER_TRANSCRIBE_MODEL
    ) == "gpt-4o-transcribe"

    monkeypatch.setenv("WHISPERWAVE_MODEL", " custom-transcribe ")
    assert config._model_env(
        "WHISPERWAVE_MODEL", config.DEFAULT_WHISPER_TRANSCRIBE_MODEL
    ) == "custom-transcribe"


def test_delay_is_model_specific():
    assert config.transcription_delay_for_model(
        "gpt-realtime-whisper", "high"
    ) == "high"
    assert config.transcription_delay_for_model(
        "gpt-4o-transcribe", "high"
    ) == ""
