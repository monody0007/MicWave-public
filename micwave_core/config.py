"""Configuration defaults and safe environment parsers for micwave-core.

Contains zero hardcoded secrets and performs no implicit ``.env`` file reads at
import time; callers inject credentials via environment variables or Secret
Manager loaders.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime-2.1-mini"
OPENAI_REALTIME_MODEL = (
    os.getenv("OPENAI_REALTIME_MODEL") or ""
).strip() or DEFAULT_OPENAI_REALTIME_MODEL

OPENAI_REALTIME_MODALITIES = os.getenv("OPENAI_REALTIME_MODALITIES", "text").split(",")


def _optional_int_in_range(name: str, low: int, high: int) -> Optional[int]:
    """Return int(env[name]) if set, parseable and within [low, high], else None."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r (not an integer); ignoring", name, raw)
        return None
    if not (low <= value <= high):
        logger.warning(
            "Invalid %s=%r (must be an integer in [%d, %d]); ignoring",
            name,
            raw,
            low,
            high,
        )
        return None
    return value


BRAINWAVE_MAX_OUTPUT_TOKENS = _optional_int_in_range(
    "BRAINWAVE_MAX_OUTPUT_TOKENS", 1, 4096
)

# ── WhisperWave transcription-session configuration ──
DEFAULT_WHISPER_TRANSCRIBE_MODEL = "gpt-4o-transcribe"


def _model_env(name: str, default: str) -> str:
    return (os.getenv(name) or "").strip() or default


WHISPER_TRANSCRIBE_MODEL = _model_env(
    "WHISPERWAVE_MODEL", DEFAULT_WHISPER_TRANSCRIBE_MODEL
)

WHISPERWAVE_MODALITIES = os.getenv("WHISPERWAVE_MODALITIES", "text").split(",")

_VALID_DELAYS = ("minimal", "low", "medium", "high", "xhigh")
_DELAY_CAPABLE_MODELS = frozenset({"gpt-realtime-whisper"})


def _delay_env(name: str, default: str = "low") -> str:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw not in _VALID_DELAYS:
        logger.warning(
            "Invalid %s=%r (must be one of %s); using %s",
            name,
            raw,
            list(_VALID_DELAYS),
            default,
        )
        return default
    return raw


WHISPERWAVE_DELAY = _delay_env("WHISPERWAVE_DELAY", "low")


def transcription_delay_for_model(model: str, delay: str) -> str:
    """Return a delay only when the selected model supports the field."""
    if model not in _DELAY_CAPABLE_MODELS or not isinstance(delay, str):
        return ""
    return delay.strip()


WHISPERWAVE_LANGUAGE = (os.getenv("WHISPERWAVE_LANGUAGE") or "").strip()

_VALID_NOISE = ("near_field", "far_field")


def _noise_env(name: str) -> str:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return ""
    if raw not in _VALID_NOISE:
        logger.warning(
            "Invalid %s=%r (must be near_field/far_field); disabling", name, raw
        )
        return ""
    return raw


WHISPERWAVE_NOISE_REDUCTION = _noise_env("WHISPERWAVE_NOISE_REDUCTION")
WHISPERWAVE_INPUT_SAMPLE_RATE = 24000
