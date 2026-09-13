import logging
import os

logger = logging.getLogger(__name__)

# ── WhisperWave: pure Realtime transcription session (task 0436 Phase 2) ──
# The whole point of this prototype: a `type:"transcription"` Realtime session
# backed by a transcription-only model. `gpt-4o-transcribe` accepts a vocabulary
# bias prompt inside its transcription config, while the session still has no
# assistant answer surface, response marker, or answer guard.
DEFAULT_WHISPER_TRANSCRIBE_MODEL = "gpt-4o-transcribe"


def _model_env(name: str, default: str) -> str:
    return (os.getenv(name) or "").strip() or default


WHISPER_TRANSCRIBE_MODEL = _model_env(
    "WHISPERWAVE_MODEL", DEFAULT_WHISPER_TRANSCRIBE_MODEL
)

# Kept for menubar↔server protocol compatibility; unused by a transcription
# session (which has no output_modalities face).
OPENAI_REALTIME_MODALITIES = os.getenv("WHISPERWAVE_MODALITIES", "text").split(",")

# Streaming transcription delay tier for gpt-realtime-whisper. The live API
# rejects this field for gpt-4o-transcribe, so the client applies it only to a
# compatible model. Invalid configured values still fall back to "low".
_VALID_DELAYS = ("minimal", "low", "medium", "high", "xhigh")
_DELAY_CAPABLE_MODELS = frozenset({"gpt-realtime-whisper"})


def _delay_env(name: str, default: str = "low") -> str:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    if raw not in _VALID_DELAYS:
        logger.warning(
            "Invalid %s=%r (must be one of %s); using %s",
            name, raw, list(_VALID_DELAYS), default,
        )
        return default
    return raw


WHISPERWAVE_DELAY = _delay_env("WHISPERWAVE_DELAY", "low")


def transcription_delay_for_model(model: str, delay: str) -> str:
    """Return a delay only when the selected model supports the field."""

    if model not in _DELAY_CAPABLE_MODELS or not isinstance(delay, str):
        return ""
    return delay.strip()

# Optional language hint (e.g. "en", "zh"). Empty by default so CN/EN
# code-switching is not forced to a single language.
WHISPERWAVE_LANGUAGE = (os.getenv("WHISPERWAVE_LANGUAGE") or "").strip()

# Optional input noise reduction: near_field / far_field. Empty = off (null).
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

# Realtime session sample rate for input audio (Hz).
WHISPERWAVE_INPUT_SAMPLE_RATE = 24000
