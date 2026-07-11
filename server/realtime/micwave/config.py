import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Global configuration for OpenAI realtime model
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-mini-2025-12-15")

# Modalities for realtime sessions (text-only output by default)
OPENAI_REALTIME_MODALITIES = os.getenv("OPENAI_REALTIME_MODALITIES", "text").split(",")


def _optional_int_in_range(name: str, low: int, high: int):
    """Return int(env[name]) if set, parseable and within [low, high], else None.

    None means the field is omitted from the API payload. An unparseable or
    out-of-range value is ignored with a warning rather than crashing at import
    (task 0436 M7).
    """
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
            name, raw, low, high,
        )
        return None
    return value


# Optional realtime output cap (task 0436 A2/M7). Left unset by default so no
# field is sent to the API — this preserves today's behaviour of relying on the
# API defaults. GA Realtime has no `temperature` field (removed with the beta
# interface on 2026-05-12), so only max_output_tokens is exposed; the GA schema
# constrains it to an integer in [1, 4096].
BRAINWAVE_MAX_OUTPUT_TOKENS = _optional_int_in_range("BRAINWAVE_MAX_OUTPUT_TOKENS", 1, 4096)
