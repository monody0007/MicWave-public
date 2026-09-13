"""Assembly helper for OpenAI Realtime transcription deltas."""


def append_transcript_delta(current: str, delta: str) -> str:
    """Append one provider delta exactly as newly available transcript text."""

    return current + delta
