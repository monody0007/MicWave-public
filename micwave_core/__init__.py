"""Authoritative headless voice transcription core (micwave-core).

Exposes audio preprocessing (100Hz HPF + gated dynamic AGC + pause splitting),
OpenAI Realtime conversation & Whisper transcription WebSocket clients,
transcript merging/normalization/answer-guard utilities, JSONL transcript store,
and FastAPI factories for WebSocket realtime and HTTP phone batch lanes.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .audio_gain import (
    SpeechGainStats,
    StreamingSpeechAGC,
    measure_speech_activity_pcm16,
    preprocess_speech_pcm16,
    speech_activity_secs,
    split_pcm_on_pauses,
)
from .audio_persistence import TurnAudioCache, WhisperTurnAudioCache
from .audio_spool import AudioSpoolError, TurnAudioSpool
from .config import (
    DEFAULT_OPENAI_REALTIME_MODEL,
    DEFAULT_WHISPER_TRANSCRIBE_MODEL,
    OPENAI_REALTIME_MODALITIES,
    OPENAI_REALTIME_MODEL,
    WHISPER_TRANSCRIBE_MODEL,
)
from .openai_realtime_client import (
    CommitNotAcknowledgedError,
    OpenAIRealtimeAudioTextClient,
    TurnIsolationBrokenError,
)
from .prompts import (
    PROMPTS,
    TRANSCRIPTION_BIAS_PROMPT,
    get_optimize_prompt,
    get_realtime_prompt,
    transcription_prompt_for_model,
)
from .realtime_client_base import RealtimeClientBase
from .transcript_merge import join_segment_texts, merge_incremental_text
from .transcript_store import TranscriptStoreReader, append_record, history_file_path

__all__ = [
    "__version__",
    "AudioSpoolError",
    "CommitNotAcknowledgedError",
    "DEFAULT_OPENAI_REALTIME_MODEL",
    "DEFAULT_WHISPER_TRANSCRIBE_MODEL",
    "OPENAI_REALTIME_MODALITIES",
    "OPENAI_REALTIME_MODEL",
    "OpenAIRealtimeAudioTextClient",
    "PROMPTS",
    "RealtimeClientBase",
    "SpeechGainStats",
    "StreamingSpeechAGC",
    "TRANSCRIPTION_BIAS_PROMPT",
    "TranscriptStoreReader",
    "TurnAudioCache",
    "TurnAudioSpool",
    "TurnIsolationBrokenError",
    "WHISPER_TRANSCRIBE_MODEL",
    "WhisperTurnAudioCache",
    "append_record",
    "get_optimize_prompt",
    "get_realtime_prompt",
    "history_file_path",
    "join_segment_texts",
    "measure_speech_activity_pcm16",
    "merge_incremental_text",
    "preprocess_speech_pcm16",
    "speech_activity_secs",
    "split_pcm_on_pauses",
    "transcription_prompt_for_model",
]
