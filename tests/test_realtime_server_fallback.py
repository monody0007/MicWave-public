import asyncio
import json
import logging
import pathlib
import sys
import time

import numpy as np
import pytest
import scipy.signal
from starlette.websockets import WebSocketState

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai_realtime_client import OpenAIRealtimeAudioTextClient
from prompts import get_optimize_prompt, get_realtime_prompt
from realtime_server import TurnSession, TurnSessionConfig
from realtime_text_utils import (
    NO_SPEECH_PLACEHOLDER,
    emitted_novel_material_ratio,
    normalize_for_similarity,
    parse_ratio_env,
    transcription_similarity_ratio,
)

MARKER = "下面是不改变语言的语音识别结果：\n\n"


def payload_types(session):
    """Ordered list of (type, isNewResponse/status) tuples for ordering asserts."""
    out = []
    for p in session._websocket.sent_payloads:
        if p.get("type") == "text" and p.get("isNewResponse") is True:
            out.append("replacement")
        elif p.get("type") == "status":
            out.append(f"status:{p.get('status')}")
    return out


async def drive_marker_answer(session, answer: str):
    """Event-driven: push the marker then an answer through the real delta
    handler so _emitted_text / homonym corrector state is built the same way the
    live pipeline builds it (no direct _marker_seen/_emitted_text injection)."""
    await session._on_text_delta({"delta": MARKER})
    for ch in answer:
        await session._on_text_delta({"delta": ch})


class FakeWebSocket:
    client_state = WebSocketState.CONNECTED

    def __init__(self):
        self.sent_payloads = []

    async def send_text(self, text):
        self.sent_payloads.append(json.loads(text))


def make_config(**overrides):
    values = {
        "keep_provider_session": True,
        "provider_session_max_turns": 8,
        "provider_session_max_age_sec": 7200,
        "provider_init_max_attempts": 3,
        "provider_init_retry_delay_sec": 0.5,
        "response_finalize_timeout_sec": 120.0,
        "input_transcript_grace_sec": 0.05,
        "suspicious_marker_audio_sec": 1.0,
        "suspicious_marker_emitted_chars": 60,
        "suspicious_input_transcript_grace_sec": 0.5,
        "input_transcript_replacement_min_ratio": 1.5,
        "input_transcript_replacement_min_delta_chars": 20,
        "default_source_sample_rate": 48000,
        "marker_prefix": "下面是不改变语言的语音识别结果：\n\n",
        "max_prefix_deltas": 20,
        "transcription_failure_rotate_threshold": 2,
        "passthrough_without_marker": False,
        "answer_guard_min_similarity": 0.60,
        "answer_guard_novel_material_ratio": 0.55,
        "answer_guard_grace_sec": 0.0,
        "similarity_hard_cap_chars": 6000,
        "no_speech_guard_enabled": True,
        "no_speech_floor_margin_db": 4.0,
        "no_speech_min_active_dbfs": -55.0,
        "no_speech_peak_floor_dbfs": -55.0,
        "no_speech_min_run_frames": 3,
        "no_speech_hb300_min_ratio": 0.0,
        "no_speech_nearfield_p90_dbfs": -26.0,
    }
    values.update(overrides)
    return TurnSessionConfig(**values)


def make_session(**config_overrides):
    session = TurnSession(FakeWebSocket(), make_config(**config_overrides))
    session._active_turn_id = 111
    return session


def set_processed_audio_duration(session, seconds):
    session._processed_audio_bytes = int(
        seconds * session._audio_processor.target_sample_rate * 2
    )


# ── S1 dispatcher harness: drive the REAL serial provider dispatcher ─────────
# The S1 finalize barrier must be exercised through
# OpenAIRealtimeAudioTextClient.receive_messages() (the real serial dispatcher),
# not by calling handlers directly, because the deadlock the fix removes only
# exists in that serial loop (task 0436 S1 R4).

_STOP = object()


class FakeProviderWS:
    """Async-iterable provider WebSocket backed by an asyncio.Queue.

    receive_messages() does ``async for message in self.ws`` and json.loads each
    frame, so we yield queued JSON strings in order — exactly how frames arrive
    over the wire.
    """

    def __init__(self):
        self._queue = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def push(self, frame: dict):
        await self._queue.put(json.dumps(frame, ensure_ascii=False))

    async def send(self, data):
        # A real provider WS accepts sends (e.g. input_audio_buffer.clear on
        # finalize); record and no-op so finalize's buffer clear doesn't warn.
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._queue.get()
        if item is _STOP:
            raise StopAsyncIteration
        return item


def make_dispatcher_session(**config_overrides):
    """A session wired to a real client whose provider dispatcher we can drive."""
    session = make_session(**config_overrides)
    client = OpenAIRealtimeAudioTextClient(api_key="k", model="m")
    ws = FakeProviderWS()
    client.ws = ws
    client.register_handler("default", client.default_handler)
    session._register_client_handlers(client)
    session._client = client
    session._openai_ready.set()
    return session, client, ws


async def _wait_until(predicate, timeout=2.0):
    deadline = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > deadline:
            raise AssertionError("condition not met within timeout")
        await asyncio.sleep(0.005)


async def drain_finalize(session, timeout=2.0):
    """Wait for the scheduled finalize barrier task to be created and finish."""
    await _wait_until(lambda: session._finalize_task is not None, timeout=timeout)
    try:
        await asyncio.wait_for(asyncio.shield(session._finalize_task), timeout=timeout)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        pass


async def stop_dispatcher(task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def drive_marker_answer_via_dispatcher(ws, answer: str):
    await ws.push({"type": "response.text.delta", "delta": MARKER})
    for ch in answer:
        await ws.push({"type": "response.text.delta", "delta": ch})


def sent_text_payloads(session):
    return [
        payload
        for payload in session._websocket.sent_payloads
        if payload.get("type") == "text"
    ]


async def finalize_from_response_done(session):
    await session._on_response_done({})
    await drain_finalize(session)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body_deltas",
    [
        [NO_SPEECH_PLACEHOLDER + "真实内容"],
        ["（没有识", "别到输入）。 \n真实内容"],
    ],
)
async def test_no_speech_prefix_with_real_content_is_removed_across_deltas(
    body_deltas,
):
    session = make_session()
    await session._on_text_delta({"delta": MARKER})
    for delta in body_deltas:
        await session._on_text_delta({"delta": delta})
    await finalize_from_response_done(session)

    assert session._emitted_text == "真实内容"
    assert NO_SPEECH_PLACEHOLDER not in "".join(
        payload["content"] for payload in sent_text_payloads(session)
    )


@pytest.mark.asyncio
async def test_no_speech_only_with_empty_asr_emits_one_canonical_placeholder(caplog):
    session = make_session()
    set_processed_audio_duration(session, 0.25)
    await session._on_text_delta({"delta": MARKER + NO_SPEECH_PLACEHOLDER})
    with caplog.at_level(logging.WARNING):
        await finalize_from_response_done(session)

    assert sent_text_payloads(session) == [
        {
            "type": "text",
            "content": NO_SPEECH_PLACEHOLDER,
            "isNewResponse": False,
            "turn_id": 111,
        }
    ]
    assert [payload["type"] for payload in session._websocket.sent_payloads] == [
        "text",
        "status",
    ]
    assert any(
        "audio_sec=0.25 processed_audio_bytes=12000" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_no_speech_only_with_asr_uses_fallback_without_placeholder():
    session = make_session()
    faithful = "这是输入侧 ASR 的忠实转写。"
    await session._on_text_delta({"delta": MARKER + NO_SPEECH_PLACEHOLDER})
    await session._on_input_transcription_completed({"transcript": faithful})
    await finalize_from_response_done(session)

    assert sent_text_payloads(session) == [
        {
            "type": "text",
            "content": faithful,
            "isNewResponse": True,
            "turn_id": 111,
        }
    ]
    assert session._emitted_text == faithful


@pytest.mark.asyncio
async def test_no_speech_guard_normal_text_has_no_extra_prefix_buffering():
    session = make_session()
    content = "正常内容不以括号开头，并且应按现有 delta 流直接进入校正器。"
    await session._on_text_delta({"delta": MARKER})
    await session._on_text_delta({"delta": content})

    assert session._no_speech_prefix_guard._buffer == ""
    assert session._current_emitted_view() == content
    await finalize_from_response_done(session)
    assert session._emitted_text == content


@pytest.mark.asyncio
async def test_no_speech_guard_preserves_legal_parenthesized_body():
    session = make_session()
    content = "（一）今天开会"
    await session._on_text_delta({"delta": MARKER + "（一"})
    await session._on_text_delta({"delta": "）今天开会"})
    await finalize_from_response_done(session)
    assert session._emitted_text == content


@pytest.mark.asyncio
async def test_marker_body_discards_extra_leading_whitespace_before_emit():
    session = make_session()
    await session._on_text_delta({"delta": MARKER + "  \n"})
    await session._on_text_delta({"delta": "\n真实正文"})
    await session._on_text_delta({"delta": "\n\n第二段"})
    await finalize_from_response_done(session)
    assert session._emitted_text == "真实正文\n\n第二段"
    assert "".join(
        payload["content"] for payload in sent_text_payloads(session)
    ) == "真实正文\n\n第二段"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "placeholder",
    [
        "（没有识别到语音输入）",
        "（未识别到有效语音输入）",
        "(没有识别到输入)",
    ],
)
async def test_no_speech_guard_filters_model_wording_variants(placeholder):
    session = make_session()
    await session._on_text_delta({"delta": MARKER + placeholder + " 后续正文"})
    await finalize_from_response_done(session)
    assert session._emitted_text == "后续正文"


@pytest.mark.asyncio
async def test_no_speech_guard_does_not_emit_on_unsuccessful_finalize():
    session = make_session()
    await session._on_text_delta({"delta": MARKER + NO_SPEECH_PLACEHOLDER})
    await session._finalize_turn("error")
    assert sent_text_payloads(session) == []


@pytest.mark.asyncio
async def test_no_speech_guard_disabled_restores_placeholder_passthrough():
    session = make_session(no_speech_guard_enabled=False)
    await session._on_text_delta({"delta": MARKER + NO_SPEECH_PLACEHOLDER})
    await session._finalize_turn("response.done")
    assert sent_text_payloads(session) == [
        {
            "type": "text",
            "content": NO_SPEECH_PLACEHOLDER,
            "isNewResponse": False,
            "turn_id": 111,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "（",
        "（" + "一" * 21 + "）今天开会",
    ],
)
async def test_no_speech_guard_response_end_preserves_partial_or_over_limit_body(
    content,
):
    session = make_session()
    await session._on_text_delta({"delta": MARKER + content})
    await session._finalize_turn("response.done")
    assert "".join(payload["content"] for payload in sent_text_payloads(session)) == content


@pytest.mark.asyncio
async def test_finalize_replaces_leaked_placeholder_variant_with_canonical_text():
    session = make_session()
    session._marker_seen = True
    session._emitted_text = "(未识别到有效语音输入). "
    await session._finalize_turn("response.done")
    assert sent_text_payloads(session) == [
        {
            "type": "text",
            "content": NO_SPEECH_PLACEHOLDER,
            "isNewResponse": True,
            "turn_id": 111,
        }
    ]


@pytest.mark.asyncio
async def test_finalize_forces_one_replacement_for_bypassed_signal_verdict(
    caplog,
):
    session = make_session()
    await session._send_text_payload("2021年")
    session._turn_signal_verdict = "no_speech_signal"

    with caplog.at_level(logging.WARNING):
        await session._finalize_turn("response.done")

    replacements = [
        payload
        for payload in sent_text_payloads(session)
        if payload["isNewResponse"] is True
    ]
    assert replacements == [
        {
            "type": "text",
            "content": NO_SPEECH_PLACEHOLDER,
            "isNewResponse": True,
            "turn_id": 111,
        }
    ]
    assert session._emitted_text == NO_SPEECH_PLACEHOLDER
    assert sum(
        "Intentional-speech verdict forced canonical replacement"
        in record.message
        for record in caplog.records
    ) == 1


@pytest.mark.asyncio
async def test_leaked_placeholder_with_asr_still_prioritizes_fallback_replacement():
    session = make_session(answer_guard_min_similarity=0.0)
    faithful = "输入侧 ASR 有真实内容。"
    session._marker_seen = True
    session._emitted_text = "（未识别到语音输入）"
    await session._on_input_transcription_completed({"transcript": faithful})
    await session._apply_input_transcription_fallback("response.done")
    assert sent_text_payloads(session) == [
        {
            "type": "text",
            "content": faithful,
            "isNewResponse": True,
            "turn_id": 111,
        }
    ]


def test_no_speech_guard_env_defaults_on_and_zero_disables(monkeypatch):
    monkeypatch.delenv("BRAINWAVE_NO_SPEECH_GUARD", raising=False)
    monkeypatch.delenv("BRAINWAVE_NO_SPEECH_FLOOR_MARGIN_DB", raising=False)
    monkeypatch.delenv("BRAINWAVE_NO_SPEECH_MIN_ACTIVE_DBFS", raising=False)
    monkeypatch.delenv("BRAINWAVE_NO_SPEECH_PEAK_FLOOR_DBFS", raising=False)
    monkeypatch.delenv("BRAINWAVE_NO_SPEECH_MIN_RUN_FRAMES", raising=False)
    monkeypatch.delenv("BRAINWAVE_NO_SPEECH_HB300_MIN_RATIO", raising=False)
    monkeypatch.delenv("BRAINWAVE_NO_SPEECH_NEARFIELD_P90_DBFS", raising=False)
    default_config = TurnSessionConfig.from_env()
    assert default_config.no_speech_guard_enabled is True
    assert default_config.no_speech_floor_margin_db == 4.0
    assert default_config.no_speech_min_active_dbfs == -55.0
    assert default_config.no_speech_peak_floor_dbfs == -55.0
    assert default_config.no_speech_min_run_frames == 3
    assert default_config.no_speech_hb300_min_ratio == 0.0
    assert default_config.no_speech_nearfield_p90_dbfs == -26.0

    monkeypatch.setenv("BRAINWAVE_NO_SPEECH_GUARD", "0")
    monkeypatch.setenv("BRAINWAVE_NO_SPEECH_FLOOR_MARGIN_DB", "12.5")
    monkeypatch.setenv("BRAINWAVE_NO_SPEECH_MIN_ACTIVE_DBFS", "-48.0")
    monkeypatch.setenv("BRAINWAVE_NO_SPEECH_PEAK_FLOOR_DBFS", "-53.0")
    monkeypatch.setenv("BRAINWAVE_NO_SPEECH_MIN_RUN_FRAMES", "4")
    monkeypatch.setenv("BRAINWAVE_NO_SPEECH_HB300_MIN_RATIO", "0.31")
    monkeypatch.setenv("BRAINWAVE_NO_SPEECH_NEARFIELD_P90_DBFS", "-24.5")
    custom_config = TurnSessionConfig.from_env()
    assert custom_config.no_speech_guard_enabled is False
    assert custom_config.no_speech_floor_margin_db == 12.5
    assert custom_config.no_speech_min_active_dbfs == -48.0
    assert custom_config.no_speech_peak_floor_dbfs == -53.0
    assert custom_config.no_speech_min_run_frames == 4
    assert custom_config.no_speech_hb300_min_ratio == 0.31
    assert custom_config.no_speech_nearfield_p90_dbfs == -24.5


# ── W2: deterministic processed-PCM digital-silence gate ───────────────────


class _RecordingAudioCache:
    def __init__(self):
        self.started_turns = []
        self.accumulated = []
        self.enqueued = []

    def start_turn(self, turn_id):
        self.started_turns.append(turn_id)

    def accumulate(self, pcm_bytes):
        self.accumulated.append(pcm_bytes)

    def enqueue_turn(self, turn_id, outcome, sample_rate, channels=1):
        self.enqueued.append(
            {
                "turn_id": turn_id,
                "outcome": outcome,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )
        return True


class _RecordingProviderClient:
    def __init__(self):
        self.sent_audio = []
        self.commit_calls = 0
        self.start_response_calls = 0
        self.clear_audio_buffer_calls = 0
        self.refresh_session_calls = 0
        self.close_calls = 0
        self._open = True

    async def send_audio(self, chunk):
        self.sent_audio.append(chunk)

    async def commit_audio(self):
        self.commit_calls += 1

    async def start_response(self, instructions=None):
        self.start_response_calls += 1

    async def clear_audio_buffer(self):
        self.clear_audio_buffer_calls += 1

    async def refresh_session(self, modalities=None, instructions=None):
        self.refresh_session_calls += 1

    async def close(self):
        self.close_calls += 1
        self._open = False

    def _is_ws_open(self):
        return self._open


def make_recording_gate_session(**config_overrides):
    session = make_session(**config_overrides)
    session._audio_processor.set_source_sample_rate(24000)  # processed PCM is byte-stable
    client = _RecordingProviderClient()
    audio_cache = _RecordingAudioCache()
    session._client = client
    session._active_model = "m"
    session._provider_session_turns = 1
    session._provider_session_started_at = time.time()
    session._openai_ready.set()
    session._is_recording = True
    session._audio_cache = audio_cache
    audio_cache.start_turn(session._active_turn_id)
    return session, client, audio_cache


FRAME_SAMPLES = 720
SAMPLE_RATE = 24000


def pcm_sine_rms_frames(frame_levels_dbfs, trailing_samples=0):
    """Deterministic 1 kHz carrier with one requested RMS level per 30 ms frame."""

    frame_axis = np.arange(FRAME_SAMPLES, dtype=np.float64)
    carrier = np.sin(2.0 * np.pi * 1000.0 * frame_axis / SAMPLE_RATE)
    carrier_rms = float(np.sqrt(np.mean(np.square(carrier))))
    frames = []
    for level_dbfs in frame_levels_dbfs:
        target_rms = 32768.0 * (10.0 ** (level_dbfs / 20.0))
        frame = np.rint(carrier * (target_rms / carrier_rms))
        frames.append(np.clip(frame, -32768, 32767).astype(np.int16))
    if trailing_samples:
        if not frames:
            raise ValueError("trailing samples require at least one frame level")
        frames.append(frames[-1][:trailing_samples])
    return np.concatenate(frames).astype(np.int16).tobytes()


def pcm_speech_proxy(floor_dbfs, burst_dbfs, trailing_samples=0):
    return pcm_sine_rms_frames(
        [floor_dbfs] * 10 + [burst_dbfs] * 3 + [floor_dbfs] * 10,
        trailing_samples=trailing_samples,
    )


def pcm_dither(amplitude=2, frames=4):
    samples = np.resize(
        np.array([-amplitude, amplitude], dtype=np.int16),
        FRAME_SAMPLES * frames,
    )
    return samples.tobytes()


def pcm_frames_with_rms_levels(frames, levels_dbfs):
    scaled_frames = []
    for frame, level_dbfs in zip(frames, levels_dbfs):
        frame = np.asarray(frame, dtype=np.float64)
        frame_rms = float(np.sqrt(np.mean(np.square(frame))))
        if frame_rms == 0.0:
            raise ValueError("fixture frame must contain non-zero signal")
        target_rms = 32768.0 * (10.0 ** (level_dbfs / 20.0))
        scaled = np.rint(frame * (target_rms / frame_rms))
        scaled_frames.append(
            np.clip(scaled, -32768, 32767).astype(np.int16)
        )
    return np.concatenate(scaled_frames).tobytes()


def pcm_low_frequency_profile(active_dbfs):
    """Causal 100-250 Hz residual; its finite roll-off gives ~12% HB300."""

    levels_dbfs = [-60.0] * 10 + [active_dbfs] * 24 + [-60.0] * 10
    rng = np.random.default_rng(5)
    white = rng.standard_normal(FRAME_SAMPLES * len(levels_dbfs))
    bandpass = scipy.signal.butter(
        2,
        [100.0, 250.0],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )
    filtered = scipy.signal.sosfilt(bandpass, white).reshape(
        -1, FRAME_SAMPLES
    )
    return pcm_frames_with_rms_levels(filtered, levels_dbfs)


def pcm_intentional_speech_profile(active_dbfs):
    frame_axis = np.arange(FRAME_SAMPLES, dtype=np.float64)
    phase = 2.0 * np.pi * frame_axis / SAMPLE_RATE
    floor_frame = np.sin(150.0 * phase)
    voiced_frame = (
        np.sin(150.0 * phase)
        + 0.35 * np.sin(300.0 * phase)
        + 0.20 * np.sin(450.0 * phase)
    )
    rng = np.random.default_rng(42)
    friction = rng.standard_normal(24 * FRAME_SAMPLES)
    friction_bandpass = scipy.signal.butter(
        4,
        [3500.0, 4500.0],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )
    friction_frames = scipy.signal.sosfiltfilt(
        friction_bandpass, friction
    ).reshape(24, FRAME_SAMPLES)
    friction_frames /= np.sqrt(
        np.mean(np.square(friction_frames), axis=1)
    )[:, None]
    active_frames = [
        voiced_frame + 0.50 * friction_frame
        for friction_frame in friction_frames
    ]
    frames = (
        [floor_frame] * 10
        + active_frames
        + [floor_frame] * 10
    )
    levels_dbfs = [-65.0] * 10 + [active_dbfs] * 24 + [-65.0] * 10
    return pcm_frames_with_rms_levels(frames, levels_dbfs)


def pcm_w6_speech_policy_profile(
    *,
    floor_frames,
    active_frames,
    floor_dbfs,
    active_dbfs,
    hb300_ratio,
):
    """Deterministic two-band fixture for W6 policy boundaries."""

    frame_axis = np.arange(FRAME_SAMPLES, dtype=np.float64)
    phase = 2.0 * np.pi * frame_axis / SAMPLE_RATE
    floor_frame = np.sin(200.0 * phase)
    high_band_amplitude = float(
        np.sqrt(hb300_ratio / (1.0 - hb300_ratio))
    )
    active_frame = floor_frame + high_band_amplitude * np.sin(400.0 * phase)
    frames = (
        [floor_frame] * floor_frames
        + [active_frame] * active_frames
        + [floor_frame] * floor_frames
    )
    levels_dbfs = (
        [floor_dbfs] * floor_frames
        + [active_dbfs] * active_frames
        + [floor_dbfs] * floor_frames
    )
    return pcm_frames_with_rms_levels(frames, levels_dbfs)


def pcm_w6_quiet_speech_profile():
    return pcm_w6_speech_policy_profile(
        floor_frames=10,
        active_frames=12,
        floor_dbfs=-57.0,
        active_dbfs=-51.0,
        hb300_ratio=0.05,
    )


def pcm_w6_low_register_speech_profile():
    return pcm_w6_speech_policy_profile(
        floor_frames=4,
        active_frames=28,
        floor_dbfs=-65.0,
        active_dbfs=-32.0,
        hb300_ratio=0.19,
    )


def pcm_w6_extremely_quiet_breath_profile():
    """Sustained floor+5 dB noise with no speech structure."""

    floor_frames = 10
    active_frames = 16
    levels_dbfs = (
        [-57.0] * floor_frames
        + [-52.0] * active_frames
        + [-57.0] * floor_frames
    )
    rng = np.random.default_rng(2026072101)
    frames = rng.standard_normal((len(levels_dbfs), FRAME_SAMPLES))
    return pcm_frames_with_rms_levels(frames, levels_dbfs)


def pcm_keyboard_transient_profile():
    rng = np.random.default_rng(20260721)
    levels_dbfs = [-70.0] * 30
    # Three isolated events, none longer than two contiguous 30 ms frames.
    for frame_index in (4, 5, 12, 20, 21):
        levels_dbfs[frame_index] = -30.0
    frames = rng.standard_normal((len(levels_dbfs), FRAME_SAMPLES))
    return pcm_frames_with_rms_levels(frames, levels_dbfs)


def assert_normal_provider_path(session, client, audio_cache, pcm):
    assert b"".join(client.sent_audio) == pcm
    assert b"".join(audio_cache.accumulated) == pcm
    assert client.commit_calls == 1
    assert client.start_response_calls == 1
    assert sent_text_payloads(session) == []
    assert audio_cache.enqueued == []


def assert_turn_signal_log(caplog, verdict):
    metric_logs = [
        record.message
        for record in caplog.records
        if record.levelno == logging.INFO
        and record.message.startswith("turn signal metrics: ")
    ]
    assert len(metric_logs) == 1
    metric_log = metric_logs[0]
    for field in (
        "peak=",
        "floor=",
        "thr=",
        "active=",
        "maxrun=",
        "hb300w=",
        "p90act=",
        f"verdict={verdict}",
        "audio_sec=",
    ):
        assert field in metric_log
    return metric_log


def assert_no_speech_signal_result(session, client, audio_cache, caplog, pcm):
    assert b"".join(client.sent_audio) == pcm
    assert b"".join(audio_cache.accumulated) == pcm
    assert client.commit_calls == 0
    assert client.start_response_calls == 0
    assert client.clear_audio_buffer_calls == 1
    assert client.close_calls == 0
    assert session._client is client
    assert sum(
        payload.get("content") == NO_SPEECH_PLACEHOLDER
        for payload in sent_text_payloads(session)
    ) == 1
    assert session._websocket.sent_payloads[-1] == {
        "type": "status",
        "status": "idle",
        "turn_id": 111,
    }
    assert audio_cache.enqueued == [
        {
            "turn_id": 111,
            "outcome": "no_speech_signal",
            "sample_rate": 24000,
            "channels": 1,
        }
    ]
    warning = next(
        record.message
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "No speech signal detected on stop" in record.message
    )
    for metric_name in (
        "peak_dbfs=",
        "noise_floor_dbfs=",
        "active_threshold_dbfs=",
        "active_frames=",
        "total_frames=",
        "max_run_frames=",
        "hb300_weighted_ratio=",
        "active_p90_dbfs=",
        "audio_sec=",
    ):
        assert metric_name in warning
    assert_turn_signal_log(caplog, "no_speech_signal")
    assert not any(
        "No speech recognized on successful finalize" in record.message
        or "Suspicious short transcription on finalize" in record.message
        for record in caplog.records
    )


def assert_digital_silence_result(
    session, client, audio_cache, caplog, expected_audio_bytes
):
    assert client.commit_calls == 0
    assert client.start_response_calls == 0
    assert client.clear_audio_buffer_calls == 1
    assert client.close_calls == 0
    assert session._client is client
    assert session._websocket.sent_payloads == [
        {
            "type": "text",
            "content": NO_SPEECH_PLACEHOLDER,
            "isNewResponse": False,
            "turn_id": 111,
        },
        {"type": "status", "status": "idle", "turn_id": 111},
    ]
    assert audio_cache.enqueued == [
        {
            "turn_id": 111,
            "outcome": "digital_silence",
            "sample_rate": 24000,
            "channels": 1,
        }
    ]
    expected_audio_sec = expected_audio_bytes / (24000 * 2)
    assert any(
        record.levelno == logging.WARNING
        and "Digital silence detected on stop" in record.message
        and f"audio_sec={expected_audio_sec:.2f}" in record.message
        and f"processed_audio_bytes={expected_audio_bytes}" in record.message
        for record in caplog.records
    )
    assert_turn_signal_log(caplog, "digital_silence")
    assert not any(
        "No speech signal detected on stop" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_all_zero_processed_pcm_skips_provider_and_finalizes_digital_silence(
    caplog,
):
    session, client, audio_cache = make_recording_gate_session()
    zero_pcm = b"\x00\x00" * 1200
    await session._handle_audio_bytes(zero_pcm)

    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert client.sent_audio == [zero_pcm]
    assert_digital_silence_result(
        session, client, audio_cache, caplog, expected_audio_bytes=len(zero_pcm)
    )


@pytest.mark.asyncio
async def test_zero_byte_turn_skips_provider_and_finalizes_digital_silence(caplog):
    session, client, audio_cache = make_recording_gate_session()

    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert client.sent_audio == []
    assert_digital_silence_result(
        session, client, audio_cache, caplog, expected_audio_bytes=0
    )


@pytest.mark.asyncio
async def test_mixed_zero_and_nonzero_processed_pcm_uses_normal_provider_path():
    session, client, audio_cache = make_recording_gate_session()
    zero_pcm = b"\x00\x00" * 20
    nonzero_pcm = b"\x00\x00\x01\x00"
    await session._handle_audio_bytes(zero_pcm)
    await session._handle_audio_bytes(nonzero_pcm)
    session._turn_done.set()

    await session._handle_stop_recording({"turn_id": 111})

    assert session._turn_is_digital_silence() is False
    assert client.sent_audio == [zero_pcm, nonzero_pcm]
    assert client.commit_calls == 1
    assert client.start_response_calls == 1
    assert sent_text_payloads(session) == []
    assert audio_cache.enqueued == []


@pytest.mark.asyncio
async def test_digital_silence_guard_disabled_restores_provider_commit_path():
    session, client, audio_cache = make_recording_gate_session(
        no_speech_guard_enabled=False
    )
    zero_pcm = b"\x00\x00" * 20
    await session._handle_audio_bytes(zero_pcm)
    session._turn_done.set()

    await session._handle_stop_recording({"turn_id": 111})

    assert session._turn_is_digital_silence() is True
    assert client.sent_audio == [zero_pcm]
    assert client.commit_calls == 1
    assert client.start_response_calls == 1
    assert sent_text_payloads(session) == []
    assert audio_cache.enqueued == []


@pytest.mark.asyncio
async def test_digital_silence_keeps_provider_healthy_for_next_normal_turn():
    session, client, audio_cache = make_recording_gate_session()
    await session._handle_audio_bytes(b"\x00\x00" * 20)
    await session._handle_stop_recording({"turn_id": 111})

    assert session._client is client
    assert session._openai_ready.is_set()
    assert client.clear_audio_buffer_calls == 1
    assert client.close_calls == 0

    await session._handle_start_recording(
        {"turn_id": 222, "model": "m", "input_sample_rate": 24000}
    )
    assert client.refresh_session_calls == 1
    assert session._processed_audio_bytes == 0
    assert session._turn_is_digital_silence() is True
    assert session._turn_frame_rms_dbfs == []
    assert session._turn_frame_energy_total == []
    assert session._turn_frame_energy_hb300 == []
    assert session._turn_peak_sample_abs == 0
    assert session._turn_frame_carry == b""

    normal_pcm = b"\x01\x00" * 40
    await session._handle_audio_bytes(normal_pcm)
    assert session._turn_is_digital_silence() is False

    stop_task = asyncio.create_task(
        session._handle_stop_recording({"turn_id": 222})
    )
    await _wait_until(lambda: client.start_response_calls == 1)
    normal_text = "第二轮正常语音输入已经成功处理。"
    await drive_marker_answer(session, normal_text)
    await session._on_response_done({})
    await drain_finalize(session)
    await asyncio.wait_for(stop_task, timeout=1.0)

    assert client.sent_audio[-1] == normal_pcm
    assert client.commit_calls == 1
    assert client.start_response_calls == 1
    assert client.close_calls == 0
    assert session._emitted_text == normal_text
    assert sum(
        payload.get("content") == NO_SPEECH_PLACEHOLDER
        for payload in sent_text_payloads(session)
    ) == 1
    assert [entry["outcome"] for entry in audio_cache.enqueued] == [
        "digital_silence",
        "response.done",
    ]


# ── W3: deterministic processed-PCM frame-energy gate ──────────────────────


@pytest.mark.asyncio
async def test_dither_near_silence_skips_provider_with_complete_metric_log(caplog):
    session, client, audio_cache = make_recording_gate_session()
    pcm = pcm_dither(amplitude=2, frames=4)
    # Deliberately split away from 720-sample boundaries to exercise frame carry.
    split_a = 333 * 2
    split_b = (333 + 901) * 2
    chunks = [pcm[:split_a], pcm[split_a:split_b], pcm[split_b:]]
    for chunk in chunks:
        await session._handle_audio_bytes(chunk)

    metrics = session._no_speech_signal_metrics()
    assert metrics is not None
    assert metrics["peak_dbfs"] == pytest.approx(-84.29, abs=0.01)
    assert metrics["noise_floor_dbfs"] == pytest.approx(-84.29, abs=0.01)
    assert metrics["active_threshold_dbfs"] == -55.0
    assert metrics["active_frames"] == 0
    assert metrics["total_frames"] == 4
    assert metrics["no_speech_signal"] is True
    assert session._turn_frame_carry == b""

    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert client.sent_audio == chunks
    assert_no_speech_signal_result(
        session, client, audio_cache, caplog, pcm
    )


@pytest.mark.asyncio
async def test_loud_steady_sine_has_no_active_frames_and_is_blocked(caplog):
    session, client, audio_cache = make_recording_gate_session()
    pcm = pcm_sine_rms_frames([-30.0] * 12)
    await session._handle_audio_bytes(pcm)

    metrics = session._no_speech_signal_metrics()
    assert metrics is not None
    assert metrics["peak_dbfs"] == pytest.approx(-26.99, abs=0.02)
    assert metrics["noise_floor_dbfs"] == pytest.approx(-30.0, abs=0.02)
    assert metrics["active_threshold_dbfs"] == pytest.approx(-26.0, abs=0.02)
    assert metrics["active_frames"] == 0
    assert metrics["total_frames"] == 12

    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert_no_speech_signal_result(
        session, client, audio_cache, caplog, pcm
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("floor_dbfs", "burst_dbfs", "expected_threshold_dbfs"),
    [
        pytest.param(-60.0, -30.0, -55.0, id="room-floor-voice-burst"),
        pytest.param(-65.0, -45.0, -55.0, id="low-volume-voice-burst"),
        pytest.param(-35.0, -20.0, -31.0, id="fan-floor-voice-burst"),
    ],
)
async def test_am_burst_speech_proxies_use_normal_provider_path(
    floor_dbfs,
    burst_dbfs,
    expected_threshold_dbfs,
):
    session, client, audio_cache = make_recording_gate_session()
    pcm = pcm_speech_proxy(floor_dbfs, burst_dbfs)
    await session._handle_audio_bytes(pcm)

    metrics = session._no_speech_signal_metrics()
    assert metrics is not None
    assert metrics["noise_floor_dbfs"] == pytest.approx(floor_dbfs, abs=0.02)
    assert metrics["active_threshold_dbfs"] == pytest.approx(
        expected_threshold_dbfs, abs=0.02
    )
    assert metrics["active_frames"] == 3
    assert metrics["total_frames"] == 23
    assert metrics["no_speech_signal"] is False

    session._turn_done.set()
    await session._handle_stop_recording({"turn_id": 111})

    assert_normal_provider_path(session, client, audio_cache, pcm)


@pytest.mark.asyncio
async def test_sub_frame_nonzero_audio_fails_open_to_provider():
    session, client, audio_cache = make_recording_gate_session()
    samples = np.resize(np.array([-1, 1], dtype=np.int16), 200)
    pcm = samples.tobytes()
    await session._handle_audio_bytes(pcm)

    assert session._no_speech_signal_metrics() is None
    assert len(session._turn_frame_carry) == len(pcm)

    session._turn_done.set()
    await session._handle_stop_recording({"turn_id": 111})

    assert_normal_provider_path(session, client, audio_cache, pcm)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pcm",
    [
        pytest.param(pcm_dither(amplitude=2, frames=4), id="dither"),
        pytest.param(pcm_sine_rms_frames([-30.0] * 4), id="steady-sine"),
    ],
)
async def test_frame_energy_gate_disabled_restores_provider_path(pcm):
    session, client, audio_cache = make_recording_gate_session(
        no_speech_guard_enabled=False
    )
    await session._handle_audio_bytes(pcm)
    assert session._no_speech_signal_metrics()["no_speech_signal"] is True

    session._turn_done.set()
    await session._handle_stop_recording({"turn_id": 111})

    assert_normal_provider_path(session, client, audio_cache, pcm)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "pcm",
        "expected_peak_dbfs",
        "expected_floor_dbfs",
        "expected_threshold_dbfs",
        "expected_active_frames",
        "expected_total_frames",
        "expected_no_speech",
    ),
    [
        pytest.param(
            pcm_speech_proxy(-70.0, -30.0),
            -26.99,
            -70.0,
            -55.0,
            3,
            23,
            False,
            id="quiet-room-voice",
        ),
        pytest.param(
            pcm_sine_rms_frames([-37.0] * 2 + [-35.0] * 19 + [-33.0] * 2),
            -29.99,
            -35.0,
            -31.0,
            0,
            23,
            True,
            id="steady-fan",
        ),
        pytest.param(
            pcm_speech_proxy(-35.0, -20.0),
            -16.99,
            -35.0,
            -31.0,
            3,
            23,
            False,
            id="fan-plus-voice",
        ),
        pytest.param(
            pcm_speech_proxy(-65.0, -45.0),
            -41.99,
            -65.0,
            -55.0,
            3,
            23,
            False,
            id="whisper",
        ),
        pytest.param(
            pcm_speech_proxy(-75.0, -60.0),
            -57.05,
            -75.06,
            -55.0,
            0,
            23,
            True,
            id="breath",
        ),
        pytest.param(
            pcm_dither(amplitude=2, frames=4),
            -84.29,
            -84.29,
            -55.0,
            0,
            4,
            True,
            id="two-lsb-dither",
        ),
    ],
)
async def test_design_intent_signal_cases_have_expected_frame_metrics(
    pcm,
    expected_peak_dbfs,
    expected_floor_dbfs,
    expected_threshold_dbfs,
    expected_active_frames,
    expected_total_frames,
    expected_no_speech,
):
    session, _, _ = make_recording_gate_session()
    await session._handle_audio_bytes(pcm)

    metrics = session._no_speech_signal_metrics()
    assert metrics is not None
    assert metrics["peak_dbfs"] == pytest.approx(expected_peak_dbfs, abs=0.02)
    assert metrics["noise_floor_dbfs"] == pytest.approx(
        expected_floor_dbfs, abs=0.02
    )
    assert metrics["active_threshold_dbfs"] == pytest.approx(
        expected_threshold_dbfs, abs=0.02
    )
    assert metrics["active_frames"] == expected_active_frames
    assert metrics["total_frames"] == expected_total_frames
    assert metrics["no_speech_signal"] is expected_no_speech


# ── W6: true-speech-first policy; HB300 x p90 is opt-in ─────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "pcm",
        "expected_floor_dbfs",
        "expected_threshold_dbfs",
        "expected_p90_dbfs",
        "expected_hb300",
        "expected_active_frames",
        "expected_total_frames",
        "expected_max_run",
        "expected_active_ratio",
    ),
    [
        pytest.param(
            pcm_w6_quiet_speech_profile(),
            -57.0,
            -53.0,
            -51.0,
            0.04991,
            12,
            32,
            12,
            0.375,
            id="quiet-low-hb300-speech",
        ),
        pytest.param(
            pcm_w6_low_register_speech_profile(),
            -65.0,
            -55.0,
            -32.0,
            0.19005,
            28,
            36,
            28,
            0.77778,
            id="low-register-speech",
        ),
        pytest.param(
            pcm_intentional_speech_profile(-27.0),
            -65.0,
            -55.0,
            -27.0,
            0.37499,
            24,
            44,
            24,
            0.54545,
            id="normal-intentional-speech",
        ),
    ],
)
async def test_w6_true_speech_profiles_pass_default_policy(
    pcm,
    expected_floor_dbfs,
    expected_threshold_dbfs,
    expected_p90_dbfs,
    expected_hb300,
    expected_active_frames,
    expected_total_frames,
    expected_max_run,
    expected_active_ratio,
    caplog,
):
    session, client, audio_cache = make_recording_gate_session()
    await session._handle_audio_bytes(pcm)

    metrics = session._no_speech_signal_metrics()
    assert metrics is not None
    assert session._config.no_speech_floor_margin_db == 4.0
    assert session._config.no_speech_min_active_dbfs == -55.0
    assert session._config.no_speech_hb300_min_ratio == 0.0
    assert (
        len(session._turn_frame_rms_dbfs)
        == len(session._turn_frame_energy_total)
        == len(session._turn_frame_energy_hb300)
        == metrics["total_frames"]
    )
    assert metrics["noise_floor_dbfs"] == pytest.approx(
        expected_floor_dbfs, abs=0.05
    )
    assert metrics["active_threshold_dbfs"] == pytest.approx(
        expected_threshold_dbfs, abs=0.02
    )
    assert metrics["active_p90_dbfs"] == pytest.approx(
        expected_p90_dbfs, abs=0.01
    )
    assert metrics["hb300_weighted_ratio"] == pytest.approx(
        expected_hb300, abs=0.0001
    )
    assert metrics["active_frames"] == expected_active_frames
    assert metrics["total_frames"] == expected_total_frames
    assert metrics["max_active_run_frames"] == expected_max_run
    assert (
        metrics["active_frames"] / metrics["total_frames"]
        == pytest.approx(expected_active_ratio, abs=0.0001)
    )
    assert metrics["no_speech_signal"] is False

    session._turn_done.set()
    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert_turn_signal_log(caplog, "pass")
    assert_normal_provider_path(session, client, audio_cache, pcm)


@pytest.mark.asyncio
async def test_w6b_extremely_quiet_breath_is_accepted_tradeoff(caplog):
    # Accepted trade-off (W6b): sustained floor+5 dB breathing contains no
    # speech, but reaches the provider so equally weak speech is not killed.
    session, client, audio_cache = make_recording_gate_session()
    pcm = pcm_w6_extremely_quiet_breath_profile()
    await session._handle_audio_bytes(pcm)

    metrics = session._no_speech_signal_metrics()
    assert metrics is not None
    assert session._config.no_speech_floor_margin_db == 4.0
    assert metrics["noise_floor_dbfs"] == pytest.approx(-57.0, abs=0.02)
    assert metrics["active_threshold_dbfs"] == pytest.approx(-53.0, abs=0.02)
    assert metrics["active_p90_dbfs"] == pytest.approx(-52.0, abs=0.02)
    assert metrics["active_frames"] == 16
    assert metrics["total_frames"] == 36
    assert metrics["max_active_run_frames"] == 16
    assert metrics["no_speech_signal"] is False

    session._turn_done.set()
    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert_turn_signal_log(caplog, "pass")
    assert_normal_provider_path(session, client, audio_cache, pcm)


@pytest.mark.asyncio
async def test_w6_default_accepts_phantom_as_true_speech_priority_tradeoff(
    caplog,
):
    # Accepted trade-off: with the acoustically inseparable HB300 leg disabled,
    # a phantom residual reaches the provider so true dictation is not killed.
    session, client, audio_cache = make_recording_gate_session()
    pcm = pcm_low_frequency_profile(-31.0)
    await session._handle_audio_bytes(pcm)

    metrics = session._no_speech_signal_metrics()
    assert metrics is not None
    assert metrics["hb300_weighted_ratio"] == pytest.approx(0.11748, abs=0.0001)
    assert metrics["active_p90_dbfs"] == pytest.approx(-31.0, abs=0.01)
    assert metrics["no_speech_signal"] is False

    session._turn_done.set()
    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert_turn_signal_log(caplog, "pass")
    assert_normal_provider_path(session, client, audio_cache, pcm)


@pytest.mark.asyncio
async def test_w6_hb300_env_opt_in_still_blocks_phantom(
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("BRAINWAVE_NO_SPEECH_HB300_MIN_RATIO", "0.22")
    opt_in_config = TurnSessionConfig.from_env()
    session, client, audio_cache = make_recording_gate_session(
        no_speech_hb300_min_ratio=opt_in_config.no_speech_hb300_min_ratio
    )
    pcm = pcm_low_frequency_profile(-31.0)
    await session._handle_audio_bytes(pcm)

    metrics = session._no_speech_signal_metrics()
    assert metrics is not None
    assert session._config.no_speech_hb300_min_ratio == 0.22
    assert metrics["hb300_weighted_ratio"] < 0.22
    assert metrics["active_p90_dbfs"] < -26.0
    assert metrics["no_speech_signal"] is True

    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert_no_speech_signal_result(
        session, client, audio_cache, caplog, pcm
    )


@pytest.mark.asyncio
async def test_w6_keyboard_transients_remain_blocked_by_maxrun(caplog):
    session, client, audio_cache = make_recording_gate_session()
    pcm = pcm_keyboard_transient_profile()
    await session._handle_audio_bytes(pcm)

    metrics = session._no_speech_signal_metrics()
    assert metrics is not None
    assert session._config.no_speech_floor_margin_db == 4.0
    assert metrics["active_threshold_dbfs"] == -55.0
    assert metrics["active_frames"] == 5
    assert metrics["max_active_run_frames"] == 2
    assert metrics["no_speech_signal"] is True

    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert_no_speech_signal_result(
        session, client, audio_cache, caplog, pcm
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pcm",
    [
        pytest.param(
            pcm_low_frequency_profile(-31.0), id="phantom-profile"
        ),
        pytest.param(
            pcm_intentional_speech_profile(-27.0), id="normal-speech"
        ),
        pytest.param(
            pcm_intentional_speech_profile(-45.0), id="quiet-speech"
        ),
        pytest.param(
            pcm_low_frequency_profile(-20.0), id="loud-low-frequency"
        ),
        pytest.param(
            pcm_keyboard_transient_profile(), id="keyboard-transients"
        ),
    ],
)
async def test_w5_guard_disabled_sends_every_calibrated_profile(
    pcm, caplog
):
    session, client, audio_cache = make_recording_gate_session(
        no_speech_guard_enabled=False
    )
    await session._handle_audio_bytes(pcm)
    session._turn_done.set()

    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})

    assert_turn_signal_log(caplog, "pass")
    assert_normal_provider_path(session, client, audio_cache, pcm)


def test_current_prompt_leaves_silent_body_empty_and_accessors_stay_aligned():
    prompt = get_realtime_prompt()
    assert prompt == get_optimize_prompt()
    assert "正文留空，不输出任何说明性文字" in prompt
    assert "正文仅输出：（没有识别到输入）" not in prompt


@pytest.mark.asyncio
async def test_marker_seen_long_audio_short_output_waits_and_replaces_delayed_input_transcript():
    session = make_session()
    realtime_text = "DeepSeek V4 label sentence."
    fallback_text = (
        "This is the faithful transcript about OpenCode, Codex, Claude -p, "
        "DeepSeek V4 Pro, and CLI agent capability registration. "
        "It is much more complete than the stale short realtime output."
    )
    session._marker_seen = True
    session._emitted_text = realtime_text
    set_processed_audio_duration(session, 2.0)

    async def complete_input_transcription_later():
        await asyncio.sleep(0.02)
        await session._on_input_transcription_completed({"transcript": fallback_text})

    completion_task = asyncio.create_task(complete_input_transcription_later())
    await session._apply_input_transcription_fallback("response.text.done")
    await completion_task

    replacement_payloads = [
        payload
        for payload in session._websocket.sent_payloads
        if payload.get("type") == "text" and payload.get("isNewResponse") is True
    ]
    assert replacement_payloads == [
        {
            "type": "text",
            "content": fallback_text,
            "isNewResponse": True,
            "turn_id": 111,
        }
    ]
    assert session._emitted_text == fallback_text


@pytest.mark.asyncio
async def test_marker_seen_suspicious_delta_only_timeout_does_not_replace_realtime_text():
    session = make_session(suspicious_input_transcript_grace_sec=0.05)
    realtime_text = "DeepSeek V4 label sentence."
    partial_delta = (
        "This partial input transcript is much longer than the stale realtime "
        "text, but it is only a delta and must not replace marker-following "
        "output before the completed event arrives."
    )
    session._marker_seen = True
    session._emitted_text = realtime_text
    set_processed_audio_duration(session, 2.0)

    await session._on_input_transcription_delta({"delta": partial_delta})
    await session._apply_input_transcription_fallback("response.text.done")

    assert session._websocket.sent_payloads == []
    assert session._emitted_text == realtime_text
    assert session._input_transcript_text == partial_delta
    assert session._input_transcript_completed is False


@pytest.mark.asyncio
async def test_marker_seen_suspicious_delta_then_failed_does_not_replace_realtime_text():
    session = make_session(suspicious_input_transcript_grace_sec=0.5)
    realtime_text = "DeepSeek V4 label sentence."
    partial_delta = (
        "This partial input transcript is materially longer than the stale "
        "realtime text, but the transcription then fails, so it is not a "
        "completed replacement candidate."
    )
    session._marker_seen = True
    session._emitted_text = realtime_text
    set_processed_audio_duration(session, 2.0)

    await session._on_input_transcription_delta({"delta": partial_delta})

    async def fail_input_transcription_later():
        await asyncio.sleep(0.02)
        await session._on_input_transcription_failed(
            {"error": {"type": "server_error", "code": "failed", "message": "boom"}}
        )

    failure_task = asyncio.create_task(fail_input_transcription_later())
    await session._apply_input_transcription_fallback("response.text.done")
    await failure_task

    assert session._websocket.sent_payloads == []
    assert session._emitted_text == realtime_text
    assert session._input_transcript_text == partial_delta
    assert session._input_transcript_done.is_set()
    assert session._input_transcript_completed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("audio_seconds", "emitted_text"),
    [
        (0.5, "short marker-following text"),
        (2.0, "x" * 80),
    ],
)
async def test_marker_seen_fast_path_does_not_wait_for_short_audio_or_long_output(
    audio_seconds,
    emitted_text,
):
    session = make_session(suspicious_input_transcript_grace_sec=0.5)
    session._marker_seen = True
    session._emitted_text = emitted_text
    set_processed_audio_duration(session, audio_seconds)

    await asyncio.wait_for(
        session._apply_input_transcription_fallback("response.text.done"),
        timeout=0.1,
    )

    assert session._websocket.sent_payloads == []
    assert session._emitted_text == emitted_text
    assert not session._input_transcript_done.is_set()


@pytest.mark.asyncio
async def test_marker_not_seen_still_waits_for_regular_input_transcript_fallback():
    session = make_session(input_transcript_grace_sec=0.5)
    realtime_answer = "Sure, here is an answer instead of a transcript."
    fallback_text = "Faithful input transcription that should replace the answer."
    session._marker_seen = False
    session._emitted_text = realtime_answer
    set_processed_audio_duration(session, 0.5)

    async def complete_input_transcription_later():
        await asyncio.sleep(0.02)
        await session._on_input_transcription_completed({"transcript": fallback_text})

    completion_task = asyncio.create_task(complete_input_transcription_later())
    await session._apply_input_transcription_fallback("response.text.done")
    await completion_task

    assert session._websocket.sent_payloads == [
        {
            "type": "text",
            "content": fallback_text,
            "isNewResponse": True,
            "turn_id": 111,
        }
    ]
    assert session._emitted_text == fallback_text


@pytest.mark.asyncio
async def test_over_20_deltas_without_marker_emits_without_faking_marker_seen():
    # 0281 guard: never drop words. 0436 G3: do not fake marker_seen — flag
    # emitted-without-marker so finalize's no-marker replacement still governs.
    session = make_session()
    for i in range(25):
        await session._on_text_delta({"delta": f"word{i} "})

    assert session._marker_seen is False
    assert session._emitted_without_marker is True
    text_payloads = [
        p for p in session._websocket.sent_payloads if p.get("type") == "text"
    ]
    assert text_payloads, "expected streamed deltas (0281: no word drop)"
    assert "word0" in session._emitted_text


@pytest.mark.asyncio
async def test_no_marker_20_delta_path_replaced_by_input_transcript_on_finalize():
    session = make_session(input_transcript_grace_sec=0.5)
    for i in range(25):
        await session._on_text_delta({"delta": f"answer{i} "})
    assert session._emitted_without_marker is True
    assert session._marker_seen is False

    fallback_text = "这是忠实的语音转写文本，不是模型的回答。"

    async def complete_input_transcription_later():
        await asyncio.sleep(0.02)
        await session._on_input_transcription_completed({"transcript": fallback_text})

    completion_task = asyncio.create_task(complete_input_transcription_later())
    await session._apply_input_transcription_fallback("response.text.done")
    await completion_task

    replacement_payloads = [
        p
        for p in session._websocket.sent_payloads
        if p.get("type") == "text" and p.get("isNewResponse") is True
    ]
    assert replacement_payloads and replacement_payloads[-1]["content"] == fallback_text
    assert session._emitted_text == fallback_text


@pytest.mark.asyncio
async def test_marker_seen_answer_guard_replaces_dissimilar_completed_transcript():
    # Short audio (not suspicious) but marker-following output is an answer that
    # diverges from the faithful transcript -> answer-similarity guard replaces.
    session = make_session()
    session._marker_seen = True
    session._emitted_text = "Sure, Kerry Park is a great photo spot near Seattle."
    set_processed_audio_duration(session, 0.5)  # NOT suspicious (below 1.0s)

    faithful = "帮我调研西雅图周围三十分钟内适合摄影出片的景点"
    await session._on_input_transcription_completed({"transcript": faithful})
    await session._apply_input_transcription_fallback("response.text.done")

    replacement_payloads = [
        p for p in session._websocket.sent_payloads if p.get("isNewResponse") is True
    ]
    assert replacement_payloads and replacement_payloads[-1]["content"] == faithful
    assert session._emitted_text == faithful


@pytest.mark.asyncio
async def test_marker_seen_answer_guard_keeps_similar_paraphrase():
    # The guard must not fire on light punctuation/casing polish of the same
    # content (similarity stays well above the threshold).
    session = make_session()
    session._marker_seen = True
    original = "帮我调研一下西雅图周围三十分钟内适合摄影出片的景点。"
    session._emitted_text = original
    set_processed_audio_duration(session, 0.5)

    faithful = "帮我调研西雅图周围三十分钟内适合摄影出片的景点"
    await session._on_input_transcription_completed({"transcript": faithful})
    await session._apply_input_transcription_fallback("response.text.done")

    assert session._websocket.sent_payloads == []
    assert session._emitted_text == original


# ── S1: single finalize barrier — replacement must precede idle ──────────────
# All finalize cases run through the REAL provider dispatcher (task 0436 S1 R4).


@pytest.mark.asyncio
async def test_text_done_alone_does_not_finalize_or_idle():
    # Only response.done finalizes (task 0436 S1); text.done just records.
    session = make_session(answer_guard_grace_sec=0.5)
    await drive_marker_answer(session, "今天下午三点开会讨论下季度的计划安排")
    await session._on_response_text_done({})
    assert session._finalized is False
    assert session._finalize_task is None
    assert "status:idle" not in payload_types(session)


@pytest.mark.asyncio
async def test_s1_late_asr_between_text_done_and_response_done_replaces_before_idle():
    # (③) ASR completes after text.done but before response.done — the barrier
    # must emit the replacement before status:idle so the IME never discards it.
    session, client, ws = make_dispatcher_session(answer_guard_grace_sec=0.5)
    dispatcher = asyncio.create_task(client.receive_messages())
    faithful = "帮我调研西雅图周围三十分钟内适合摄影出片的景点"
    try:
        await drive_marker_answer_via_dispatcher(
            ws, "Sure, Kerry Park is a great photo spot near Seattle."
        )
        await ws.push({"type": "response.text.done"})
        await ws.push({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": faithful,
        })
        await ws.push({"type": "response.done"})
        await drain_finalize(session)
    finally:
        await stop_dispatcher(dispatcher)

    types = payload_types(session)
    assert "replacement" in types and "status:idle" in types
    assert types.index("replacement") < types.index("status:idle")
    assert types.count("status:idle") == 1
    assert session._emitted_text == faithful


@pytest.mark.asyncio
async def test_s1_late_asr_queued_within_grace_replaces_before_idle():
    # (①) The R3-critical case: ASR is queued shortly AFTER response.done. With
    # the serial dispatcher, finalize must be decoupled so the dispatcher keeps
    # consuming and delivers the ASR while the barrier waits on grace — otherwise
    # it self-deadlocks and only fails open. Assert replacement precedes the one
    # idle and no timeout was counted.
    session, client, ws = make_dispatcher_session(answer_guard_grace_sec=0.5)
    dispatcher = asyncio.create_task(client.receive_messages())
    faithful = "帮我调研西雅图周围三十分钟内适合摄影出片的景点"
    try:
        await drive_marker_answer_via_dispatcher(
            ws, "Sure, Kerry Park is a great photo spot near Seattle."
        )
        await ws.push({"type": "response.text.done"})
        await ws.push({"type": "response.done"})

        # ASR arrives ~20ms after response.done was queued — the dispatcher is
        # free to consume it because response.done no longer blocks on it.
        async def feed_asr_late():
            await asyncio.sleep(0.02)
            await ws.push({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": faithful,
            })

        feeder = asyncio.create_task(feed_asr_late())
        await drain_finalize(session)
        await feeder
    finally:
        await stop_dispatcher(dispatcher)

    types = payload_types(session)
    assert "replacement" in types and "status:idle" in types
    assert types.index("replacement") < types.index("status:idle")
    assert types.count("status:idle") == 1
    assert session._emitted_text == faithful
    assert session._input_transcript_wait_timeouts == 0


@pytest.mark.asyncio
async def test_s1_late_asr_outside_grace_fails_open_no_replacement_still_idle():
    # (②) ASR never arrives within grace — fail open (no replacement), still send
    # a single idle, and count the timeout.
    session, client, ws = make_dispatcher_session(answer_guard_grace_sec=0.05)
    dispatcher = asyncio.create_task(client.receive_messages())
    try:
        await drive_marker_answer_via_dispatcher(
            ws, "Sure, Kerry Park is a great photo spot near Seattle."
        )
        await ws.push({"type": "response.text.done"})
        await ws.push({"type": "response.done"})
        await drain_finalize(session)
    finally:
        await stop_dispatcher(dispatcher)

    types = payload_types(session)
    assert "replacement" not in types
    assert types.count("status:idle") == 1
    assert session._input_transcript_wait_timeouts == 1


@pytest.mark.asyncio
async def test_s1_normal_turn_asr_already_done_adds_no_wait():
    # Happy path: a faithful marker-following output whose ASR already completed
    # must NOT wait (no perceptible latency red line, task 0436 S1).
    session, client, ws = make_dispatcher_session(answer_guard_grace_sec=5.0)
    dispatcher = asyncio.create_task(client.receive_messages())
    sentence = "今天下午三点开会讨论下季度的计划安排"
    try:
        await drive_marker_answer_via_dispatcher(ws, sentence)
        await ws.push({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": sentence,
        })
        await ws.push({"type": "response.text.done"})
        await ws.push({"type": "response.done"})
        started = time.perf_counter()
        await drain_finalize(session)
        elapsed = time.perf_counter() - started
    finally:
        await stop_dispatcher(dispatcher)

    assert elapsed < 1.0  # did not burn the 5s grace (ASR already done)
    assert "replacement" not in payload_types(session)  # faithful -> kept
    assert "status:idle" in payload_types(session)
    assert session._input_transcript_wait_timeouts == 0


@pytest.mark.asyncio
async def test_s1_new_turn_cancels_pending_finalize_no_pollution():
    # (④) A finalize barrier still parked on grace when a new turn starts must be
    # cancelled and must NOT send its stale replacement/idle onto the new turn.
    session, client, ws = make_dispatcher_session(answer_guard_grace_sec=5.0)
    dispatcher = asyncio.create_task(client.receive_messages())
    try:
        await drive_marker_answer_via_dispatcher(
            ws, "Sure, Kerry Park is a great photo spot near Seattle."
        )
        # An ASR IS queued for the old turn, but no completed event is delivered,
        # so the barrier stays parked on grace (5s).
        await ws.push({"type": "response.text.done"})
        await ws.push({"type": "response.done"})
        await _wait_until(lambda: session._finalize_task is not None)
        old_task = session._finalize_task
        await _wait_until(lambda: not old_task.done() and session._finalized is False)

        payloads_before = len(session._websocket.sent_payloads)
        # New turn starts (the exact reset start_recording performs).
        session._reset_turn_state(active_turn_id=222)
        await asyncio.gather(old_task, return_exceptions=True)

        assert old_task.cancelled()
        assert session._finalize_task is None
        # The stale barrier emitted nothing after the reset (no replacement/idle).
        assert len(session._websocket.sent_payloads) == payloads_before
        assert "replacement" not in payload_types(session)
        assert "status:idle" not in payload_types(session)
    finally:
        await stop_dispatcher(dispatcher)


@pytest.mark.asyncio
async def test_s1_double_terminal_is_idempotent():
    # (⑤) A second terminal event (or a stop-fallback racing response.done) must
    # not finalize twice: one replacement, one idle. Covers both the in-flight
    # guard and the already-finalized guard in _schedule_finalize.
    session, client, ws = make_dispatcher_session(answer_guard_grace_sec=0.5)
    dispatcher = asyncio.create_task(client.receive_messages())
    faithful = "帮我调研西雅图周围三十分钟内适合摄影出片的景点"
    try:
        await drive_marker_answer_via_dispatcher(
            ws, "Sure, Kerry Park is a great photo spot near Seattle."
        )
        await ws.push({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": faithful,
        })
        await ws.push({"type": "response.done"})
        await _wait_until(lambda: session._finalize_task is not None)
        # In-flight guard: scheduling again returns the same task, not a new one.
        first_task = session._finalize_task
        session._schedule_finalize("response.done")
        assert session._finalize_task is first_task
        await drain_finalize(session)
        # Already-finalized guard: a duplicate response.done schedules nothing.
        assert session._finalized is True
        session._schedule_finalize("response.done")
        assert session._finalize_task is first_task
        await ws.push({"type": "response.done"})
        await asyncio.sleep(0.02)
    finally:
        await stop_dispatcher(dispatcher)

    types = payload_types(session)
    assert types.count("replacement") == 1
    assert types.count("status:idle") == 1
    assert session._emitted_text == faithful


@pytest.mark.asyncio
async def test_s1_mid_guard_concurrent_finalize_during_answer_guard_no_replacement_after_idle(
    monkeypatch,
):
    # (⑥) Mid-guard race (task 0436 S1 R6): once the finalize barrier is decoupled
    # from response.done, a concurrent _finalize_turn — reachable via a provider
    # `error` — can land inside the answer-guard `to_thread` window, send the one
    # status:idle and clear _active_turn_id while the barrier is parked in its
    # worker thread. Without a post-to_thread re-check the barrier resumes and
    # sends the faithful replacement AFTER idle, breaking S1's invariant
    # "replacement strictly precedes the single idle" (the IME then discards a
    # correct transcript, replaying the very S1 symptom).
    #
    # Drive the error through the REAL serial dispatcher while the answer-guard
    # similarity diff is deterministically held open in its thread.
    import threading
    import realtime_server as rs

    reached = threading.Event()   # set inside the diff -> to_thread window is open
    proceed = threading.Event()   # test releases the diff after the finalize lands
    real_ratio = rs.transcription_similarity_ratio

    def blocking_ratio(current_text, fallback_text, cap):
        reached.set()
        proceed.wait(timeout=2.0)  # hold the answer-guard window open
        return real_ratio(current_text, fallback_text, cap)

    monkeypatch.setattr(rs, "transcription_similarity_ratio", blocking_ratio)

    session, client, ws = make_dispatcher_session(answer_guard_grace_sec=0.5)
    dispatcher = asyncio.create_task(client.receive_messages())
    # Marker-following answer that diverges from the faithful transcript, so the
    # answer-similarity guard reaches the bounded diff (our to_thread window).
    faithful = "帮我调研西雅图周围三十分钟内适合摄影出片的景点"
    try:
        await drive_marker_answer_via_dispatcher(
            ws, "Sure, Kerry Park is a great photo spot near Seattle."
        )
        # ASR completes BEFORE response.done so the answer-guard branch runs.
        await ws.push({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": faithful,
        })
        await ws.push({"type": "response.text.done"})
        await ws.push({"type": "response.done"})  # schedules the finalize barrier

        # Park the barrier inside the answer-guard to_thread diff.
        await _wait_until(lambda: reached.is_set())

        # Land a concurrent finalize via the REAL dispatcher: a provider error
        # sends the single idle and clears _active_turn_id mid-window.
        await ws.push({"type": "error", "error": {"message": "boom"}})
        await _wait_until(lambda: session._finalized is True)

        # Release the diff; the barrier resumes past both to_thread calls.
        proceed.set()
        await drain_finalize(session)
    finally:
        proceed.set()
        await stop_dispatcher(dispatcher)

    types = payload_types(session)
    # S1 invariant: exactly one idle, and NO replacement after it.
    assert types.count("status:idle") == 1, types
    idle_idx = types.index("status:idle")
    assert "replacement" not in types[idle_idx + 1:], (
        f"replacement emitted after idle — S1 mid-guard violation: {types}"
    )
    assert session._finalized is True


# ── M6: guard compares the full corrected view (held tail included) ──────────


@pytest.mark.asyncio
async def test_short_sentence_held_tail_uses_full_view_no_false_trigger():
    # An 18-char faithful sentence is mostly still in the homonym corrector's
    # 16-char hold tail at fallback time, so _emitted_text is short. The guard
    # must compare the FULL corrected view (task 0436 M6) so a faithful
    # transcript scores ~1.0 and is not falsely replaced. Driven through the real
    # dispatcher so the deferred finalize barrier runs end to end.
    session, client, ws = make_dispatcher_session(answer_guard_grace_sec=0.5)
    dispatcher = asyncio.create_task(client.receive_messages())
    sentence = "今天下午三点开会讨论下季度的计划安排"  # 18 chars
    try:
        await drive_marker_answer_via_dispatcher(ws, sentence)
        # Let the dispatcher consume every queued delta before inspecting state.
        await _wait_until(lambda: session._current_emitted_view().strip() == sentence)
        assert len(session._emitted_text) < len(sentence)  # tail still held
        await ws.push({"type": "response.text.done"})
        await ws.push({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": sentence,
        })
        await ws.push({"type": "response.done"})
        await drain_finalize(session)
    finally:
        await stop_dispatcher(dispatcher)

    assert "replacement" not in payload_types(session)
    assert "status:idle" in payload_types(session)


# ── M5: bounded similarity runtime ──────────────────────────────────────────


def test_similarity_hard_cap_bounds_runtime_on_long_input():
    # A pathological long input must be capped so the diff stays bounded instead
    # of running the full near-O(n^2) SequenceMatcher (task 0436 M5).
    left = "".join(chr(0x4e00 + (i % 400)) for i in range(20000))
    right = "".join(chr(0x4e00 + ((i * 7 + 3) % 400)) for i in range(20000))
    assert len(normalize_for_similarity(left)) > 6000  # uncapped would be O(n^2)

    t0 = time.perf_counter()
    ratio = transcription_similarity_ratio(left, right, max_chars=2000)
    dt = time.perf_counter() - t0

    assert 0.0 <= ratio <= 1.0
    assert dt < 1.5  # capped to 2000 chars -> bounded


def test_cap_head_tail_sampling_exposes_suffix_answer_and_stays_bounded():
    # task 0436 M3-R4: a faithful transcript (6000 unique chars) whose realtime
    # echo appended an 8000-char disjoint answer used to score similarity=1.0 /
    # novel=0.0 under prefix-only capping (the shared prefix hid the suffix).
    # Head+tail sampling must expose the divergence so the answer guard fires.
    transcript = "".join(chr(0x3400 + i) for i in range(6000))       # block A
    answer = "".join(chr(0x9000 + i) for i in range(8000))           # disjoint B
    emitted = transcript + answer
    cap = 6000

    similarity = transcription_similarity_ratio(emitted, transcript, cap)
    novel = emitted_novel_material_ratio(emitted, transcript, cap)
    # Below the 0.60 similarity threshold used by the guard -> replacement.
    assert similarity < 0.60, similarity
    assert novel > 0.0, novel

    # A normal long transcription (emitted ≈ full transcript) is NOT mis-killed.
    faithful = "".join(chr(0x3400 + (i % 9000)) for i in range(20000))
    assert transcription_similarity_ratio(faithful, faithful, cap) == 1.0
    assert emitted_novel_material_ratio(faithful, faithful, cap) == 0.0

    # Runtime stays bounded across the benchmark sizes (each diff sees <= cap).
    for n in (1000, 2500, 5000, 10000, 20000):
        big = "".join(chr(0x3400 + (i % 12000)) for i in range(n))
        t0 = time.perf_counter()
        r = transcription_similarity_ratio(big, big[::-1], cap)
        assert time.perf_counter() - t0 < 1.5, n
        assert 0.0 <= r <= 1.0


# ── M4: normalization keeps symbols; novel-material signal ──────────────────


def test_normalize_keeps_symbols_strips_only_punctuation():
    assert normalize_for_similarity("C++") == "c++"      # Sm symbol kept
    assert normalize_for_similarity("$100") == "$100"    # Sc symbol kept
    assert normalize_for_similarity("C.") == "c"          # Po punctuation dropped
    # Symbols distinguish tokens that the old S-stripping folded together.
    assert normalize_for_similarity("C++") != normalize_for_similarity("C--")


def test_normalize_keeps_hash_so_csharp_stays_distinct():
    # task 0436 M4 residual: '#' is Unicode Po, so dropping all P* folded 'C#'
    # onto 'C' and 'F#' onto 'F'. '#' is kept so these stay distinct.
    assert "#" in normalize_for_similarity("C#")
    assert normalize_for_similarity("C") != normalize_for_similarity("C#")
    assert normalize_for_similarity("C#") != normalize_for_similarity("C++")
    assert normalize_for_similarity("F#") == "f#"
    # Ordinary sentence punctuation is still dropped.
    assert normalize_for_similarity("好的。") == "好的"


def test_novel_material_ratio_flags_appended_answer_not_faithful_subset():
    question = "什么是供应链管理"
    restate_answer = question + "它是指商品从生产到消费流动的计划与协调控制"
    assert emitted_novel_material_ratio(restate_answer, question) > 0.55
    assert emitted_novel_material_ratio(question, question) == 0.0
    assert emitted_novel_material_ratio(question, restate_answer) == 0.0


# ── S2: same-turn retry preserves pending PCM; no loss, no duplication ───────


class _FakeProviderClient:
    def __init__(self, connect_counter, fail_first):
        self.sent_audio = []
        self.commit_calls = 0
        self.start_response_calls = 0
        self.clear_audio_buffer_calls = 0
        self._open = False
        self._connect_counter = connect_counter
        self._fail_first = fail_first

    async def connect(self, modalities=None, instructions=None):
        self._connect_counter["n"] += 1
        if self._fail_first and self._connect_counter["n"] == 1:
            raise RuntimeError("confirmation failed")
        self._open = True

    async def send_audio(self, chunk):
        self.sent_audio.append(chunk)

    async def commit_audio(self):
        self.commit_calls += 1

    async def start_response(self, instructions=None):
        self.start_response_calls += 1

    async def clear_audio_buffer(self):
        self.clear_audio_buffer_calls += 1

    async def close(self):
        self._open = False

    def _is_ws_open(self):
        return self._open

    def register_handler(self, *a, **k):
        pass

    def set_on_disconnect(self, *a, **k):
        pass

    async def refresh_session(self, modalities=None, instructions=None):
        pass


@pytest.mark.asyncio
async def test_same_turn_retry_preserves_and_delivers_all_pcm_exactly_once():
    session = make_session(
        provider_init_max_attempts=1,
        provider_init_retry_delay_sec=0.0,
    )
    session._audio_processor.set_source_sample_rate(24000)  # no resample: bytes stable
    counter = {"n": 0}
    clients = []

    async def fake_create_client(model=None):
        client = _FakeProviderClient(counter, fail_first=True)
        clients.append(client)
        return client

    session._create_client = fake_create_client

    # input_sample_rate == target so process_audio_chunk is a byte-stable no-op.
    start_msg = {"turn_id": 5, "model": "m", "input_sample_rate": 24000}

    # New turn 5: first start_recording -> connect fails -> not ready.
    await session._handle_start_recording(start_msg)
    assert not session._openai_ready.is_set()

    # PCM arrives for turn 5 while the provider is not ready -> pending buffer.
    chunk_a = b"\x01\x00" * 100
    chunk_b = b"\x02\x00" * 100
    await session._handle_audio_bytes(chunk_a)
    await session._handle_audio_bytes(chunk_b)
    assert session._pending_audio_chunks == [chunk_a, chunk_b]

    # Same-turn retry -> pending preserved -> rebuild connects -> flush pending.
    await session._handle_start_recording(start_msg)
    assert session._openai_ready.is_set()

    live_client = clients[-1]
    # All buffered PCM forwarded exactly once, in order (no loss, no dup).
    assert live_client.sent_audio == [chunk_a, chunk_b]

    # Subsequent PCM streams straight through.
    chunk_c = b"\x03\x00" * 100
    await session._handle_audio_bytes(chunk_c)
    assert live_client.sent_audio == [chunk_a, chunk_b, chunk_c]

    total_bytes = sum(len(c) for c in live_client.sent_audio)
    assert total_bytes == len(chunk_a) + len(chunk_b) + len(chunk_c)


@pytest.mark.asyncio
async def test_same_turn_retry_counts_failure_window_speech_before_stop():
    session = make_session(
        provider_init_max_attempts=1,
        provider_init_retry_delay_sec=0.0,
    )
    session._audio_processor.set_source_sample_rate(24000)
    audio_cache = _RecordingAudioCache()
    session._audio_cache = audio_cache
    counter = {"n": 0}
    clients = []

    async def fake_create_client(model=None):
        client = _FakeProviderClient(counter, fail_first=True)
        clients.append(client)
        return client

    session._create_client = fake_create_client
    start_msg = {"turn_id": 5, "model": "m", "input_sample_rate": 24000}

    await session._handle_start_recording(start_msg)
    assert session._is_recording is False
    assert not session._openai_ready.is_set()

    failure_window_speech = pcm_speech_proxy(
        -65.0, -45.0, trailing_samples=137
    )
    await session._handle_audio_bytes(failure_window_speech)
    assert session._pending_audio_chunks == [failure_window_speech]
    assert session._processed_audio_bytes == len(failure_window_speech)
    assert session._turn_is_digital_silence() is False
    before_retry_metrics = session._no_speech_signal_metrics()
    assert before_retry_metrics is not None
    assert before_retry_metrics["active_frames"] == 3
    assert before_retry_metrics["total_frames"] == 23
    assert before_retry_metrics["no_speech_signal"] is False
    before_retry_frame_values = list(session._turn_frame_rms_dbfs)
    before_retry_frame_energy_total = list(
        session._turn_frame_energy_total
    )
    before_retry_frame_energy_hb300 = list(
        session._turn_frame_energy_hb300
    )
    before_retry_peak = session._turn_peak_sample_abs
    before_retry_carry = session._turn_frame_carry
    assert len(before_retry_carry) == 137 * 2

    await session._handle_start_recording(start_msg)
    live_client = clients[-1]
    assert live_client.sent_audio == [failure_window_speech]
    assert session._pending_audio_chunks == []
    assert session._processed_audio_bytes == len(failure_window_speech)
    assert session._turn_frame_rms_dbfs == before_retry_frame_values
    assert (
        session._turn_frame_energy_total
        == before_retry_frame_energy_total
    )
    assert (
        session._turn_frame_energy_hb300
        == before_retry_frame_energy_hb300
    )
    assert session._turn_peak_sample_abs == before_retry_peak
    assert session._turn_frame_carry == before_retry_carry
    assert session._no_speech_signal_metrics() == before_retry_metrics

    stop_task = asyncio.create_task(session._handle_stop_recording({"turn_id": 5}))
    await _wait_until(lambda: live_client.start_response_calls == 1)
    await drive_marker_answer(session, "失败窗口中的真实语音已进入正常转写路径。")
    await session._on_response_done({})
    await drain_finalize(session)
    await asyncio.wait_for(stop_task, timeout=1.0)

    assert live_client.commit_calls == 1
    assert live_client.start_response_calls == 1
    assert all(
        payload.get("content") != NO_SPEECH_PLACEHOLDER
        for payload in sent_text_payloads(session)
    )
    assert [entry["outcome"] for entry in audio_cache.enqueued] == ["response.done"]


@pytest.mark.asyncio
async def test_refresh_failure_rebuilds_and_preserves_pending_audio():
    # A reused session whose refresh raises (the client's 5s x2 confirmation
    # timeout, task 0436 S2) must rebuild a fresh client WITHOUT discarding the
    # pending audio buffer for the same turn.
    session = make_session(provider_init_max_attempts=1, provider_init_retry_delay_sec=0.0)
    new_clients = []

    class _StaleClient:
        def __init__(self):
            self._open = True

        async def refresh_session(self, modalities=None, instructions=None):
            raise RuntimeError("refresh timeout x2")

        async def close(self):
            self._open = False

        def _is_ws_open(self):
            return self._open

    async def fake_create_client(model=None):
        client = _FakeProviderClient({"n": 0}, fail_first=False)
        new_clients.append(client)
        return client

    session._create_client = fake_create_client
    session._client = _StaleClient()
    session._openai_ready.set()
    session._active_model = "m"
    session._active_turn_id = 7
    session._is_recording = True
    pending = [b"\x01\x00" * 10, b"\x02\x00" * 10]
    session._pending_audio_chunks = list(pending)

    ok = await session._init_or_reuse_client(model="m", turn_id=7)

    assert ok is True
    assert len(new_clients) == 1              # rebuilt after refresh failure
    assert session._openai_ready.is_set()
    assert session._pending_audio_chunks == pending  # not cleared by the rebuild


# ── N1: shared [0,1] ratio env parser ───────────────────────────────────────


def test_parse_ratio_env_rejects_bogus_inf_and_out_of_range(monkeypatch):
    monkeypatch.delenv("MW_TEST_RATIO", raising=False)
    assert parse_ratio_env("MW_TEST_RATIO", 0.6) == 0.6  # unset -> default
    monkeypatch.setenv("MW_TEST_RATIO", "0.42")
    assert parse_ratio_env("MW_TEST_RATIO", 0.6) == 0.42  # valid
    monkeypatch.setenv("MW_TEST_RATIO", "bogus")
    assert parse_ratio_env("MW_TEST_RATIO", 0.6) == 0.6  # non-numeric
    monkeypatch.setenv("MW_TEST_RATIO", "inf")
    assert parse_ratio_env("MW_TEST_RATIO", 0.6) == 0.6  # non-finite
    monkeypatch.setenv("MW_TEST_RATIO", "1.5")
    assert parse_ratio_env("MW_TEST_RATIO", 0.6) == 0.6  # above 1
    monkeypatch.setenv("MW_TEST_RATIO", "-0.1")
    assert parse_ratio_env("MW_TEST_RATIO", 0.6) == 0.6  # below 0
