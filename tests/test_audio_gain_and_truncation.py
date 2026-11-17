"""Task 1193 (W0322): phone dictation lost its second half.

Covers the speech front-end (audio_gain), pause splitting, segment joining,
the truncated / looping input-transcript guards, and the streaming lane's
integration (TurnSession). Audio is synthesised to reproduce the measured T80
profile; no real recording is used.
"""

import asyncio
import dataclasses
import logging
import pathlib
import sys

import numpy as np
import pytest
import scipy.signal


from micwave_core.audio_gain import (  # noqa: E402
    StreamingSpeechAGC,
    measure_speech_activity_pcm16,
    preprocess_speech_pcm16,
    split_pcm_on_pauses,
)
from micwave_core.realtime_text_utils import (  # noqa: E402
    INCOMPLETE_TURN_MARKER,
    NO_SPEECH_PLACEHOLDER,
    OutputTruncationAssessment,
    assess_output_truncation,
    is_likely_truncated_input_transcript,
    is_runaway_repetition_transcript,
    scaled_suspicious_output_chars,
    strip_diagnostic_markers,
    transcription_similarity_ratio,
)
from micwave_core.transcript_merge import join_segment_texts  # noqa: E402
from .test_realtime_server_fallback import (  # noqa: E402
    MARKER,
    assert_no_speech_signal_result,
    drain_finalize,
    make_recording_gate_session,
    make_session,
    sent_text_payloads,
    set_processed_audio_duration,
)

SR = 24000


# ── synthetic audio ─────────────────────────────────────────────────────────


def _rms(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _dbfs(rms_value: float) -> float:
    return 20.0 * np.log10(max(rms_value, 1e-9) / 32768.0)


def _band(x, lo, hi, order=4):
    sos = scipy.signal.butter(order, [lo / (SR / 2), hi / (SR / 2)], btype="bandpass", output="sos")
    return scipy.signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def _syllables(rng, n, a, b, level, speech, on):
    """Harmonic syllables (f0 150-220 Hz, partials to 2.6 kHz, formant
    weighting around 700 / 1600 Hz) with sin^2 envelopes, scaled so the phrase
    [a, b) has RMS ``level``."""
    seg = np.zeros(n)
    s = a
    while s < b - 0.12:
        e = min(s + 0.15 + 0.1 * rng.random(), b)
        i0, i1 = int(s * SR), int(e * SR)
        m = i1 - i0
        tt = np.arange(m) / SR
        f0 = 150 + 70 * rng.random()
        tone = np.zeros(m)
        for h in range(1, 17):
            f = f0 * h
            if f > 2600:
                break
            amp = 1.0 / (1 + abs(f - 700) / 500) + 0.5 / (1 + abs(f - 1600) / 400)
            tone += amp * np.sin(2 * np.pi * f * tt + rng.random() * 2 * np.pi)
        seg[i0:i1] += tone * np.sin(np.pi * np.arange(m) / m) ** 2
        on[i0:i1] = True
        s = e + 0.05 + 0.05 * rng.random()
    ia, ib = int(a * SR), int(b * SR)
    seg[ia:ib] *= level / _rms(seg[ia:ib])
    speech += seg


T80_LONG_PAUSE = (66.06, 70.62)


def build_t80_like(seed=80):
    """92 s at 24 kHz shaped after the measured T80 recording:

    * 10-80 Hz handling rumble, 1/f^2 amplitude spectrum, RMS 165;
    * broadband floor so the post-HPF floor lands near the measured ~8;
    * 0-3 s near-field speech (phrase RMS 190), a 3.6-12 s fade, then
      12-90 s far-field phrases whose 10 s speech-band RMS sits inside the
      measured T80 range 13.3-46.4 separated by 1.5-4.5 s pauses, one of them 66.06-70.62 s;
    * a 6 ms stop-button transient at 91.3 s, peak 5414.
    """
    rng = np.random.default_rng(seed)
    n = int(92.0 * SR)
    f = np.fft.rfftfreq(n, 1 / SR)
    amp = np.where((f >= 10) & (f <= 80), 1 / np.maximum(f, 1) ** 2, 0)
    rumble = np.fft.irfft(amp * np.exp(1j * rng.uniform(0, 2 * np.pi, f.size)), n)
    rumble *= 165.0 / _rms(rumble)
    floor = rng.standard_normal(n)
    floor *= 5.0 / _rms(floor)

    speech = np.zeros(n)
    on = np.zeros(n, bool)
    phrases = [(0.2, 2.9, 190.0)]
    cur = 3.6
    for level in (110.0, 70.0, 45.0):
        phrases.append((cur, cur + 2.0, level))
        cur += 2.9
    cur = 12.0
    while cur < 89.5:
        end = min(cur + 1.5 + 3.0 * rng.random(), 89.6)
        if cur < T80_LONG_PAUSE[0] < end:
            end = T80_LONG_PAUSE[0]
        phrases.append((cur, end, float(28 + 20 * rng.random())))
        nxt = end + 1.5 + 3.0 * rng.random()
        if end == T80_LONG_PAUSE[0] or end < T80_LONG_PAUSE[1] and nxt > T80_LONG_PAUSE[0]:
            nxt = max(nxt, T80_LONG_PAUSE[1])
        cur = nxt
    for a, b, level in phrases:
        _syllables(rng, n, a, b, level, speech, on)

    c0, m = int(91.3 * SR), int(0.006 * SR)
    burst = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.0012 * SR))
    bg = rumble + floor + speech
    lo_c, hi_c = 0.0, 2.0 * 5414.0 / np.max(np.abs(burst))
    for _ in range(60):  # scale the click so the recording peak is 5414
        mid = (lo_c + hi_c) / 2
        if np.max(np.abs(bg[c0:c0 + m] + mid * burst)) > 5414.0:
            hi_c = mid
        else:
            lo_c = mid
    x = bg.copy()
    x[c0:c0 + m] += lo_c * burst
    pcm = np.clip(np.round(x), -32768, 32767).astype("<i2")
    return {"pcm": pcm.tobytes(), "speech": speech, "on": on, "rumble": rumble}


@pytest.fixture(scope="module")
def t80():
    return build_t80_like()


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64)


# ── preprocess_speech_pcm16 on the T80 profile ─────────────────────────────


def test_synthetic_t80_matches_measured_profile(t80):
    x = _samples(t80["pcm"])
    t = np.arange(x.size) / SR
    assert np.max(np.abs(x)) == 5414
    speech_band = _band(t80["speech"], 120, 6000)
    buckets = [_rms(speech_band[(t >= a) & (t < min(a + 10, 90))]) for a in range(12, 90, 10)]
    assert all(13.3 <= b <= 46.4 for b in buckets), buckets  # T80 measured 13.3-46.4
    assert _rms(t80["rumble"]) == pytest.approx(165.0)
    assert _rms(x) == pytest.approx(184.4, rel=0.10)  # T80 global RMS 184.4


def test_peak_normalisation_leaves_far_field_speech_below_minus_40_dbfs(t80):
    # Why the old approach failed: the 91.3 s transient pins peak gain at ~6x.
    x = _samples(t80["pcm"])
    t = np.arange(x.size) / SR
    normalised = x * (32767.0 / np.max(np.abs(x)))
    far_speech = (t >= 12) & (t < 90) & t80["on"]
    assert _dbfs(_rms(_band(normalised, 300, 3400)[far_speech])) < -40.0


def test_preprocess_lifts_far_field_speech_keeps_pauses_quiet_and_limits_transient(t80):
    pcm = t80["pcm"]
    out, stats = preprocess_speech_pcm16(pcm)
    assert stats.applied is True
    assert stats.reason == "dynamic_agc"
    assert len(out) == len(pcm)
    x, y = _samples(pcm), _samples(out)
    t = np.arange(x.size) / SR

    def rumble_to_speech(sig):
        return _rms(_band(sig, 10, 80, 2)) / _rms(_band(sig, 300, 3400))

    # Rumble relative to the speech band drops by more than 15 dB.
    assert 20 * np.log10(rumble_to_speech(x) / rumble_to_speech(y)) > 15.0

    far_speech = (t >= 12) & (t < 90) & t80["on"]
    assert -26.0 <= _dbfs(_rms(y[far_speech])) <= -15.0
    pause = (t >= T80_LONG_PAUSE[0]) & (t < T80_LONG_PAUSE[1])
    assert _dbfs(_rms(y[pause])) < -42.0
    transient = y[int(91.2 * SR):int(91.5 * SR)]
    assert np.max(np.abs(transient)) < 32000
    assert np.max(np.abs(y)) < 32000  # soft limiter: no sample at full scale
    # Near-field opening gets far less gain than the far-field stretch.
    near = (t < 3) & t80["on"]
    gain_near = _rms(y[near]) / _rms(_band(x, 120, 6000)[near])
    gain_far = _rms(y[far_speech]) / _rms(_band(x, 120, 6000)[far_speech])
    assert gain_near < 30.0
    assert gain_far > 40.0


def test_preprocess_gain_is_capped_by_noise_floor():
    rng = np.random.default_rng(5)
    n = SR * 6
    # Floor RMS ~60 -> effective max gain 420/60 = 7 even though speech is quiet.
    x = rng.standard_normal(n) * 60.0
    t = np.arange(n) / SR
    x += np.where((t % 1.0) < 0.4, 400.0 * np.sin(2 * np.pi * 440 * t), 0.0)
    out, stats = preprocess_speech_pcm16(np.round(x).astype("<i2").tobytes())
    assert stats.applied is True
    assert stats.max_gain <= 420.0 / stats.noise_floor_rms + 1e-6
    assert stats.max_gain < 8.0


def test_preprocess_healthy_near_field_speech_gets_unity_gain():
    t = np.arange(SR * 4) / SR
    env = np.where((t % 0.5) < 0.3, 1.0, 0.02)
    x = 5000.0 * env * np.sin(2 * np.pi * 300 * t)
    out, stats = preprocess_speech_pcm16(np.round(x).astype("<i2").tobytes())
    assert stats.reason == "hpf_only"
    assert stats.max_gain == stats.min_gain == 1.0


@pytest.mark.parametrize(
    "pcm,reason",
    [
        (b"", "empty"),
        (b"\x05", "empty"),
        (b"\x00\x00" * SR * 3, "digital_silence"),
        (b"\x01\x00" * SR * 3, "no_speech"),
        (b"\x10\x00" * 1000, "too_short"),
    ],
)
def test_preprocess_passes_trivial_input_through_untouched(pcm, reason):
    out, stats = preprocess_speech_pcm16(pcm)
    assert out is pcm
    assert stats.applied is False
    assert stats.reason == reason


def test_preprocess_refuses_to_amplify_stationary_hiss():
    rng = np.random.default_rng(7)
    hiss = np.clip(rng.normal(0, 45.0, SR * 20), -200, 200).astype("<i2").tobytes()
    out, stats = preprocess_speech_pcm16(hiss)
    assert out is hiss
    assert (stats.applied, stats.reason) == (False, "no_speech")
    assert stats.snr_p95 < 2.2


def test_preprocess_long_all_zero_input_short_circuits():
    pcm = bytes(2 * int(465.42 * SR))
    out, stats = preprocess_speech_pcm16(pcm)
    assert out is pcm and stats.reason == "digital_silence"


def test_preprocess_keeps_odd_trailing_byte(t80):
    pcm = t80["pcm"][: SR * 2 * 10] + b"\x7f"
    out, stats = preprocess_speech_pcm16(pcm)
    assert stats.applied is True
    assert len(out) == len(pcm) and out[-1:] == b"\x7f"


# ── StreamingSpeechAGC ─────────────────────────────────────────────────────


def _stream(agc, pcm, rng):
    outs, pos = [], 0
    while pos < len(pcm):
        k = int(rng.integers(1, 9601))  # odd sizes split samples across chunks
        outs.append(agc.process_chunk(pcm[pos:pos + k]))
        pos += k
    return b"".join(outs)


def test_streaming_agc_lifts_far_field_speech_and_preserves_length(t80):
    pcm = t80["pcm"]
    agc = StreamingSpeechAGC(24000)
    out = _stream(agc, pcm, np.random.default_rng(3))
    assert len(out) == len(pcm)
    x, y = _samples(pcm), _samples(out)
    t = np.arange(x.size) / SR
    far_speech = (t >= 12) & (t < 90) & t80["on"]
    raw_far = _rms(_band(x, 120, 6000)[far_speech])
    assert _dbfs(_rms(y[far_speech])) - _dbfs(raw_far) > 25.0
    pause_inner = (t >= T80_LONG_PAUSE[0] + 0.5) & (t < T80_LONG_PAUSE[1])
    assert _dbfs(_rms(y[pause_inner])) < -42.0
    assert np.max(np.abs(y)) < 32000


def test_streaming_agc_odd_chunks_carry_a_byte_and_reset_clears_state():
    agc = StreamingSpeechAGC(24000)
    assert agc.process_chunk(b"\x01") == b""
    assert len(agc.process_chunk(b"\x00\x02\x00")) == 4
    agc.reset()
    assert agc.process_chunk(b"\x00\x00") == b"\x00\x00"


def test_streaming_agc_does_not_amplify_silence_or_stationary_hiss():
    rng = np.random.default_rng(7)
    hiss = np.clip(rng.normal(0, 45.0, SR * 20), -200, 200).astype("<i2").tobytes()
    agc = StreamingSpeechAGC(24000)
    out = b"".join(agc.process_chunk(hiss[i:i + 4800]) for i in range(0, len(hiss), 4800))
    assert agc.current_gain == 1.0
    assert abs(_rms(_samples(out)) - _rms(_samples(hiss))) < 0.05 * _rms(_samples(hiss))

    zero = bytes(4800)
    agc.reset()
    assert all(agc.process_chunk(zero) == zero for _ in range(20))
    assert agc.current_gain == 1.0


# ── split_pcm_on_pauses ────────────────────────────────────────────────────


def _phrases_with_pauses(total_sec, pauses, level=600.0, seed=11):
    """Speech everywhere except the given (start, end) pauses."""
    rng = np.random.default_rng(seed)
    n = int(total_sec * SR)
    speech = np.zeros(n)
    on = np.zeros(n, bool)
    edges = [0.0] + [v for p in pauses for v in p] + [total_sec]
    for a, b in zip(edges[0::2], edges[1::2]):
        _syllables(rng, n, a, b, level, speech, on)
    x = speech + rng.standard_normal(n) * 6.0
    return np.round(x).astype("<i2").tobytes()


def test_split_keeps_t80_and_164s_recordings_whole(t80):
    assert split_pcm_on_pauses(t80["pcm"]) == [t80["pcm"]]
    pcm = _phrases_with_pauses(164.8, [(40.0, 41.5), (100.0, 102.0)])
    assert split_pcm_on_pauses(pcm) == [pcm]


def test_split_keeps_all_zero_and_speechless_long_input_whole():
    zero = bytes(2 * int(465.42 * SR))
    assert split_pcm_on_pauses(zero) == [zero]
    rng = np.random.default_rng(2)
    hiss = np.round(rng.standard_normal(SR * 200) * 40).astype("<i2").tobytes()
    assert split_pcm_on_pauses(hiss) == [hiss]


def test_split_cuts_long_recording_at_quietest_pause_centre():
    pauses = [(60.0, 61.0), (120.0, 122.0), (150.0, 150.8), (205.0, 206.2)]
    pcm = _phrases_with_pauses(260.0, pauses)
    segments = split_pcm_on_pauses(pcm)
    assert b"".join(segments) == pcm
    assert len(segments) == 2
    cut_sec = len(segments[0]) / (2 * SR)
    # Window [99 s, 240 s]: candidates at 120-122, 150-150.8, 205-206.2. All
    # are equally quiet, so the longest pause wins and the cut sits at its
    # centre; detected pause edges move by a few 30 ms frames with syllable
    # tails.
    assert abs(cut_sec - 121.0) <= 0.15
    assert all(len(s) / (2 * SR) <= 240.0 for s in segments)


def test_split_without_pause_cuts_only_beyond_hard_limit():
    assert len(split_pcm_on_pauses(_phrases_with_pauses(230.0, []))) == 1
    pcm = _phrases_with_pauses(300.0, [])
    segments = split_pcm_on_pauses(pcm)
    assert b"".join(segments) == pcm
    assert len(segments) == 2
    assert 99.0 <= len(segments[0]) / (2 * SR) <= 240.0


# ── join_segment_texts ─────────────────────────────────────────────────────


def test_join_segment_texts_boundaries():
    assert join_segment_texts(["第一段。", "  ", "第二段"]) == "第一段。第二段"
    assert join_segment_texts(["我们先看部署", "再看测试"]) == "我们先看部署，再看测试"
    assert join_segment_texts(["deploy the", "service now"]) == "deploy the service now"
    # Segments do not overlap in audio: text on both sides of a cut was said
    # twice and is kept twice (W0322 R2).
    assert join_segment_texts(["然后我们讨论方案", "讨论方案的细节"]) == "然后我们讨论方案，讨论方案的细节"
    assert join_segment_texts(["第一段", "[片段2转录失败]", "第三段"]) == "第一段\n[片段2转录失败]\n第三段"
    assert join_segment_texts(["行一\n行二", "行三"]) == "行一\n行二，行三"
    assert join_segment_texts([]) == ""


# ── truncated / looping transcript guards ──────────────────────────────────

T80_SENTENCES = [
    "所以这个命令其实是内置在我们自己的工作台里面的，它和后端的接口基本上是一一对应的。",
    "但是我觉得现在有一个比较大的问题，就是这个后端还是太重了，很多功能根本用不上。",
    "前面我们讨论过一个可以单独移植出来的轻量版本，应该先把那个版本整理清楚再动手。",
    "之后要把后端里那些没有用的功能尽量拆掉，重新开一个干净的分支去做精简。",
    "目标是变成一个非常简洁、只保留我们真正需要的能力的服务，代码仓库里不要有干扰。",
    "那些额外的功能堆在仓库里，是我们迟迟上不了线的一个很重要的原因。",
    "除了这个问题之外，还有部署流程和监控告警两块，也需要一起重新梳理一遍。",
    "包括刚才提到的接口兼容性，整合之后我们还要回头再检查一次它的逻辑是否一致。",
    "如果我们想要提高这个项目的整体质量，就得在修的过程中把每个模块的边界讲清楚。",
    "我希望下周能先出一个精简版的设计稿，大家看过之后再决定具体的拆分顺序。",
    "测试这边也要同步准备，至少把核心路径的回归用例补齐，避免拆完之后行为悄悄变掉。",
    "最后是文档，每拆掉一个功能就在变更记录里写清楚原因，方便以后追溯。",
    "还有一点是权限，新服务上线前要和安全那边对一下账号和密钥的管理方式。",
    "性能方面先不追求极致，但启动时间和内存占用要有基线数据，后面才好比较。",
    "如果这些都顺利，月底可以先在内部灰度，收集一周的反馈再决定是否全面切换。",
    "大概就是这些，大家有补充的话今天下班前发给我，我汇总之后明天上午同步。",
]
T80_REALTIME_TEXT = "".join(T80_SENTENCES)
T80_INPUT_TEXT = "".join(T80_SENTENCES[:2]).replace("，", "").replace("。", "")
T80_AUDIO_SEC = 91.6


def test_t80_fixture_shape():
    assert len(T80_REALTIME_TEXT) >= 560
    assert 70 <= len(T80_INPUT_TEXT) <= 100
    assert transcription_similarity_ratio(T80_REALTIME_TEXT, T80_INPUT_TEXT) < 0.60


def test_truncated_guard_recognises_t80():
    assert is_likely_truncated_input_transcript(T80_REALTIME_TEXT, T80_INPUT_TEXT, T80_AUDIO_SEC)


@pytest.mark.parametrize(
    "emitted,transcript,audio_sec",
    [
        # Short recording: the answer guard is never relaxed.
        ("什么是供应链管理它是指商品从生产到消费流动的计划", "什么是供应链管理", 6.0),
        (T80_REALTIME_TEXT, T80_INPUT_TEXT, 19.9),
        # Transcript dense enough to be complete (>= 1.8 chars/s).
        (T80_REALTIME_TEXT, T80_REALTIME_TEXT[:200], T80_AUDIO_SEC),
        # Output not materially longer.
        (T80_REALTIME_TEXT[:150], T80_INPUT_TEXT, T80_AUDIO_SEC),
        # Pure answer unrelated to the 30 s recording's transcript.
        (
            "第一，项目延期到下个月中旬；第二，预算维持不变但需要重新分配给测试团队；"
            "第三，周五之前由产品经理提交新的里程碑计划并同步给所有相关负责人。"
            "另外还提到要补充用户访谈的样本数量，并且尽快确定外部合作方的报价范围。",
            "帮我总结一下今天下午会上定下来的事情",
            30.0,
        ),
        # Faithful 6000-char prefix + 8000-char answer: beyond 12 chars/s.
        (
            "".join(chr(0x3400 + i) for i in range(6000)) + "".join(chr(0x9000 + i) for i in range(8000)),
            "".join(chr(0x3400 + i) for i in range(6000)),
            30.0,
        ),
        ("", T80_INPUT_TEXT, T80_AUDIO_SEC),
        (T80_REALTIME_TEXT, "", T80_AUDIO_SEC),
    ],
)
def test_truncated_guard_rejects(emitted, transcript, audio_sec):
    assert not is_likely_truncated_input_transcript(emitted, transcript, audio_sec)


def test_runaway_repetition_guard():
    assert is_runaway_repetition_transcript("我们再看一下这个问题" * 30, 60.0)
    assert is_runaway_repetition_transcript(T80_REALTIME_TEXT * 4, 20.0)  # > 18 chars/s
    assert not is_runaway_repetition_transcript(T80_REALTIME_TEXT, T80_AUDIO_SEC)
    assert not is_runaway_repetition_transcript("好的好的好的", 2.0)  # too short to judge


def test_scaled_suspicious_output_chars():
    assert scaled_suspicious_output_chars(20.0) == 120
    assert scaled_suspicious_output_chars(91.6) == 201
    assert scaled_suspicious_output_chars(200.0) == 440
    assert scaled_suspicious_output_chars(-5.0) == 120


# ── TurnSession (streaming lane) integration ───────────────────────────────


async def _drive_marker_text(session, text, item_id=None, chunk=7):
    first = {"delta": MARKER}
    if item_id:
        first["item_id"] = item_id
    await session._on_text_delta(first)
    for i in range(0, len(text), chunk):
        data = {"delta": text[i:i + chunk]}
        if item_id:
            data["item_id"] = item_id
        await session._on_text_delta(data)


def _replacements(session):
    return [p for p in sent_text_payloads(session) if p.get("isNewResponse") is True]


@pytest.mark.asyncio
async def test_turn_session_keeps_t80_output_against_truncated_transcript(caplog):
    session = make_session()
    set_processed_audio_duration(session, T80_AUDIO_SEC)
    await _drive_marker_text(session, T80_REALTIME_TEXT)
    await session._on_input_transcription_completed({"transcript": T80_INPUT_TEXT})
    with caplog.at_level(logging.WARNING):
        await session._on_response_done({})
        await drain_finalize(session)
    assert _replacements(session) == []
    assert session._emitted_text == T80_REALTIME_TEXT
    assert "input transcript looks truncated" in caplog.text
    assert session._websocket.sent_payloads[-1]["status"] == "idle"


@pytest.mark.asyncio
async def test_turn_session_answer_guard_still_replaces_30s_pure_answer():
    session = make_session()
    set_processed_audio_duration(session, 30.0)
    answer = (
        "第一，项目延期到下个月中旬；第二，预算维持不变但需要重新分配给测试团队；"
        "第三，周五之前由产品经理提交新的里程碑计划并同步给所有相关负责人。"
    )
    question = "帮我总结一下今天下午会上定下来的事情"
    await _drive_marker_text(session, answer)
    await session._on_input_transcription_completed({"transcript": question})
    await session._on_response_done({})
    await drain_finalize(session)
    assert [p["content"] for p in _replacements(session)] == [question]


@pytest.mark.asyncio
async def test_turn_session_keeps_output_when_transcript_is_a_repeat_loop():
    session = make_session()
    set_processed_audio_duration(session, 60.0)
    loop = "我们再看一下这个问题" * 30
    await _drive_marker_text(session, "短")
    await session._on_input_transcription_completed({"transcript": loop})
    await session._on_response_done({})
    await drain_finalize(session)
    assert _replacements(session) == []


@pytest.mark.asyncio
async def test_turn_session_suspicious_threshold_scales_with_audio_length():
    # make_session uses a 60-char base; at 90 s the threshold is 2.2 * 90 = 198,
    # so a 180-char realtime output of a 400-char transcript is suspicious.
    session = make_session(answer_guard_min_similarity=0.0)
    set_processed_audio_duration(session, 90.0)
    realtime, full = T80_REALTIME_TEXT[:180], T80_REALTIME_TEXT[:400]
    await _drive_marker_text(session, realtime)
    await session._on_input_transcription_completed({"transcript": full})
    await session._on_response_done({})
    await drain_finalize(session)
    assert [p["content"] for p in _replacements(session)] == [full]


@pytest.mark.asyncio
async def test_turn_session_strips_marker_repeated_by_second_output_item():
    session = make_session()
    body_one = "第一个输出项的内容到这里结束。"
    body_two = "第二个输出项继续后面的内容。"
    await _drive_marker_text(session, body_one, item_id="item_1")
    for delta in ["下面是不改变", "语言的语音识别结果：", "\n", "\n", body_two]:
        await session._on_text_delta({"delta": delta, "item_id": "item_2"})
    await session._on_response_done({})
    await drain_finalize(session)
    assert session._emitted_text == f"{body_one}\n{body_two}"
    assert all("语音识别结果" not in p["content"] for p in sent_text_payloads(session))


@pytest.mark.asyncio
async def test_turn_session_second_item_without_marker_is_kept_verbatim():
    session = make_session()
    await _drive_marker_text(session, "第一段。", item_id="item_1")
    for delta in ["下面", "是新的内容"]:
        await session._on_text_delta({"delta": delta, "item_id": "item_2"})
    await session._on_response_done({})
    await drain_finalize(session)
    assert session._emitted_text == "第一段。\n下面是新的内容"


@pytest.mark.asyncio
async def test_turn_session_marks_incomplete_response_before_idle(caplog):
    session = make_session()
    await _drive_marker_text(session, "说到一半的内容")
    with caplog.at_level(logging.WARNING):
        await session._on_response_done(
            {"response": {"status": "incomplete", "status_details": {"reason": "max_output_tokens"}}}
        )
        await drain_finalize(session)
    texts = sent_text_payloads(session)
    assert texts[-1]["content"] == "\n" + INCOMPLETE_TURN_MARKER
    assert session._emitted_text.endswith("\n" + INCOMPLETE_TURN_MARKER)
    assert session._websocket.sent_payloads[-1]["status"] == "idle"
    assert "status=incomplete" in caplog.text


@pytest.mark.asyncio
async def test_turn_session_completed_response_gets_no_incomplete_marker():
    session = make_session()
    await _drive_marker_text(session, "完整的内容")
    await session._on_response_done({"response": {"status": "completed"}})
    await drain_finalize(session)
    assert INCOMPLETE_TURN_MARKER not in session._emitted_text


@pytest.mark.asyncio
async def test_turn_session_incomplete_rescued_by_fallback_is_still_marked():
    session = make_session()
    set_processed_audio_duration(session, 2.0)
    realtime_text = "短输出"
    fallback = "这是完整的输入转写内容，比模型输出长得多，覆盖了整段录音的每一句话，包括结尾的安排。"
    await _drive_marker_text(session, realtime_text)
    await session._on_input_transcription_completed({"transcript": fallback})
    await session._on_response_done({"response": {"status": "incomplete"}})
    await drain_finalize(session)
    # A completed input transcript does not prove it covers the recording
    # (T80), so the abnormal end stays visible (W0322 R2 Severe #7).
    assert [p["content"] for p in _replacements(session)] == [fallback]
    assert session._emitted_text == f"{fallback}\n{INCOMPLETE_TURN_MARKER}"


@pytest.mark.asyncio
async def test_agc_enabled_noise_turn_is_still_blocked_and_cache_gets_raw_audio(caplog):
    session, client, audio_cache = make_recording_gate_session()
    session._config = dataclasses.replace(session._config, input_agc_enabled=True)
    rng = np.random.default_rng(9)
    hiss = np.clip(rng.normal(0, 45.0, SR * 3), -200, 200).astype("<i2").tobytes()
    for i in range(0, len(hiss), 4800):
        await session._handle_audio_bytes(hiss[i:i + 4800])
    with caplog.at_level(logging.INFO):
        await session._handle_stop_recording({"turn_id": 111})
    assert b"".join(audio_cache.accumulated) == hiss
    assert client.commit_calls == 0  # no_speech_signal: provider never asked
    assert sum(p.get("content") == NO_SPEECH_PLACEHOLDER for p in sent_text_payloads(session)) == 1
    assert audio_cache.enqueued[0]["outcome"] == "no_speech_signal"
    # Stationary hiss gets no gain on the way to the provider either.
    sent = _samples(b"".join(client.sent_audio))
    assert abs(_rms(sent) - _rms(_samples(hiss))) < 0.05 * _rms(_samples(hiss))


@pytest.mark.asyncio
async def test_agc_enabled_sends_gained_audio_but_caches_raw(t80):
    session, client, audio_cache = make_recording_gate_session()
    session._config = dataclasses.replace(session._config, input_agc_enabled=True)
    pcm = t80["pcm"][: SR * 2 * 30]
    for i in range(0, len(pcm), 4800):
        await session._handle_audio_bytes(pcm[i:i + 4800])
    assert b"".join(audio_cache.accumulated) == pcm
    sent = b"".join(client.sent_audio)
    assert len(sent) == len(pcm) and sent != pcm
    assert session._processed_audio_bytes == len(pcm)
    t = np.arange(len(pcm) // 2) / SR
    far = (t >= 12) & t80["on"][: t.size]
    assert _rms(_samples(sent)[far]) > 10 * _rms(_band(_samples(pcm), 120, 6000)[far])


def test_from_env_turns_agc_on_and_dataclass_default_is_off(monkeypatch):
    from micwave_core.realtime_server import TurnSessionConfig

    monkeypatch.delenv("BRAINWAVE_INPUT_AGC_ENABLED", raising=False)
    assert TurnSessionConfig.from_env().input_agc_enabled is True
    monkeypatch.setenv("BRAINWAVE_INPUT_AGC_ENABLED", "0")
    assert TurnSessionConfig.from_env().input_agc_enabled is False
    assert make_session()._config.input_agc_enabled is False



# ── W0322 Round 2: counterexamples from the cross-model review ──────────────

SUPPLY_CHAIN_QUESTION = "什么是供应链管理"
SUPPLY_CHAIN_ANSWER = (
    "什么是供应链管理？供应链管理是指从原材料采购、生产制造、仓储物流到最终交付给客户的"
    "全流程计划、协调与优化的管理体系，它的核心目标是在保证服务水平的前提下尽量降低整体成本，"
    "并提高各个环节之间的信息透明度和响应速度。"
)


def _question_then_silence(question_sec=3.0, total_sec=30.0, question_first=True, seed=5):
    """One 3 s spoken question and 27 s of near-silence (floor RMS 5)."""
    rng = np.random.default_rng(seed)
    q = np.zeros(int(question_sec * SR))
    t0 = 0.0
    while t0 < question_sec - 0.25:
        i0, i1 = int(t0 * SR), int(min(t0 + 0.2, question_sec) * SR)
        tt = np.arange(i1 - i0) / SR
        f0 = 150 + 60 * rng.random()
        q[i0:i1] += 600 * sum(np.sin(2 * np.pi * f0 * h * tt) / h for h in range(1, 12)) * (
            np.sin(np.pi * np.arange(i1 - i0) / (i1 - i0)) ** 2
        )
        t0 += 0.28
    silence = np.zeros(int((total_sec - question_sec) * SR))
    x = np.concatenate([q, silence] if question_first else [silence, q])
    x += rng.standard_normal(x.size) * 5.0
    return np.round(x).astype("<i2").tobytes()


def test_supply_chain_fixture_is_the_reviewed_shape():
    assert len(SUPPLY_CHAIN_ANSWER) >= 90 and SUPPLY_CHAIN_ANSWER.startswith(SUPPLY_CHAIN_QUESTION)
    # The text alone looks like a stalled transcript to the R1 rules.
    assert transcription_similarity_ratio(SUPPLY_CHAIN_ANSWER, SUPPLY_CHAIN_QUESTION) < 0.60


@pytest.mark.parametrize("question_first", [True, False])
def test_speech_activity_of_short_question_in_long_silence(question_first):
    pcm = _question_then_silence(question_first=question_first)
    active, tail = measure_speech_activity_pcm16(pcm, SR)
    assert 1.0 <= active <= 3.5
    assert tail <= 3.5 and (tail == 0.0) == question_first
    _, stats = preprocess_speech_pcm16(pcm, SR)
    assert (stats.active_speech_sec, stats.tail_active_speech_sec) == (active, tail)


def test_speech_activity_of_t80_and_trivial_input(t80):
    active, tail = measure_speech_activity_pcm16(t80["pcm"], SR)
    assert active >= 15.0 and tail >= 4.0
    _, stats = preprocess_speech_pcm16(t80["pcm"], SR)
    assert (stats.active_speech_sec, stats.tail_active_speech_sec) == (active, tail)
    assert measure_speech_activity_pcm16(b"\x00\x00" * SR, SR) == (0.0, 0.0)
    assert preprocess_speech_pcm16(b"\x00\x00" * SR, SR)[1].active_speech_sec == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # text only: 8-char question < 24 chars
        {"active_speech_sec": 2.0, "tail_active_speech_sec": 0.0},
        {"active_speech_sec": 2.0, "tail_active_speech_sec": 2.0},
        {"active_speech_sec": 2.0},
    ],
)
def test_truncated_guard_rejects_short_question_with_restating_answer(kwargs):
    assert not is_likely_truncated_input_transcript(
        SUPPLY_CHAIN_ANSWER, SUPPLY_CHAIN_QUESTION, 30.0, **kwargs
    )


def test_truncated_guard_acoustic_gates_on_t80():
    args = (T80_REALTIME_TEXT, T80_INPUT_TEXT, T80_AUDIO_SEC)
    assert is_likely_truncated_input_transcript(*args, active_speech_sec=25.0, tail_active_speech_sec=10.0)
    # Real T80 measured active_ratio 0.229 (~21 s): still recognised.
    assert is_likely_truncated_input_transcript(*args, active_speech_sec=21.0, tail_active_speech_sec=8.0)
    assert not is_likely_truncated_input_transcript(*args, active_speech_sec=14.9, tail_active_speech_sec=10.0)
    assert not is_likely_truncated_input_transcript(*args, active_speech_sec=25.0, tail_active_speech_sec=3.9)
    # Text-only fallback: >= 30 s and >= 24 chars.
    assert is_likely_truncated_input_transcript(*args)
    assert not is_likely_truncated_input_transcript(T80_REALTIME_TEXT[:300], T80_INPUT_TEXT, 29.9)


def _feed_frames(session, pcm):
    for i in range(0, len(pcm), 4800):
        session._accumulate_turn_signal_metrics(pcm[i:i + 4800])
    set_processed_audio_duration(session, len(pcm) / (2 * SR))


@pytest.mark.asyncio
@pytest.mark.parametrize("with_audio", [True, False])
async def test_turn_session_answer_guard_replaces_restating_answer_after_long_silence(with_audio):
    session = make_session()
    if with_audio:
        _feed_frames(session, _question_then_silence())
        assert session._turn_speech_activity_secs()[0] < 3.5
    else:
        set_processed_audio_duration(session, 30.0)
        assert session._turn_speech_activity_secs() is None
    await _drive_marker_text(session, SUPPLY_CHAIN_ANSWER)
    await session._on_input_transcription_completed({"transcript": SUPPLY_CHAIN_QUESTION})
    await session._on_response_done({})
    await drain_finalize(session)
    assert [p["content"] for p in _replacements(session)] == [SUPPLY_CHAIN_QUESTION]


@pytest.mark.asyncio
async def test_turn_session_keeps_t80_output_with_measured_t80_audio(t80):
    session = make_session()
    _feed_frames(session, t80["pcm"])
    active, tail = session._turn_speech_activity_secs()
    assert active >= 15.0 and tail >= 4.0
    await _drive_marker_text(session, T80_REALTIME_TEXT)
    await session._on_input_transcription_completed({"transcript": T80_INPUT_TEXT})
    await session._on_response_done({})
    await drain_finalize(session)
    assert _replacements(session) == []
    assert session._emitted_text == T80_REALTIME_TEXT


def test_join_segment_texts_keeps_words_repeated_across_a_cut():
    assert join_segment_texts(["我想说我们", "我们需要再考虑"]) == "我想说我们，我们需要再考虑"
    assert join_segment_texts(["we need to", "to think again"]) == "we need to to think again"
    parts = ["好的好的", "好的", "那我们", "那我们开始。", "开始吧"]
    joined = join_segment_texts(parts)
    assert [c for c in joined if c not in "，\n "] == [c for c in "".join(parts)]


@pytest.mark.asyncio
async def test_turn_session_body_ending_like_marker_start_survives_third_item():
    session = make_session()
    await _drive_marker_text(session, "第一项。", item_id="item_1")
    # Item 2 is only "下面是", which could still begin a repeated marker, so it
    # is held; item 3 arriving proves it was body text.
    await session._on_text_delta({"delta": "下面是", "item_id": "item_2"})
    await session._on_text_delta({"delta": "第三项。", "item_id": "item_3"})
    await session._on_response_done({})
    await drain_finalize(session)
    assert session._emitted_text == "第一项。\n下面是\n第三项。"


@pytest.mark.asyncio
async def test_turn_session_item_ending_in_marker_start_is_flushed_on_timeout_finalize():
    session = make_session()
    await _drive_marker_text(session, "第一项。", item_id="item_1")
    await session._on_text_delta({"delta": "下面是", "item_id": "item_2"})
    await session._finalize_turn("timeout")
    assert session._emitted_text == "第一项。\n下面是"


LOOP_TEXT = "我们再看一下这个问题" * 45


@pytest.mark.asyncio
async def test_turn_session_no_marker_answer_over_completed_loop_is_marked_incomplete():
    session = make_session()
    set_processed_audio_duration(session, 60.0)
    await session._on_text_delta({"delta": "这是模型对问题的回答"})
    await session._on_input_transcription_completed({"transcript": LOOP_TEXT})
    await session._on_response_done({})
    await drain_finalize(session)
    assert [p["content"] for p in _replacements(session)] == [INCOMPLETE_TURN_MARKER]
    assert session._emitted_text == INCOMPLETE_TURN_MARKER


@pytest.mark.asyncio
async def test_turn_session_no_marker_answer_over_partial_loop_keeps_0809_contract():
    session = make_session(input_transcript_grace_sec=0.01)
    set_processed_audio_duration(session, 60.0)
    # Past max_prefix_deltas (20) the no-marker text is streamed as-is.
    for ch in "这是模型没有加标记前缀但仍然逐句输出的一段比较长的内容":
        await session._on_text_delta({"delta": ch})
    assert session._emitted_without_marker
    await session._on_input_transcription_delta({"delta": LOOP_TEXT})
    await session._on_response_done({})
    await drain_finalize(session)
    assert _replacements(session) == []


@pytest.mark.asyncio
async def test_turn_session_marker_loop_output_is_rescued_by_completed_transcript():
    # Answer guard off: only the loop check may replace the output.
    session = make_session(answer_guard_min_similarity=0.0)
    set_processed_audio_duration(session, 60.0)
    await _drive_marker_text(session, LOOP_TEXT)
    faithful = T80_REALTIME_TEXT[:300]
    await session._on_input_transcription_completed({"transcript": faithful})
    await session._on_response_done({})
    await drain_finalize(session)
    assert [p["content"] for p in _replacements(session)] == [faithful]


@pytest.mark.asyncio
async def test_turn_session_marker_loop_output_and_loop_transcript_is_marked_incomplete():
    session = make_session()
    set_processed_audio_duration(session, 60.0)
    await _drive_marker_text(session, LOOP_TEXT)
    await session._on_input_transcription_completed({"transcript": "这个那个这个那个" * 40})
    await session._on_response_done({})
    await drain_finalize(session)
    assert [p["content"] for p in _replacements(session)] == [INCOMPLETE_TURN_MARKER]


@pytest.mark.asyncio
async def test_turn_session_failed_empty_response_is_incomplete_not_no_speech():
    session = make_session()
    set_processed_audio_duration(session, 5.0)
    await session._on_response_done({"response": {"status": "failed"}})
    await drain_finalize(session)
    contents = [p["content"] for p in sent_text_payloads(session)]
    assert contents == [INCOMPLETE_TURN_MARKER]
    assert NO_SPEECH_PLACEHOLDER not in session._emitted_text
    assert session._websocket.sent_payloads[-1]["status"] == "idle"


@pytest.mark.asyncio
async def test_turn_session_incomplete_with_longer_completed_transcript_keeps_marker():
    session = make_session()
    set_processed_audio_duration(session, 6.0)
    await _drive_marker_text(session, "仅仅")
    await session._on_input_transcription_completed({"transcript": "仅仅这是不完整的开头"})
    await session._on_response_done({"response": {"status": "incomplete"}})
    await drain_finalize(session)
    assert session._emitted_text.endswith("\n" + INCOMPLETE_TURN_MARKER)
    assert session._emitted_text.count(INCOMPLETE_TURN_MARKER) == 1


# W0322 R4 Severe #1: a looping model output is checked on its own, before
# and regardless of the input transcript; only a completed non-looping
# transcript may stand in, anything else marks the turn incomplete.


def _short_grace_session(**overrides):
    values = dict(
        input_transcript_grace_sec=0.01,
        suspicious_input_transcript_grace_sec=0.01,
        input_transcript_grace_per_audio_sec=0.0,
    )
    values.update(overrides)
    session = make_session(**values)
    set_processed_audio_duration(session, 60.0)
    return session


async def _drive_unmarked_text(session, text):
    # Past max_prefix_deltas (20) the no-marker text is streamed as-is.
    for ch in text:
        await session._on_text_delta({"delta": ch})
    assert session._emitted_without_marker


async def _finalize_completed(session):
    await session._on_response_done({"response": {"status": "completed"}})
    await drain_finalize(session)


def _assert_marked_incomplete(session):
    assert [p["content"] for p in _replacements(session)] == [INCOMPLETE_TURN_MARKER]
    assert session._emitted_text == INCOMPLETE_TURN_MARKER
    assert "我们再看一下这个问题" not in session._emitted_text
    assert session._websocket.sent_payloads[-1]["status"] == "idle"


@pytest.mark.asyncio
async def test_turn_session_marker_loop_with_failed_empty_transcript_is_incomplete():
    session = _short_grace_session()
    await _drive_marker_text(session, LOOP_TEXT)
    await session._on_input_transcription_failed(
        {"error": {"type": "server_error", "code": "x", "message": "m"}}
    )
    assert session._input_transcript_text == ""
    await _finalize_completed(session)
    _assert_marked_incomplete(session)


@pytest.mark.asyncio
async def test_turn_session_marker_loop_with_partial_transcript_is_incomplete():
    session = _short_grace_session()
    await _drive_marker_text(session, LOOP_TEXT)
    await session._on_input_transcription_delta({"delta": "我们先讨论第一部分的安排"})
    assert not session._input_transcript_completed
    await _finalize_completed(session)
    _assert_marked_incomplete(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", ["", "我们先讨论第一部分的安排"])
async def test_turn_session_unmarked_loop_without_completed_transcript_is_incomplete(partial):
    session = _short_grace_session()
    await _drive_unmarked_text(session, LOOP_TEXT)
    if partial:
        await session._on_input_transcription_delta({"delta": partial})
    await _finalize_completed(session)
    _assert_marked_incomplete(session)


@pytest.mark.asyncio
async def test_turn_session_unmarked_loop_is_rescued_by_completed_transcript():
    session = _short_grace_session(answer_guard_min_similarity=0.0)
    await _drive_unmarked_text(session, LOOP_TEXT)
    faithful = T80_REALTIME_TEXT[:300]
    await session._on_input_transcription_completed({"transcript": faithful})
    await _finalize_completed(session)
    assert [p["content"] for p in _replacements(session)] == [faithful]
    assert session._emitted_text == faithful


@pytest.mark.asyncio
async def test_turn_session_loop_output_waits_for_a_late_completed_transcript():
    # marker_seen with the answer guard off would skip the grace wait; a
    # looping output still waits so a late completed transcript can rescue it.
    session = _short_grace_session(
        answer_guard_min_similarity=0.0, suspicious_input_transcript_grace_sec=1.0
    )
    await _drive_marker_text(session, LOOP_TEXT)
    faithful = T80_REALTIME_TEXT[:300]

    async def late():
        await asyncio.sleep(0.1)
        await session._on_input_transcription_completed({"transcript": faithful})

    task = asyncio.create_task(late())
    await _finalize_completed(session)
    await task
    assert [p["content"] for p in _replacements(session)] == [faithful]


# W0322 R5 Severe #1: timeout, provider error and disconnect finalize without
# the response.done barrier (or race it); the loop net lives in _finalize_turn.


async def _finalize_abnormally(session, reason):
    if reason == "error":
        await session._on_error({"error": {"message": "provider broke"}})
    else:
        await session._finalize_turn(reason)


def _idle_count(session):
    return sum(
        1
        for p in session._websocket.sent_payloads
        if p.get("type") == "status" and p.get("status") == "idle"
    )


def _assert_marked_incomplete_once(session):
    reps = _replacements(session)
    assert reps and reps[-1]["content"] == INCOMPLETE_TURN_MARKER
    assert session._emitted_text == INCOMPLETE_TURN_MARKER
    assert session._websocket.sent_payloads[-1]["status"] == "idle"
    assert _idle_count(session) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["timeout", "error", "provider_disconnect_during_response"]
)
@pytest.mark.parametrize("marked", [True, False])
@pytest.mark.parametrize("partial", ["", "我们先讨论第一部分的安排"])
async def test_abnormal_finalize_replaces_loop_output_with_incomplete_marker(
    reason, marked, partial
):
    session = _short_grace_session()
    if marked:
        await _drive_marker_text(session, LOOP_TEXT)
    else:
        await _drive_unmarked_text(session, LOOP_TEXT)
    if partial:
        await session._on_input_transcription_delta({"delta": partial})
    assert len(session._emitted_text) >= 400  # corrector may still hold a tail
    await _finalize_abnormally(session, reason)
    _assert_marked_incomplete_once(session)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["timeout", "error", "provider_disconnect_during_response"]
)
async def test_abnormal_finalize_rescues_loop_output_with_completed_transcript(reason):
    session = _short_grace_session(answer_guard_min_similarity=0.0)
    await _drive_marker_text(session, LOOP_TEXT)
    faithful = T80_REALTIME_TEXT[:300]
    await session._on_input_transcription_completed({"transcript": faithful})
    await _finalize_abnormally(session, reason)
    assert _replacements(session)[-1]["content"] == faithful
    assert session._emitted_text == faithful
    assert session._websocket.sent_payloads[-1]["status"] == "idle"
    assert _idle_count(session) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["timeout", "error", "provider_disconnect_during_response"]
)
async def test_abnormal_finalize_keeps_non_looping_partial_output(reason):
    session = _short_grace_session()
    body = T80_REALTIME_TEXT[:200]
    await _drive_marker_text(session, body)
    await _finalize_abnormally(session, reason)
    assert _replacements(session) == []
    assert session._emitted_text == body
    assert _idle_count(session) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("racer", ["error", "timeout"])
async def test_abnormal_finalize_wins_race_against_waiting_barrier(racer):
    # response.done starts the barrier, which waits for a transcript that a
    # looping output needs; the abnormal path finalizes first.
    session = _short_grace_session(suspicious_input_transcript_grace_sec=1.0)
    await _drive_marker_text(session, LOOP_TEXT)
    await session._on_response_done({"response": {"status": "completed"}})
    await asyncio.sleep(0.02)
    assert session._finalize_task is not None and not session._finalize_task.done()
    assert not session._finalized
    await _finalize_abnormally(session, racer)
    await drain_finalize(session)
    _assert_marked_incomplete_once(session)
    assert [p["content"] for p in _replacements(session)] == [INCOMPLETE_TURN_MARKER]


@pytest.mark.asyncio
@pytest.mark.parametrize("racer", ["error", "timeout"])
async def test_barrier_wins_race_then_abnormal_finalize_is_noop(racer):
    session = _short_grace_session()
    await _drive_marker_text(session, LOOP_TEXT)
    await _finalize_completed(session)
    await _finalize_abnormally(session, racer)
    assert [p["content"] for p in _replacements(session)] == [INCOMPLETE_TURN_MARKER]
    assert session._emitted_text == INCOMPLETE_TURN_MARKER
    assert _idle_count(session) == 1


def _white_noise(rms, sec, seed):
    rng = np.random.default_rng(seed)
    return np.round(rng.standard_normal(int(sec * SR)) * rms).astype("<i2").tobytes()


@pytest.mark.parametrize("level", [45.0, 100.0, 300.0])
def test_streaming_agc_does_not_amplify_stationary_white_noise_at_any_level(level):
    pcm = _white_noise(level, 5.0, seed=int(level))
    agc = StreamingSpeechAGC()
    out = _stream(agc, pcm, np.random.default_rng(1))
    assert len(out) == len(pcm)
    rms_in, rms_out = _rms(_samples(pcm)), _rms(_samples(out))
    assert rms_out < 1.3 * rms_in
    if level == 100.0:
        assert rms_out < 130.0
    assert agc.current_gain < 1.3
    # In 10 ms chunks no frame of the first second gets a gain spike either
    # (the boot floor covers frames before the modulation window fills).
    fine = StreamingSpeechAGC()
    gains = []
    for i in range(0, SR * 2, 480):
        fine.process_chunk(pcm[i:i + 480])
        gains.append(fine.current_gain)
    assert max(gains) < 1.3


def test_streaming_agc_gates_noise_lifts_modulated_speech_and_falls_back():
    rng = np.random.default_rng(21)
    floor = rng.standard_normal(int(12.0 * SR)) * 60.0
    t = np.arange(floor.size) / SR
    speech = np.zeros_like(floor)
    on = (t >= 4.0) & (t < 8.0)
    t0 = 4.0
    while t0 < 7.75:
        i0, i1 = int(t0 * SR), int((t0 + 0.2) * SR)
        tt = np.arange(i1 - i0) / SR
        f0 = 150 + 60 * rng.random()
        speech[i0:i1] = 400 * sum(np.sin(2 * np.pi * f0 * h * tt) / h for h in range(1, 12)) * (
            np.sin(np.pi * np.arange(i1 - i0) / (i1 - i0)) ** 2
        )
        t0 += 0.28
    pcm = np.round(floor + speech).astype("<i2").tobytes()
    agc = StreamingSpeechAGC()
    gains = []
    for i in range(0, len(pcm), 4800):
        agc.process_chunk(pcm[i:i + 4800])
        gains.append(agc.current_gain)
    chunk_t = (np.arange(len(gains)) + 1) * 0.1
    assert max(g for g, ct in zip(gains, chunk_t) if ct <= 4.0) < 1.3  # floor only
    assert max(g for g, ct in zip(gains, chunk_t) if 4.5 <= ct <= 8.0) > 3.0  # speech lifted
    assert max(g for g, ct in zip(gains, chunk_t) if ct >= 10.5) < 1.3  # back on the floor
    assert on.any()


def test_streaming_agc_flush_returns_held_half_sample():
    agc = StreamingSpeechAGC()
    assert agc.process_chunk(b"\x01\x00\x02") == b"\x01\x00"
    assert agc.flush() == b"\x02"
    assert agc.flush() == b""
    assert agc.process_chunk(b"\x03\x00") == b"\x03\x00"


@pytest.mark.parametrize(
    "soft,hard,pause",
    [
        (0.0, 0.03, 0.45),
        (-5.0, 240.0, 0.45),
        (float("nan"), float("nan"), float("nan")),
        (float("inf"), float("inf"), 0.45),
        (1.0, float("inf"), 0.0),
        (2.0, 1.0, -1.0),
    ],
)
def test_split_sanitises_parameters_and_always_terminates(soft, hard, pause):
    import time

    pcm = _phrases_with_pauses(12.0, [(3.0, 3.6), (6.0, 6.7), (9.0, 9.5)])
    timings = []
    for _ in range(3):  # best of 3 keeps a loaded machine from flaking
        started = time.perf_counter()
        segments = split_pcm_on_pauses(pcm, SR, soft_max_sec=soft, hard_max_sec=hard, min_pause_sec=pause)
        timings.append(time.perf_counter() - started)
    assert min(timings) < 0.05
    assert b"".join(segments) == pcm and all(segments)
    # Clamped limits still mean segments of a sensible length (soft >= 1 s,
    # cut no earlier than 0.55 * soft), not one 30 ms frame per segment.
    assert all(len(seg) / (2 * SR) >= 0.5 for seg in segments[:-1])


def test_split_rejects_non_positive_sample_rate():
    with pytest.raises(ValueError):
        split_pcm_on_pauses(b"\x00\x01" * 100, 0)


# A question long enough for the text-only rules (>= 24 chars, 40 s recording)
# with a restating answer: only the measured speech activity can tell it from
# a stalled dictation.
LONG_QUESTION = "请帮我详细解释一下什么是供应链管理以及它在制造业里面到底有什么作用"
LONG_QUESTION_ANSWER = LONG_QUESTION + (
    "？供应链管理是指企业对从原材料采购、零部件生产、仓储运输到最终交付给客户的整个链条进行计划、"
    "协调和优化的过程。在制造业里，它决定了库存水平、交付周期和综合成本，好的供应链管理能让工厂在需求波动时"
    "依然保持稳定的生产节奏，并且降低缺料停线的风险。"
)


def _question_with_rumble(seed=7):
    """4 s question + 36 s of intermittent 30 Hz handling rumble (no speech)."""
    pcm = np.frombuffer(_question_then_silence(4.0, 40.0, seed=seed), "<i2").astype(np.float64)
    t = np.arange(pcm.size) / SR
    bursts = ((t >= 4.0) & ((t % 1.0) < 0.5)).astype(np.float64)
    pcm = pcm + 2000.0 * np.sin(2 * np.pi * 30.0 * t) * bursts
    return np.round(pcm).astype("<i2").tobytes()


def test_long_question_fixture_passes_text_only_rules():
    assert len(LONG_QUESTION) >= 24
    assert is_likely_truncated_input_transcript(LONG_QUESTION_ANSWER, LONG_QUESTION, 40.0)
    assert not is_likely_truncated_input_transcript(
        LONG_QUESTION_ANSWER, LONG_QUESTION, 40.0, active_speech_sec=4.0, tail_active_speech_sec=0.0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rumble", [False, True])
async def test_turn_session_measured_activity_rejects_long_question_answer(rumble):
    session = make_session()
    pcm = _question_with_rumble() if rumble else _question_then_silence(4.0, 40.0)
    _feed_frames(session, pcm)
    active, tail = session._turn_speech_activity_secs()
    assert active < 15.0  # handling rumble is not counted as talking
    await _drive_marker_text(session, LONG_QUESTION_ANSWER)
    await session._on_input_transcription_completed({"transcript": LONG_QUESTION})
    await session._on_response_done({})
    await drain_finalize(session)
    assert [p["content"] for p in _replacements(session)] == [LONG_QUESTION]


def test_measure_speech_activity_ignores_handling_rumble():
    active, _ = measure_speech_activity_pcm16(_question_with_rumble(), SR)
    assert active < 15.0


def _faint_far_field_male_voice(total_sec=90.0, seed=11):
    """Real-T80-like far-field dictation: f0 120-240 Hz, >= 300 Hz harmonics
    at RMS ~4 Int16 over a ~1 Int16 floor, syllables through the whole take."""
    rng = np.random.default_rng(seed)
    n = int(total_sec * SR)
    x = np.zeros(n)
    t0 = 0.0
    while t0 < total_sec - 0.3:
        i0, i1 = int(t0 * SR), int((t0 + 0.25) * SR)
        tt = np.arange(i1 - i0) / SR
        f0 = 120.0 + 120.0 * rng.random()
        harmonics = sum(
            np.sin(2 * np.pi * f0 * h * tt + rng.uniform(0, 2 * np.pi)) / h
            for h in range(2, 16)
            if f0 * h >= 300.0
        )
        harmonics *= 4.0 / _rms(harmonics) / np.sqrt(3 / 8)  # in-band RMS 4 under the sin^2 envelope
        env = np.sin(np.pi * np.arange(i1 - i0) / (i1 - i0)) ** 2
        x[i0:i1] += (6.0 * np.sqrt(2) * np.sin(2 * np.pi * f0 * tt) + harmonics) * env
        t0 += 0.25 + 0.15 + (0.6 if rng.random() < 0.15 else 0.0)  # with phrase pauses
    x += rng.standard_normal(n) * 1.0
    return np.round(x).astype("<i2").tobytes()


@pytest.mark.asyncio
async def test_activity_gate_counts_faint_far_field_formants_but_not_rumble(monkeypatch):
    from micwave_core import audio_gain

    pcm = _faint_far_field_male_voice()
    session = make_session()
    _feed_frames(session, pcm)
    for active, tail in (measure_speech_activity_pcm16(pcm, SR), session._turn_speech_activity_secs()):
        assert active >= 15.0 and tail >= 4.0
    # The 100 Hz AGC floor (10 Int16) on the >= 300 Hz band misses this voice.
    monkeypatch.setattr(audio_gain, "ACTIVITY_GATE_MIN_RMS", audio_gain.SPEECH_GATE_MIN_RMS)
    assert measure_speech_activity_pcm16(pcm, SR)[0] < 15.0
    assert session._turn_speech_activity_secs()[0] < 15.0
    monkeypatch.undo()
    # 30 Hz handling rumble (amplitude 2000) stays below the 2.5 floor.
    rumble = _question_with_rumble()
    rumble_session = make_session()
    _feed_frames(rumble_session, rumble)
    assert measure_speech_activity_pcm16(rumble, SR)[0] < 6.0
    assert rumble_session._turn_speech_activity_secs()[0] < 6.0



# ── W0322 m7: output truncation verdict for iOS (W0273) auto-retry ─────────


def _verdict(chars, audio_sec, active=None, tail=None, **kwargs):
    result = assess_output_truncation(
        "字" * chars, audio_sec, active_speech_sec=active, tail_active_speech_sec=tail, **kwargs,
    )
    return result.suspected_truncated, result.truncation_reason


@pytest.mark.parametrize(
    "text,stripped,chars",
    [
        (INCOMPLETE_TURN_MARKER, "", 0),
        ("[片段2转录失败]", "", 0),
        (f"前文\n{INCOMPLETE_TURN_MARKER}", "前文", 2),
        ("甲乙。\n[片段12转录失败]\n丙。", "甲乙。\n\n丙。", 3),
    ],
)
def test_diagnostic_markers_never_count_as_transcript(text, stripped, chars):
    assert strip_diagnostic_markers(text) == stripped
    assert assess_output_truncation(text, 5.0).norm_char_count == chars


def test_marker_only_output_of_speech_is_empty_not_one_line():
    result = assess_output_truncation(INCOMPLETE_TURN_MARKER, 2.0, active_speech_sec=1.2)
    assert (result.norm_char_count, result.truncation_reason) == (0, "empty_output_for_active_speech")


def test_t80_before_fix_is_suspected_and_after_fix_is_not():
    assert _verdict(90, 91.6, 31.3, 11.5) == (True, "sparse_output_for_speech_coverage")
    assert _verdict(333, 91.6, 31.3, 11.5) == (False, None)
    assert _verdict(566, 91.6, 31.3, 11.5) == (False, None)


def test_rate_fields_and_frozen_assessment():
    result = assess_output_truncation("字" * 90, 91.6, active_speech_sec=31.3, tail_active_speech_sec=11.5)
    assert (result.norm_char_count, result.chars_per_audio_sec, result.chars_per_active_sec) == (90, 0.983, 2.875)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.suspected_truncated = False
    assert isinstance(result, OutputTruncationAssessment)
    assert assess_output_truncation("字", 0.0).chars_per_audio_sec == 0.0
    assert assess_output_truncation("字", 5.0, active_speech_sec=0.0).chars_per_active_sec is None


def test_sparse_coverage_needs_both_rates_below_threshold():
    # W0273-m14: 15 s speech + 120 s background silence + 11.5 s speech.
    assert _verdict(137, 146.5, 23.2, 10.1) == (False, None)  # 5.9 chars/active s
    assert _verdict(75, 146.5, 23.2, 10.1) == (True, "sparse_output_for_speech_coverage")
    # Just over / under the 3.4 chars per active second gate.
    assert _verdict(79, 146.5, 23.2, 10.1) == (False, None)  # 3.405
    assert _verdict(78, 146.5, 23.2, 10.1) == (True, "sparse_output_for_speech_coverage")
    # Tail speech 3.9 s misses the coverage gate (>= 4 s) but not the
    # active-speech gate (>= 3 s); below 3 s neither applies.
    assert _verdict(40, 91.6, 31.3, 3.9) == (True, "sparse_output_for_active_speech")
    assert _verdict(40, 91.6, 31.3, 2.9) == (False, None)
    # Sparse real-time rescue: 120 s, ~100 chars, 45 s active, 18 s tail.
    assert _verdict(100, 120.0, 45.0, 18.0) == (True, "sparse_output_for_speech_coverage")


def test_sparse_active_speech_for_medium_dictation():
    assert _verdict(15, 15.0, 10.0, 4.0) == (True, "sparse_output_for_active_speech")
    assert _verdict(22, 15.0, 10.0, 4.0) == (False, None)
    assert _verdict(15, 15.0, 10.0, 2.9) == (False, None)
    assert _verdict(15, 15.0, 7.9, 4.0) == (False, None)
    assert _verdict(15, 9.9, 9.0, 4.0) == (False, None)


@pytest.mark.parametrize("active", [0.0, 0.03, 0.59])
def test_silence_noise_or_rumble_with_no_text_is_not_retried(active):
    assert _verdict(0, 45.0, active, 0.0) == (False, None)


def test_short_spoken_command_with_no_text_is_suspected():
    assert _verdict(0, 1.2, 0.6, 0.0) == (True, "empty_output_for_active_speech")
    assert _verdict(0, 1.5, 1.2, 0.5) == (True, "empty_output_for_active_speech")


def test_text_only_fallback_flags_only_empty_long_audio():
    assert _verdict(0, 30.0) == (True, "empty_output_long_audio")
    assert _verdict(0, 29.9) == (False, None)
    assert _verdict(5, 120.0) == (False, None)


def test_failed_segment_and_upstream_incomplete_take_precedence():
    assert _verdict(800, 300.0, 200.0, 90.0, failed_segment_count=1, upstream_incomplete=True) == (
        True, "segment_failed",
    )
    assert _verdict(566, 91.6, 31.3, 11.5, upstream_incomplete=True) == (True, "upstream_incomplete")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_non_finite_or_negative_inputs_are_sanitised(bad):
    result = assess_output_truncation("字" * 10, bad)
    assert (result.chars_per_audio_sec, result.suspected_truncated) == (0.0, False)
    clean = assess_output_truncation("", 45.0, active_speech_sec=bad, tail_active_speech_sec=bad)
    if bad < 0:
        # Clamped to 0 s of speech: silence, not a text-only long empty audio.
        assert (clean.suspected_truncated, clean.chars_per_active_sec) == (False, None)
    else:
        # Non-finite activity is no evidence: the text-only rule applies.
        assert (clean.suspected_truncated, clean.truncation_reason) == (True, "empty_output_long_audio")
