"""Speech front-end for PCM16 sent to the realtime transcription provider.

Why this exists (task 1193 / W0322): the iOS recorder captures with
``AVAudioSession`` mode ``.measurement``, which turns off the hardware AGC and
high-pass filter. A far-field phone recording (T80, 91.6 s) carried 90 % of its
energy as 10-80 Hz handling rumble while the speech band sat at -60..-68 dBFS.
Whole-file peak normalisation was pinned at 5.4x by one stop-button transient,
so the provider's input transcript stalled after two sentences.

The fix applied here, measured on T80-T84 before it was written down:

* 2nd-order 100 Hz Butterworth high-pass removes the rumble;
* a speech-presence gate refuses to amplify silence or stationary noise
  (amplified hiss makes the transcriber hallucinate);
* a local speech-envelope AGC lifts quiet stretches toward ``target_rms`` with
  the gain capped so the noise floor never exceeds ``max_noise_out``; pauses
  longer than a short hangover fall back to a low gain so they stay quiet
  (a loud noise floor in pauses sent gpt-4o-transcribe into repeat loops);
* a soft-knee ``tanh`` limiter keeps transients from hard clipping.

``split_pcm_on_pauses`` cuts only very long recordings, and only at the centre
of a real pause, so recordings up to ``soft_max_sec`` keep one context.

Authoritative implementation shared across all voice consumers via
``micwave_core.audio_gain``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.signal

FRAME_SEC = 0.03
SPEECH_GATE_SNR = 2.2
SPEECH_GATE_MIN_RMS = 10.0
# Absolute floor of the activity gate (speech_activity_secs), which runs on
# >= 300 Hz band RMS: the 4th-order high-pass strips the male fundamental, so
# far-field T80 has a noise floor of ~1.07 and formant syllables at 2.5-10
# Int16. The 100 Hz AGC gate keeps SPEECH_GATE_MIN_RMS.
ACTIVITY_GATE_MIN_RMS = 2.5
NO_SPEECH_P95_RMS = 10.0
NO_SPEECH_SNR_P99 = 3.2
NO_SPEECH_MAX_P95_RMS = 1200.0
PAUSE_HANGOVER_FRAMES = 6  # 180 ms: keep trailing consonants at full gain
PAUSE_PREROLL_FRAMES = 10  # 300 ms: offline pass raises gain before an onset
PAUSE_NOISE_OUT = 150.0
PAUSE_MAX_GAIN = 6.0
ENVELOPE_HALF_WINDOW_FRAMES = 25  # ~1.5 s local speech-level window
GAIN_ATTACK_ALPHA = 0.6  # gain falls within 1-2 frames when input gets louder
GAIN_RELEASE_ALPHA = 1.0 - math.exp(-1.0 / 8.0)  # rises over ~8 frames (240 ms)
LIMITER_KNEE = 22000.0
LIMITER_CEILING = 31800.0


@dataclass
class SpeechGainStats:
    applied: bool
    reason: str
    noise_floor_rms: float = 0.0
    p95_rms: float = 0.0
    snr_p95: float = 0.0
    min_gain: float = 1.0
    median_gain: float = 1.0
    max_gain: float = 1.0
    output_rms: float = 0.0
    # Seconds of frames above the speech gate, in the whole recording and in
    # its second half; 0.0 when the input is empty, silent or speechless. The
    # truncated-transcript guard needs this acoustic evidence (W0322 R2): a
    # 30 s recording holding one 3 s question is not 30 s of dictation.
    active_speech_sec: float = 0.0
    tail_active_speech_sec: float = 0.0


def speech_activity_secs(frame_rms: np.ndarray, frame_sec: float) -> tuple[float, float]:
    """``(active_sec, tail_active_sec)`` of a frame-RMS series.

    A frame is speech when it reaches ``SPEECH_GATE_SNR`` times the p15 noise
    floor and ``ACTIVITY_GATE_MIN_RMS``: the AGC's SNR rule with a lower
    absolute floor, because the >= 300 Hz band drops the voice fundamental
    and far-field syllables sit at 2.5-10 Int16 there. The tail is the second
    half of the series by time. Feed it speech-band RMS (rumble removed),
    otherwise handling noise counts as talking.
    """
    if frame_rms.size == 0:
        return 0.0, 0.0
    active = _speech_active_mask(frame_rms, min_rms=ACTIVITY_GATE_MIN_RMS)
    tail = active[active.size // 2:]
    return float(np.sum(active)) * frame_sec, float(np.sum(tail)) * frame_sec


def _speech_active_mask(frame_rms: np.ndarray, min_rms: float = SPEECH_GATE_MIN_RMS) -> np.ndarray:
    noise_floor = max(float(np.percentile(frame_rms, 15)), 1.0)
    return frame_rms >= max(noise_floor * SPEECH_GATE_SNR, min_rms)


@lru_cache(maxsize=16)
def _highpass_sos(sample_rate: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    return scipy.signal.butter(
        order, cutoff_hz / (sample_rate * 0.5), btype="highpass", output="sos"
    )


def _even_samples(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype="<i2", count=len(pcm_bytes) // 2)


def _frame_rms(x: np.ndarray, frame_len: int) -> np.ndarray:
    n_frames = x.size // frame_len
    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _soft_limit(y: np.ndarray) -> np.ndarray:
    magnitude = np.abs(y)
    over = magnitude > LIMITER_KNEE
    if np.any(over):
        headroom = LIMITER_CEILING - LIMITER_KNEE
        y = y.copy()
        y[over] = np.sign(y[over]) * (
            LIMITER_KNEE + headroom * np.tanh((magnitude[over] - LIMITER_KNEE) / headroom)
        )
    return y


def _highpassed_frame_rms(samples: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 2):
    """``(filtered_signal, frame_rms, frame_len)`` after a high-pass."""
    frame_len = max(1, int(sample_rate * FRAME_SEC))
    x = scipy.signal.sosfilt(
        _highpass_sos(sample_rate, float(cutoff_hz), order), samples.astype(np.float64)
    )
    return x, _frame_rms(x, frame_len), frame_len


# Speech activity is measured on the >= 300 Hz band (4th order: 30 Hz handling
# rumble -80 dB), the band the streaming lane's no-speech gate already uses;
# the AGC's 100 Hz 2nd-order filter leaves loud rumble bursts countable.
ACTIVITY_BAND_HZ = 300.0
ACTIVITY_BAND_ORDER = 4


def measure_speech_activity_pcm16(pcm_bytes: bytes, sample_rate: int = 24000) -> tuple[float, float]:
    """``(active_speech_sec, tail_active_speech_sec)`` of raw PCM16 on the
    speech band; ``preprocess_speech_pcm16`` reports the same numbers."""
    samples = _even_samples(pcm_bytes)
    if not np.any(samples):
        return 0.0, 0.0
    _, frame_rms, _ = _highpassed_frame_rms(
        samples, sample_rate, ACTIVITY_BAND_HZ, ACTIVITY_BAND_ORDER
    )
    return speech_activity_secs(frame_rms, FRAME_SEC)


def _to_pcm16(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(y), -32768, 32767).astype("<i2")


def _local_speech_envelope(frame_rms: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Geometric mean of active-frame RMS within +/- ~0.75 s of each frame.

    Frames whose window holds fewer than two active frames take the value of the
    nearest frame that has one, so a pause keeps the level of the speaker
    around it instead of falling to the noise floor.
    """
    n = frame_rms.size
    log_level = np.where(active, np.log(np.maximum(frame_rms, 1.0)), 0.0)
    csum = np.concatenate(([0.0], np.cumsum(log_level)))
    ccount = np.concatenate(([0], np.cumsum(active.astype(np.int64))))
    idx = np.arange(n)
    lo = np.maximum(0, idx - ENVELOPE_HALF_WINDOW_FRAMES)
    hi = np.minimum(n, idx + ENVELOPE_HALF_WINDOW_FRAMES + 1)
    count = ccount[hi] - ccount[lo]
    valid = count >= 2
    if not np.any(valid):
        return np.full(n, float(np.max(frame_rms[active])) if np.any(active) else 1.0)
    env_log = np.zeros(n)
    env_log[valid] = (csum[hi] - csum[lo])[valid] / count[valid]
    if not np.all(valid):
        env_log = np.interp(idx, idx[valid], env_log[valid])
    return np.exp(env_log)


def _pause_mask(active: np.ndarray) -> np.ndarray:
    """Frames inside a pause, past the hangover and before the onset pre-roll."""
    n = active.size
    since_active = np.empty(n, dtype=np.int64)
    run = PAUSE_HANGOVER_FRAMES + 1  # leading silence counts as a pause
    for i in range(n):
        run = 0 if active[i] else run + 1
        since_active[i] = run
    until_active = np.empty(n, dtype=np.int64)
    run = PAUSE_PREROLL_FRAMES + 1
    for i in range(n - 1, -1, -1):
        run = 0 if active[i] else run + 1
        until_active[i] = run
    return (since_active > PAUSE_HANGOVER_FRAMES) & (until_active > PAUSE_PREROLL_FRAMES)


def _smooth_gain(target: np.ndarray, initial: float) -> np.ndarray:
    out = np.empty_like(target)
    g = initial
    for i, t in enumerate(target):
        alpha = GAIN_ATTACK_ALPHA if t < g else GAIN_RELEASE_ALPHA
        g += alpha * (t - g)
        out[i] = g
    return out


def preprocess_speech_pcm16(
    pcm_bytes: bytes,
    sample_rate: int = 24000,
    *,
    target_rms: float = 3200.0,
    max_gain: float = 50.0,
    max_noise_out: float = 420.0,
    hpf_cutoff_hz: float = 100.0,
) -> tuple[bytes, SpeechGainStats]:
    """High-pass + gated dynamic AGC + soft limiter for a whole recording.

    Returns the input object unchanged (``applied=False``) for empty, all-zero,
    too-short, and no-speech input. Output length always equals input length;
    an odd trailing byte is carried through untouched.
    """
    if len(pcm_bytes) < 2:
        return pcm_bytes, SpeechGainStats(applied=False, reason="empty")
    samples = _even_samples(pcm_bytes)
    if not np.any(samples):
        return pcm_bytes, SpeechGainStats(applied=False, reason="digital_silence")

    frame_len = max(1, int(sample_rate * FRAME_SEC))
    if samples.size // frame_len < 3:
        return pcm_bytes, SpeechGainStats(applied=False, reason="too_short")

    x, frame_rms, frame_len = _highpassed_frame_rms(samples, sample_rate, hpf_cutoff_hz)
    noise_floor = max(float(np.percentile(frame_rms, 15)), 1.0)
    p95 = float(np.percentile(frame_rms, 95))
    p99 = float(np.percentile(frame_rms, 99))
    snr_p95 = p95 / noise_floor
    snr_p99 = p99 / noise_floor
    measured = dict(noise_floor_rms=noise_floor, p95_rms=p95, snr_p95=snr_p95)

    if p95 < NO_SPEECH_P95_RMS or (
        snr_p95 < SPEECH_GATE_SNR
        and snr_p99 < NO_SPEECH_SNR_P99
        and p95 < NO_SPEECH_MAX_P95_RMS
    ):
        return pcm_bytes, SpeechGainStats(applied=False, reason="no_speech", **measured)

    active = _speech_active_mask(frame_rms)
    active_sec, tail_active_sec = measure_speech_activity_pcm16(pcm_bytes, sample_rate)
    measured.update(active_speech_sec=active_sec, tail_active_speech_sec=tail_active_sec)
    healthy = (
        np.any(active)
        and float(np.percentile(frame_rms[active], 25)) >= target_rms * 0.65
        and p95 >= target_rms * 0.85
    )
    if healthy:
        frame_gain = np.ones(frame_rms.size)
        reason = "hpf_only"
    else:
        effective_max_gain = min(max_gain, max(1.0, max_noise_out / noise_floor))
        speech_env = _local_speech_envelope(frame_rms, active)
        g_speech = np.clip(target_rms / np.maximum(speech_env, 1.0), 1.0, effective_max_gain)
        pause_gain = np.minimum(
            g_speech, max(1.0, min(PAUSE_MAX_GAIN, PAUSE_NOISE_OUT / noise_floor))
        )
        target_gain = np.where(_pause_mask(active), pause_gain, g_speech)
        frame_gain = _smooth_gain(target_gain, float(target_gain[0]))
        reason = "dynamic_agc"

    centers = (np.arange(frame_gain.size) + 0.5) * frame_len
    sample_gain = np.interp(np.arange(x.size, dtype=np.float64), centers, frame_gain)
    out = _to_pcm16(_soft_limit(x * sample_gain))
    output_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    out_bytes = out.tobytes() + pcm_bytes[out.size * 2:]
    return out_bytes, SpeechGainStats(
        applied=True,
        reason=reason,
        min_gain=float(np.min(frame_gain)),
        median_gain=float(np.median(frame_gain)),
        max_gain=float(np.max(frame_gain)),
        output_rms=output_rms,
        **measured,
    )


class StreamingSpeechAGC:
    """Causal per-chunk version of ``preprocess_speech_pcm16`` for the
    streaming lane. High-pass state, frame boundaries, the noise-floor and
    speech-envelope trackers carry across chunks; ``reset()`` starts a turn.

    Byte-stream contract: the input is little-endian PCM16. ``process_chunk``
    returns exactly the whole samples it can form; an odd trailing byte is held
    and joined to the next chunk, so the output of every call is an even number
    of bytes. ``flush()`` hands back a held half sample at the end of a turn;
    callers must not forward it to the provider, because
    ``input_audio_buffer.append`` takes whole Int16 samples and an orphan byte
    would shift every later sample by one byte. ``reset()`` drops it too.

    Stationary-noise gate (W0322 R2): the first ``BOOT_FRAMES`` frames are
    passed at unity gain; if they are steady the noise floor is set to their
    median. After that, whenever the last ``MOD_WINDOW_FRAMES`` frames have a
    p90/p15 ratio under ``MOD_STATIONARY_RATIO`` (fan, hiss, room tone at any
    level have about 1.0-1.4, speech is above 2.2), the frame is treated as
    noise: the floor moves toward the window p20 and the target gain is 1.0.
    Otherwise gain stays at 1.0 until a frame rises ``SPEECH_GATE_SNR`` times
    over the tracked floor.
    """

    INITIAL_NOISE_FLOOR = 35.0
    FLOOR_DOWN_ALPHA = 0.3
    FLOOR_UP_RATE = 1.001  # per frame: doubles in ~23 s
    ENV_ATTACK_ALPHA = 0.5
    ENV_RELEASE_ALPHA = 0.05
    BOOT_FRAMES = 8  # 240 ms
    BOOT_STEADY_RATIO = 1.8
    MOD_WINDOW_FRAMES = 50  # 1.5 s
    MOD_MIN_FRAMES = 10
    MOD_STATIONARY_RATIO = 1.85
    MOD_FLOOR_ALPHA = 0.3

    def __init__(
        self,
        sample_rate: int = 24000,
        *,
        target_rms: float = 3000.0,
        max_gain: float = 40.0,
        max_noise_out: float = 420.0,
        hpf_cutoff_hz: float = 100.0,
    ):
        self._sample_rate = int(sample_rate)
        self._frame_len = max(1, int(self._sample_rate * FRAME_SEC))
        self._target_rms = float(target_rms)
        self._max_gain = float(max_gain)
        self._max_noise_out = float(max_noise_out)
        self._sos = _highpass_sos(self._sample_rate, float(hpf_cutoff_hz))
        self.reset()

    def reset(self) -> None:
        self._zi = np.zeros((self._sos.shape[0], 2))
        self._byte_carry = b""
        self._frame_residual = np.zeros(0)
        self._noise_floor = self.INITIAL_NOISE_FLOOR
        self._log_env: float | None = None
        self._inactive_run = PAUSE_HANGOVER_FRAMES + 1
        self._gain = 1.0
        self._frames_seen = 0
        self._recent_rms: deque[float] = deque(maxlen=self.MOD_WINDOW_FRAMES)

    @property
    def current_gain(self) -> float:
        return self._gain

    def flush(self) -> bytes:
        """Return and clear a held odd byte (``b""`` if none). See the class
        docstring: this half sample is never meant for the provider."""
        held, self._byte_carry = self._byte_carry, b""
        return held

    def _frame_target_gain(self, rms: float) -> float:
        self._frames_seen += 1
        self._recent_rms.append(rms)
        if self._frames_seen <= self.BOOT_FRAMES:
            self._speech_target_gain(rms)
            if self._frames_seen == self.BOOT_FRAMES:
                boot = list(self._recent_rms)
                if max(boot) / max(min(boot), 1.0) < self.BOOT_STEADY_RATIO:
                    self._noise_floor = max(float(np.median(boot)), 1.0)
                    self._log_env = None
                    self._inactive_run = PAUSE_HANGOVER_FRAMES + 1
            return 1.0
        if len(self._recent_rms) >= self.MOD_MIN_FRAMES:
            window = np.fromiter(self._recent_rms, dtype=np.float64)
            p15, p20, p90 = np.percentile(window, (15, 20, 90))
            if p90 / max(p15, 1.0) < self.MOD_STATIONARY_RATIO:
                self._noise_floor = max(
                    (1.0 - self.MOD_FLOOR_ALPHA) * self._noise_floor + self.MOD_FLOOR_ALPHA * p20,
                    1.0,
                )
                self._inactive_run += 1
                return 1.0
        return self._speech_target_gain(rms)

    def _speech_target_gain(self, rms: float) -> float:
        floor = self._noise_floor
        active = rms >= max(floor * SPEECH_GATE_SNR, SPEECH_GATE_MIN_RMS)
        if rms < floor:
            floor += self.FLOOR_DOWN_ALPHA * (rms - floor)
        elif not active:
            floor = min(rms, floor * self.FLOOR_UP_RATE)
        self._noise_floor = max(floor, 1.0)

        if active:
            level = math.log(max(rms, 1.0))
            if self._log_env is None:
                self._log_env = level
            else:
                alpha = self.ENV_ATTACK_ALPHA if level > self._log_env else self.ENV_RELEASE_ALPHA
                self._log_env += alpha * (level - self._log_env)
            self._inactive_run = 0
        else:
            self._inactive_run += 1
        if self._log_env is None:
            return 1.0
        effective_max = min(self._max_gain, max(1.0, self._max_noise_out / self._noise_floor))
        g_speech = min(max(self._target_rms / math.exp(self._log_env), 1.0), effective_max)
        if self._inactive_run > PAUSE_HANGOVER_FRAMES:
            return min(g_speech, max(1.0, min(PAUSE_MAX_GAIN, PAUSE_NOISE_OUT / self._noise_floor)))
        return g_speech

    def process_chunk(self, pcm_bytes: bytes) -> bytes:
        data = self._byte_carry + pcm_bytes if self._byte_carry else pcm_bytes
        even = len(data) - (len(data) % 2)
        self._byte_carry = data[even:]
        if even == 0:
            return b""
        samples = _even_samples(data)
        if not np.any(samples) and not np.any(self._zi):
            # Filter at rest and zero input: output is zero; only advance the
            # frame trackers so timing and the floor stay consistent.
            y = np.zeros(samples.size)
            passthrough = data[:even]
        else:
            y, self._zi = scipy.signal.sosfilt(self._sos, samples.astype(np.float64), zi=self._zi)
            passthrough = None

        gains = np.empty(samples.size)
        pos = 0
        residual = self._frame_residual
        while pos < samples.size:
            need = self._frame_len - residual.size
            take = min(need, samples.size - pos)
            segment = y[pos:pos + take]
            if take == need:
                frame = np.concatenate((residual, segment)) if residual.size else segment
                rms = float(np.sqrt(np.mean(frame * frame)))
                target = self._frame_target_gain(rms)
                alpha = GAIN_ATTACK_ALPHA if target < self._gain else GAIN_RELEASE_ALPHA
                new_gain = self._gain + alpha * (target - self._gain)
                gains[pos:pos + take] = np.linspace(self._gain, new_gain, take, endpoint=False)
                self._gain = new_gain
                residual = np.zeros(0)
            else:
                gains[pos:pos + take] = self._gain
                residual = np.concatenate((residual, segment))
            pos += take
        self._frame_residual = residual

        if passthrough is not None:
            return passthrough
        return _to_pcm16(_soft_limit(y * gains)).tobytes()


def split_pcm_on_pauses(
    pcm_bytes: bytes,
    sample_rate: int = 24000,
    *,
    soft_max_sec: float = 180.0,
    hard_max_sec: float = 240.0,
    min_pause_sec: float = 0.45,
) -> list[bytes]:
    """Split a long recording at the centres of real pauses.

    Recordings up to ``soft_max_sec`` stay whole so the transcriber keeps the
    opening context. Longer ones are cut, per segment, at the quietest pause of
    at least ``min_pause_sec`` whose centre lies in
    ``[start + 0.55 * soft_max_sec, start + hard_max_sec]``. Without such a
    pause a segment is cut at the quietest frame only once the rest exceeds
    ``hard_max_sec``. All-zero input and input without speech stay whole.

    Out-of-range parameters are clamped rather than trusted (W0322 R2: a zero
    soft limit used to loop forever): ``min_pause_sec >= 0.06``,
    ``soft_max_sec >= max(1, 2 * min_pause_sec)``, ``hard_max_sec >=
    soft_max_sec`` (NaN takes the floor, or ``1.25 * soft`` for the hard
    limit; an infinite soft limit keeps the recording whole, an infinite hard
    limit cuts only at pauses), and every cut advances by at least half a soft
    segment.
    """
    if not sample_rate or sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    if not math.isfinite(min_pause_sec) or min_pause_sec < 0.06:
        min_pause_sec = 0.06
    if soft_max_sec == math.inf:
        return [pcm_bytes]  # no soft limit: never split
    soft_floor = max(1.0, min_pause_sec * 2.0)
    if math.isnan(soft_max_sec) or soft_max_sec < soft_floor:
        soft_max_sec = soft_floor
    if math.isnan(hard_max_sec) or hard_max_sec < soft_max_sec:
        hard_max_sec = soft_max_sec * 1.25

    samples = _even_samples(pcm_bytes)
    total_sec = samples.size / float(sample_rate)
    if total_sec <= soft_max_sec or not np.any(samples):
        return [pcm_bytes]

    frame_len = max(1, int(sample_rate * FRAME_SEC))
    x = scipy.signal.sosfilt(_highpass_sos(sample_rate, 100.0), samples.astype(np.float64))
    frame_rms = _frame_rms(x, frame_len)
    n_frames = frame_rms.size
    noise_floor = max(float(np.percentile(frame_rms, 15)), 1.0)
    if float(np.percentile(frame_rms, 95)) / noise_floor < SPEECH_GATE_SNR:
        return [pcm_bytes]

    quiet = frame_rms < max(noise_floor * SPEECH_GATE_SNR, SPEECH_GATE_MIN_RMS)
    min_pause_frames = max(1, int(math.ceil(min_pause_sec / FRAME_SEC)))
    valleys: list[tuple[int, int]] = []  # [start, end) frame ranges
    start = None
    for i in range(n_frames + 1):
        is_quiet = i < n_frames and bool(quiet[i])
        if is_quiet and start is None:
            start = i
        elif not is_quiet and start is not None:
            if i - start >= min_pause_frames:
                valleys.append((start, i))
            start = None

    frames_per_sec = 1.0 / FRAME_SEC
    soft_frames = soft_max_sec * frames_per_sec
    hard_frames = hard_max_sec * frames_per_sec
    cuts: list[int] = []  # frame index of each cut
    cur = 0
    while n_frames - cur > soft_frames:
        lo = cur + soft_frames * 0.55
        hi = cur + hard_frames
        candidates = [
            (float(np.mean(frame_rms[s:e])), -(e - s), (s + e) // 2)
            for s, e in valleys
            if lo <= (s + e) // 2 <= hi and (s + e) // 2 < n_frames
        ]
        if candidates:
            cut = min(candidates)[2]
        elif n_frames - cur > hard_frames:
            f_lo = int(lo)
            f_hi = min(int(hi), n_frames - 1)
            smoothed = np.convolve(frame_rms, np.ones(5) / 5.0, mode="same")
            cut = f_lo + int(np.argmin(smoothed[f_lo:f_hi + 1]))
        else:
            break
        # Progress invariant: every cut moves forward by half a soft segment,
        # so the loop ends in O(n_frames) whatever the parameters.
        cut = max(cut, cur + max(1, int(soft_frames // 2)))
        if cut >= n_frames:
            break
        cuts.append(cut)
        cur = cut
    if not cuts:
        return [pcm_bytes]

    bounds = [0] + [c * frame_len * 2 for c in cuts] + [len(pcm_bytes)]
    return [pcm_bytes[a:b] for a, b in zip(bounds, bounds[1:])]
