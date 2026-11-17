"""Seeded randomized lifecycle check for soft segmentation (task 1049, round 4).

Random speech/pause audio is cut into random-sized upload chunks and delivered
by a receive loop that handles messages strictly in order (as `run()` does).
Provider terminals arrive on their own schedule, from prompt to very slow, and
the user stops at a random point. A slow prefix lets the rollover tail grow past
soft + grace; that tail is committed whole (known limitation, no admission cap:
receive backpressure would abort the desktop recording). Whatever the
interleaving:

* the tail is never refused below the hard cap (no backpressure at soft + grace);
* every byte reaches exactly one provider buffer, in order (a short final tail
  gets one explicit all-zero pad, nothing else is added or dropped);
* a provider buffer is at most the larger of the longest queued tail and
  soft + grace, plus two upload chunks (live audio after a promotion is cut at
  the forced point as usual);
* the delivered text is the segment texts in order, with exactly one idle, and
  the receive loop is never wedged.
"""
import asyncio
import random
import struct

import pytest

from .test_soft_segment import (
    FRAME_BYTES,
    GRACE_FRAMES,
    MIN_COMMIT_BYTES,
    SOFT_FRAMES,
    install_providers,
    land_rollover,
    complete,
    soft_session,
)
from .test_transcription_server import _assert_s1, _client_text_view, _idles

SEEDS = range(80)
MAX_CHUNK = 4 * FRAME_BYTES + 6        # deliberately not 20 ms aligned
FORCE_BYTES = (SOFT_FRAMES + GRACE_FRAMES) * FRAME_BYTES
TERMINAL_RATES = (0.5, 0.1, 0.02)      # per scheduler turn: prompt ... very slow


def random_audio(rnd: random.Random) -> bytes:
    """Speech bursts and nonzero pauses of random length (never all-zero runs)."""
    parts = []
    total_frames = rnd.randint(1, 420)
    frames = 0
    while frames < total_frames:
        if rnd.random() < 0.55:
            run, amp = rnd.randint(3, 90), rnd.randint(400, 3000)
        else:
            run, amp = rnd.randint(1, 30), rnd.randint(1, 100)
        parts.append(struct.pack("<h", amp) * (480 * run))
        frames += run
    return b"".join(parts)


def random_chunks(rnd: random.Random, audio: bytes) -> list:
    chunks, pos = [], 0
    while pos < len(audio):
        size = min(len(audio) - pos, rnd.randrange(2, MAX_CHUNK, 2))
        chunks.append(audio[pos:pos + size])
        pos += size
    return chunks


async def run_seed(seed: int) -> dict:
    """One random session; returns what the invariants and coverage need."""
    rnd = random.Random(seed)
    s = soft_session()
    providers, committed = install_providers(s, 60)
    chunks = random_chunks(rnd, random_audio(rnd))
    chunks = chunks[:rnd.randint(1, len(chunks))]          # the user stops here
    audio = b"".join(chunks)
    terminal_rate = rnd.choice(TERMINAL_RATES)
    stats = {"refused": 0, "max_tail": 0}
    admit = s._append_rollover_tail_locked

    def watched_admission(chunk):
        admitted = admit(chunk)
        stats["refused"] += not admitted
        stats["max_tail"] = max(stats["max_tail"], s._rollover_tail_bytes)
        return admitted

    s._append_rollover_tail_locked = watched_admission
    done = [0]

    async def complete_next() -> bool:
        i = done[0]
        if i >= len(providers) or not committed[i].is_set():
            return False
        was_rollover = s._segment_rollover_pending
        await complete(providers[i], f"seg-{seed}-{i}", f"段{i}")
        if was_rollover:
            await land_rollover(s)
        done[0] += 1
        return True

    async def receive_loop():
        for chunk in chunks:
            await s._handle_audio_bytes(chunk)
        await s._handle_stop_recording({"turn_id": 900})

    receive = asyncio.create_task(receive_loop())
    for _ in range(40_000):
        if receive.done():
            break
        if rnd.random() < terminal_rate:
            await complete_next()
        await asyncio.sleep(0)
    await asyncio.wait_for(receive, 2)                     # a wedged loop raises here

    return {
        "seed": seed,
        "state": s,
        "audio": audio,
        "providers": providers,
        "committed": committed,
        "refused": stats["refused"],
        "max_tail": stats["max_tail"],
    }


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.asyncio
async def test_random_interleavings_conserve_audio_text_and_bounds(seed):
    run = await run_seed(seed)
    s, audio, providers, committed = (
        run["state"], run["audio"], run["providers"], run["committed"],
    )

    assert run["refused"] == 0                             # no admission cap below the hard cap
    sent = [b"".join(p.sent_audio) for p in providers if p.sent_audio]
    joined = b"".join(sent)
    assert joined[:len(audio)] == audio, "audio lost, duplicated or reordered"
    if len(joined) > len(audio):
        pad = joined[len(audio):]
        assert pad == bytes(len(pad)), "only explicit digital silence may be added"
        assert sent[-1] and len(sent[-1]) == MIN_COMMIT_BYTES
    buffer_limit = max(run["max_tail"], FORCE_BYTES) + 2 * MAX_CHUNK
    for buf in sent:
        assert len(buf) <= buffer_limit
    n_committed = sum(1 for e in committed if e.is_set())
    assert _client_text_view(s) == "".join(f"段{i}" for i in range(n_committed))
    if len(audio) < MIN_COMMIT_BYTES:
        # A whole turn under the provider minimum with no earlier segment keeps
        # the old early finish: sent live, never committed, no text.
        assert n_committed == 0
    else:
        assert n_committed == len(sent)
    assert [p for p in s._websocket.sent if p.get("type") == "error"] == []
    assert len(_idles(s)) == 1
    _assert_s1(s)


@pytest.mark.asyncio
async def test_random_seeds_actually_queue_long_tails_and_pad():
    """Guard against a vacuous suite: the interesting paths must really run."""
    long_tails, padded = 0, 0
    for seed in SEEDS:
        run = await run_seed(seed)
        long_tails += run["max_tail"] > FORCE_BYTES
        sent = [b"".join(p.sent_audio) for p in run["providers"] if p.sent_audio]
        padded += len(b"".join(sent)) > len(run["audio"])
    assert long_tails >= 10, f"only {long_tails} of {len(SEEDS)} seeds queued a tail past soft + grace"
    assert padded >= 1, "no seed ended in a padded short tail"
