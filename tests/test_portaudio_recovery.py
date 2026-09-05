import asyncio
import concurrent.futures
import threading
from types import SimpleNamespace

import pytest

import ime_menubar as ime
from ime_menubar import BrainwaveIMEApp, BrainwaveIMECore, Config, IMEState


class FakeStream:
    def __init__(self, events=None):
        self.started = False
        self.stopped = False
        self.closed = False
        self.start_threads = []
        self.events = events

    def start_stream(self):
        self.start_threads.append(threading.current_thread().name)
        self.started = True

    def stop_stream(self):
        if self.events is not None:
            self.events.append("stream.stop")
        self.stopped = True

    def close(self):
        if self.events is not None:
            self.events.append("stream.close")
        self.closed = True


class FakePyAudio:
    def __init__(self, open_results, *, events=None, index=0):
        self.open_results = list(open_results)
        self.open_threads = []
        self.terminate_threads = []
        self.callbacks = []
        self.events = events
        self.index = index
        self.on_terminate = None

    def open(self, **kwargs):
        self.open_threads.append(threading.current_thread().name)
        self.callbacks.append(kwargs["stream_callback"])
        result = self.open_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def terminate(self):
        self.terminate_threads.append(threading.current_thread().name)
        if self.events is not None:
            self.events.append(f"instance-{self.index}.terminate")
        if self.on_terminate is not None:
            self.on_terminate()


class FakePyAudioFactory:
    def __init__(self, instance_plans, *, events=None):
        self.instance_plans = list(instance_plans)
        self.instances = []
        self.creation_threads = []
        self.events = events

    def __call__(self):
        self.creation_threads.append(threading.current_thread().name)
        plan = self.instance_plans.pop(0)
        if isinstance(plan, BaseException):
            raise plan
        index = len(self.instances)
        if self.events is not None:
            self.events.append(f"instance-{index}.create")
        instance = FakePyAudio(plan, events=self.events, index=index)
        self.instances.append(instance)
        return instance


def make_core(monkeypatch, instance_plans, *, events=None):
    monkeypatch.setenv("BRAINWAVE_RECENT_AUDIO_CACHE_ENABLED", "0")
    factory = FakePyAudioFactory(instance_plans, events=events)
    monkeypatch.setattr(ime.pyaudio, "PyAudio", factory)
    core = BrainwaveIMECore(Config())
    core.loop = asyncio.get_running_loop()
    core.state = IMEState.IDLE
    core._ensure_audio_pipeline = lambda: None
    core._should_refresh_ws_before_turn = lambda: False
    core._play_sound = lambda _name: None
    core._archive_failed_turn_audio = lambda _reason: None
    core._archive_recent_turn_audio = lambda _reason: None
    core._stop_session_connect_deadline_sec = 0
    core._session_started = False
    core._start_session_task = lambda: setattr(core, "_session_started", True)
    return core, factory


def cleanup_core(core):
    core.loop = None
    core.state = IMEState.IDLE
    core.cleanup()


@pytest.mark.asyncio
async def test_audio_open_first_attempt_keeps_existing_portaudio(monkeypatch):
    stream = FakeStream()
    core, factory = make_core(monkeypatch, [[stream]])
    initial_instance = core.pyaudio_instance
    try:
        await core._start_recording()

        assert core.state is IMEState.RECORDING
        assert core.audio_stream is stream
        assert stream.started is True
        assert core._session_started is True
        assert core.pyaudio_instance is initial_instance
        assert len(factory.instances) == 1
        assert initial_instance.terminate_threads == []
        assert all(name.startswith("pyaudio") for name in factory.creation_threads)
        assert all(name.startswith("pyaudio") for name in initial_instance.open_threads)
        assert all(name.startswith("pyaudio") for name in stream.start_threads)
    finally:
        cleanup_core(core)


@pytest.mark.asyncio
async def test_audio_open_rebuilds_once_then_starts_recording(monkeypatch, capsys):
    stream = FakeStream()
    core, factory = make_core(
        monkeypatch,
        [[RuntimeError("stale device table")], [stream]],
    )
    old_instance = core.pyaudio_instance
    cleaned_up = False
    try:
        await core._start_recording()

        output = capsys.readouterr().out
        assert "Audio open failed (attempt 1/2)" in output
        assert "audio stream opened on attempt 2/2" in output
        assert len(factory.instances) == 2
        assert old_instance.terminate_threads == [old_instance.open_threads[0]]
        assert core.pyaudio_instance is factory.instances[1]
        assert core.audio_stream is stream
        assert stream.started is True
        assert core.state is IMEState.RECORDING
        assert core._session_started is True
        replacement = factory.instances[1]
        assert all(name.startswith("pyaudio") for name in replacement.open_threads)
        assert all(name.startswith("pyaudio") for name in stream.start_threads)
        cleanup_core(core)
        cleaned_up = True
        assert all(
            name.startswith("pyaudio") for name in replacement.terminate_threads
        )
    finally:
        if not cleaned_up:
            cleanup_core(core)


@pytest.mark.asyncio
async def test_audio_open_two_failures_returns_cleanly_to_idle(
    monkeypatch,
    capsys,
):
    core, factory = make_core(
        monkeypatch,
        [
            [RuntimeError("stale device table")],
            [RuntimeError("replacement device unavailable")],
        ],
    )
    try:
        await core._start_recording()

        output = capsys.readouterr().out
        assert "Audio open failed (attempt 1/2)" in output
        assert "Audio open failed after PortAudio rebuild (attempt 2/2)" in output
        assert "[IME] Audio error: replacement device unavailable" in output
        assert len(factory.instances) == 2
        assert factory.instances[0].terminate_threads
        assert core.state is IMEState.IDLE
        assert core.audio_stream is None
        assert core._active_recording_token is None
        assert core._audio_consumer_paused is True
        assert core._audio_drained.is_set()
        assert core._session_started is False
    finally:
        cleanup_core(core)


@pytest.mark.asyncio
async def test_stop_tail_window_callback_continues_until_stream_close(monkeypatch):
    stream = FakeStream()
    core, factory = make_core(monkeypatch, [[stream]])
    try:
        await core._start_recording()
        callback = factory.instances[0].callbacks[0]

        stop_task = asyncio.create_task(core._stop_recording())
        await asyncio.sleep(0.02)
        callback_result = callback(b"\x00\x00" * 160, 160, None, 0)

        assert core.state is IMEState.PROCESSING
        assert stream.closed is False
        assert callback_result[1] == ime.pyaudio.paContinue
        await stop_task
    finally:
        cleanup_core(core)


@pytest.mark.asyncio
async def test_recovery_stops_when_token_is_invalidated_during_rebuild(monkeypatch):
    unused_stream = FakeStream()
    core, factory = make_core(
        monkeypatch,
        [[RuntimeError("stale device table")], [unused_stream]],
    )
    old_instance = core.pyaudio_instance
    old_instance.on_terminate = core._invalidate_recording_token
    try:
        await core._start_recording()

        replacement = factory.instances[1]
        assert len(factory.instances) == 2
        assert old_instance.terminate_threads
        assert core.pyaudio_instance is replacement
        assert replacement.open_threads == []
        assert core.audio_stream is None
        assert unused_stream.started is False
        assert core.state is IMEState.IDLE
        assert core._active_recording_token is None
        assert core._audio_consumer_paused is True
        assert core._audio_drained.is_set()
        assert core._session_started is False
    finally:
        cleanup_core(core)


@pytest.mark.asyncio
async def test_rebuild_guard_rejects_recording_and_concurrent_attempt(monkeypatch):
    core, factory = make_core(monkeypatch, [[], [], []])
    old_instance = core.pyaudio_instance
    try:
        core.state = IMEState.RECORDING
        assert await core.rebuild_audio_device_table() is False
        assert core.pyaudio_instance is old_instance
        assert len(factory.instances) == 1
        assert old_instance.terminate_threads == []

        core.state = IMEState.IDLE
        core._audio_rebuild_in_progress = True
        assert await core.rebuild_audio_device_table() is False
        assert core.pyaudio_instance is old_instance
        assert len(factory.instances) == 1
        assert old_instance.terminate_threads == []
    finally:
        core._audio_rebuild_in_progress = False
        cleanup_core(core)


@pytest.mark.asyncio
async def test_rebuild_closes_live_stream_before_terminating_instance(
    monkeypatch,
    capsys,
):
    events = []
    live_stream = FakeStream(events)
    core, factory = make_core(monkeypatch, [[], []], events=events)
    old_instance = core.pyaudio_instance
    core.audio_stream = live_stream
    events.clear()
    try:
        assert await core.rebuild_audio_device_table() is True

        output = capsys.readouterr().out
        replacement = factory.instances[1]
        assert "PortAudio rebuild succeeded (attempt 1/1)" in output
        assert core.audio_stream is None
        assert core.pyaudio_instance is replacement
        assert events.index("stream.stop") < events.index("instance-0.terminate")
        assert events.index("stream.close") < events.index("instance-0.terminate")
        assert events.index("instance-0.terminate") < events.index("instance-1.create")
        assert all(
            name.startswith("pyaudio") for name in old_instance.terminate_threads
        )
        assert all(name.startswith("pyaudio") for name in factory.creation_threads)
    finally:
        cleanup_core(core)


@pytest.mark.asyncio
async def test_rebuild_failure_is_logged_and_handled_for_restart(monkeypatch, capsys):
    core, factory = make_core(monkeypatch, [[], RuntimeError("initialize stuck")])
    old_instance = core.pyaudio_instance
    try:
        assert await core.rebuild_audio_device_table() is True

        output = capsys.readouterr().out
        assert "PortAudio rebuild failed (attempt 1/1): initialize stuck" in output
        assert old_instance.terminate_threads
        assert core.pyaudio_instance is None
        assert core._audio_rebuild_in_progress is False
        assert len(factory.instances) == 1
    finally:
        cleanup_core(core)


@pytest.mark.asyncio
async def test_repeated_rebuild_then_cleanup_is_idempotent(monkeypatch):
    core, factory = make_core(monkeypatch, [[], [], []])

    assert await core.rebuild_audio_device_table() is True
    assert await core.rebuild_audio_device_table() is True
    current_instance = core.pyaudio_instance

    cleanup_core(core)
    first_counts = [len(instance.terminate_threads) for instance in factory.instances]
    core.cleanup()

    assert first_counts == [1, 1, 1]
    assert [len(instance.terminate_threads) for instance in factory.instances] == first_counts
    assert len(current_instance.terminate_threads) == 1
    assert all(
        name.startswith("pyaudio")
        for instance in factory.instances
        for name in instance.terminate_threads
    )
    assert core.pyaudio_instance is None


def test_cleanup_timeout_does_not_wait_again_for_executor(capsys):
    class TimeoutFuture:
        def result(self, timeout):
            assert timeout == 2
            raise concurrent.futures.TimeoutError

    class TimeoutExecutor:
        def __init__(self):
            self.shutdown_calls = []

        def submit(self, _callable):
            return TimeoutFuture()

        def shutdown(self, **kwargs):
            self.shutdown_calls.append(kwargs)

    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core._active_recording_token = object()
    core.loop = None
    core._recent_audio_worker = None
    core._pyaudio_executor_shutdown = False
    core._pyaudio_executor = TimeoutExecutor()

    core.cleanup()

    output = capsys.readouterr().out
    assert "PortAudio terminate timed out after 2s" in output
    assert core._pyaudio_executor.shutdown_calls == [
        {"wait": False, "cancel_futures": True}
    ]
    assert core._pyaudio_executor_shutdown is True


@pytest.mark.asyncio
async def test_restart_rebuilds_first_and_continues_after_audio_error(
    monkeypatch,
):
    order = []

    class FakeCore:
        state = IMEState.IDLE
        ws = object()
        ws_connected = True
        _ws_connected_wall_ts = 1.0
        _start_requested = True
        _server_connected = True
        _session_task = None
        _receive_task = object()
        _processing_entered_ts = 1.0

        async def rebuild_audio_device_table(self):
            order.append("audio-rebuild")
            raise RuntimeError("device layer unavailable")

        async def disconnect_websocket(self):
            order.append("disconnect")

        def _set_state(self, state):
            self.state = state

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app.core = FakeCore()
    app.config = Config()
    app.loop = asyncio.get_running_loop()
    app._terminate_server_processes = lambda: order.append("backend-terminate")
    port_checks = iter((False, True))
    app._is_server_port_open = lambda: next(port_checks)
    app._write_server_pid_file = lambda _pid: order.append("pid-written")

    async def connect_async():
        order.append("connect")

    app._connect_async = connect_async
    monkeypatch.setattr(
        ime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(pid=12345),
    )

    await app._restart_service_async()

    assert order.index("audio-rebuild") < order.index("disconnect")
    assert order.index("disconnect") < order.index("backend-terminate")
    assert "pid-written" in order
    assert "connect" in order
