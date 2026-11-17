import asyncio
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest
from websockets.exceptions import ConnectionClosedOK

import ime_menubar
from backend_runtime import (
    BACKEND_ECHO,
    BACKEND_WHISPER,
    BackendCoordinator,
    BackendSelectionStore,
    build_default_descriptors,
)
from ime_menubar import (
    BrainwaveIMEApp,
    BrainwaveIMECore,
    Config,
    IMEState,
    RecordingMode,
)


class _NativeItem:
    def __init__(self):
        self.enabled = None
        self.title = ""
        self.state = 0
        self.view = None
        self.action = None
        self.menu = None

    def setEnabled_(self, enabled):
        self.enabled = bool(enabled)

    def setTitle_(self, title):
        self.title = str(title)

    def setState_(self, state):
        self.state = int(state)

    def setView_(self, view):
        self.view = view

    def setAction_(self, action):
        self.action = action


class _MenuItem:
    def __init__(self):
        self._menuitem = _NativeItem()

    @property
    def title(self):
        return self._menuitem.title

    @title.setter
    def title(self, title):
        self._menuitem.setTitle_(title)

    @property
    def state(self):
        return self._menuitem.state

    @state.setter
    def state(self, state):
        self._menuitem.setState_(state)


class _DescriptorSupervisor:
    def __init__(self, descriptors):
        self.descriptors = descriptors

    def descriptor(self, key):
        return self.descriptors[key]


class _LifecycleSupervisor(_DescriptorSupervisor):
    def __init__(self, descriptors):
        super().__init__(descriptors)
        self.started = []
        self.restarted = []
        self.stopped = []
        self.stop_failure = None

    def start(self, key, *, timeout=10.0):
        self.started.append(key)

    def restart(self, key, *, timeout=10.0):
        self.restarted.append(key)

    def stop(self, key, *, timeout=5.0):
        self.stopped.append(key)
        if self.stop_failure is not None and key == BACKEND_ECHO:
            raise self.stop_failure


def _prime_lifecycle(app):
    app._lifecycle_state_lock = threading.Lock()
    app._lifecycle_operation_generation = 0
    app._active_lifecycle_operation = None
    app._active_lifecycle_kind = None
    app._switch_in_progress = False


def _prime_backend_rows(app, descriptors):
    """Stand up the header row plus one row per registered lane."""
    app.backend_header_item = _MenuItem()
    app.backend_row_items = {}
    for key in ime_menubar.BACKEND_MENU_ORDER:
        if key not in descriptors:
            continue
        item = _MenuItem()
        item._backend_key = key
        app.backend_row_items[key] = item
    return app.backend_row_items


def _checked_lane(app):
    """The single lane carrying the check mark, or a failure if not single."""
    checked = [
        key for key, item in app.backend_row_items.items() if item.state == 1
    ]
    assert len(checked) == 1, checked
    return checked[0]


def _prime_backend_presentation(app):
    _prime_backend_rows(app, app._backend_supervisor.descriptors)
    app._sync_provider_menu = lambda: None
    app._sync_model_menu = lambda: None
    app._sync_status_menu = lambda _state: None


def test_icons_hotkey_and_lane_row_menu_state(tmp_path):
    assert BrainwaveIMEApp.STATE_ICONS == {
        IMEState.IDLE: "🌊",
        IMEState.RECORDING: "🔵",
        IMEState.PROCESSING: "🌀",
        IMEState.DISCONNECTED: "⚫",
    }
    assert Config().hotkey_keycode == 50

    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = _DescriptorSupervisor(descriptors)
    app._current_backend = BACKEND_ECHO
    app._switch_in_progress = False
    rows = _prime_backend_rows(app, descriptors)
    # No sibling fork here, so the picker is the two lanes this repo owns.
    assert list(rows) == [BACKEND_ECHO, BACKEND_WHISPER]

    app._sync_backend_menu(IMEState.IDLE)
    assert app.backend_header_item.title == "Using: EchoWave"
    assert app.backend_header_item._menuitem.enabled is False
    assert rows[BACKEND_ECHO].title == "EchoWave"
    assert rows[BACKEND_WHISPER].title == "WhisperWave"
    assert _checked_lane(app) == BACKEND_ECHO
    assert rows[BACKEND_ECHO]._menuitem.enabled is False
    assert rows[BACKEND_WHISPER]._menuitem.enabled is True

    app._current_backend = BACKEND_WHISPER
    app._sync_backend_menu(IMEState.DISCONNECTED)
    assert app.backend_header_item.title == "Using: WhisperWave"
    assert _checked_lane(app) == BACKEND_WHISPER
    assert rows[BACKEND_WHISPER]._menuitem.enabled is False
    assert rows[BACKEND_ECHO]._menuitem.enabled is True

    for state in (IMEState.RECORDING, IMEState.PROCESSING):
        app._sync_backend_menu(state)
        assert app.backend_header_item.title == "Using: WhisperWave", state
        assert _checked_lane(app) == BACKEND_WHISPER, state
        for key, item in rows.items():
            assert item._menuitem.enabled is False, (state, key)


def test_app_installs_header_then_lane_rows_as_the_first_menu_block(
    tmp_path,
    monkeypatch,
):
    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    supervisor = _LifecycleSupervisor(descriptors)
    store = BackendSelectionStore(tmp_path / "selection.json")
    registered_timers = []

    class FakeCore:
        def __init__(self, config, **_callbacks):
            self.config = config
            self.state = IMEState.DISCONNECTED

    class FakeStateTimer:
        def __init__(self, callback, interval):
            self.callback = callback
            self.interval = interval
            self._nstimer = object()
            self.start_calls = 0

        def start(self):
            self.start_calls += 1

    class FakeDelayedTimer:
        def __init__(self, interval, callback):
            self.interval = interval
            self.callback = callback
            self.start_calls = 0

        def start(self):
            self.start_calls += 1

    monkeypatch.setattr(ime_menubar, "BrainwaveIMECore", FakeCore)
    monkeypatch.setattr(ime_menubar.rumps, "Timer", FakeStateTimer)
    monkeypatch.setattr(ime_menubar.threading, "Timer", FakeDelayedTimer)
    monkeypatch.setattr(
        ime_menubar,
        "_register_timer_for_menu_tracking",
        lambda timer, run_loop=None: registered_timers.append(timer) or timer._nstimer,
    )
    monkeypatch.setattr(ime_menubar, "_accessibility_is_trusted", lambda prompt=False: True)
    monkeypatch.setattr(ime_menubar, "_input_monitoring_is_allowed", lambda prompt=False: True)
    monkeypatch.setenv("BRAINWAVE_HISTORY_DIR", str(tmp_path / "history"))

    app = BrainwaveIMEApp(
        supervisor=supervisor,
        selection_store=store,
        startup_decision=SimpleNamespace(
            requested_backend=BACKEND_ECHO,
            effective_backend=BACKEND_ECHO,
            diagnostic=None,
        ),
    )

    native_menu = app.menu._menu
    # Header, then one row per lane, then the separator, then Status.
    assert native_menu.itemAtIndex_(0) is app.backend_header_item._menuitem
    assert list(app.backend_row_items) == [BACKEND_ECHO, BACKEND_WHISPER]
    assert native_menu.itemAtIndex_(1) is app.backend_row_items[BACKEND_ECHO]._menuitem
    assert (
        native_menu.itemAtIndex_(2) is app.backend_row_items[BACKEND_WHISPER]._menuitem
    )
    assert native_menu.itemAtIndex_(3).isSeparatorItem()
    assert native_menu.itemAtIndex_(4) is app.status_item._menuitem

    assert app.backend_header_item.title == "Using: EchoWave"
    assert app.backend_header_item._menuitem.title() == "Using: EchoWave"
    assert app.backend_header_item._menuitem.isEnabled() is False
    assert app.backend_row_items[BACKEND_ECHO]._menuitem.title() == "EchoWave"
    assert app.backend_row_items[BACKEND_ECHO]._menuitem.state() == 1
    assert app.backend_row_items[BACKEND_WHISPER]._menuitem.state() == 0
    # The rows are ordinary command items again: real actions, no custom view.
    assert app.backend_row_items[BACKEND_ECHO]._menuitem.view() is None
    assert app.backend_row_items[BACKEND_WHISPER]._menuitem.action() is not None
    assert app._state_timer.start_calls == 1
    assert registered_timers == [app._state_timer]


def test_tracking_mode_registration_reuses_started_native_timer_and_fails_loudly():
    native_timer = object()
    timer = SimpleNamespace(_nstimer=native_timer)

    class FakeRunLoop:
        def __init__(self):
            self.calls = []

        def addTimer_forMode_(self, candidate, mode):
            self.calls.append((candidate, mode))

    run_loop = FakeRunLoop()
    result = ime_menubar._register_timer_for_menu_tracking(timer, run_loop=run_loop)

    assert result is native_timer
    assert run_loop.calls == [
        (native_timer, ime_menubar.NSEventTrackingRunLoopMode)
    ]
    with pytest.raises(RuntimeError, match="_nstimer"):
        ime_menubar._register_timer_for_menu_tracking(
            SimpleNamespace(),
            run_loop=run_loop,
        )


def test_selected_backend_descriptor_repoints_only_endpoint_and_next_turn_metadata(tmp_path):
    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_port=32001,
        whisper_port=32002,
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core.state = IMEState.IDLE
    core.config = Config()
    core.apply_backend_descriptor(descriptors[BACKEND_WHISPER])
    assert core.config.backend_key == BACKEND_WHISPER
    assert core.config.server_port == 32002
    assert core.config.provider == "openai-transcription"
    assert core.config.model == descriptors[BACKEND_WHISPER].model


@pytest.mark.asyncio
async def test_stale_connection_generation_is_dropped_before_text_or_paste():
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core._connection_generation = 2
    await core._handle_message(
        {"type": "text", "content": "stale", "turn_id": 1},
        connection_generation=1,
    )


@pytest.mark.asyncio
async def test_terminal_and_paste_are_exact_once_with_backend_aware_history_callback():
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core._connection_generation = 1
    core.state = IMEState.PROCESSING
    core._active_turn_id = 7
    core._terminal_turn_ids = set()
    core._failed_turn_ids = set()
    core._audio_drop_count = 0
    core._audio_ingestion_failed_reason = None
    core._last_turn_completed_ts = None
    core._last_turn_completed_wall_ts = None
    core._response_done_ts = None
    core._stop_pressed_ts = None
    core.recording_mode = RecordingMode.OPTIMIZED
    core._start_requested = True
    core._server_connected_event = asyncio.Event()
    core._server_connected = True
    core.transcript = "hello"
    core._active_turn_provider = "openai-transcription"
    core._active_turn_model = "gpt-4o-transcribe"
    core._active_turn_backend = BACKEND_WHISPER
    core._archive_recent_turn_audio = lambda _outcome: True
    core._archive_failed_turn_audio = lambda _reason: None
    core._play_sound = lambda _name: None
    core.on_transcript = None
    completions = []
    pastes = []
    core.on_transcript_complete = lambda *args: completions.append(args)

    def set_state(state):
        core.state = state

    async def input_text(text):
        pastes.append(text)

    core._set_state = set_state
    core._input_text = input_text
    terminal = {"type": "status", "status": "idle", "turn_id": 7}
    await core._handle_message(terminal, connection_generation=1)
    # Even if a reconnect/race puts the state back into processing, the same
    # turn's terminal token cannot paste twice.
    core.state = IMEState.PROCESSING
    await core._handle_message(terminal, connection_generation=1)

    assert pastes == ["hello"]
    assert len(completions) == 1
    assert completions[0][2:] == (
        "openai-transcription",
        "gpt-4o-transcribe",
        BACKEND_WHISPER,
    )


@pytest.mark.asyncio
async def test_processing_warning_keeps_channel_open_for_slow_terminal():
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core._connection_generation = 4
    core.state = IMEState.PROCESSING
    core._active_turn_id = 9
    core._processing_entered_ts = time.perf_counter() - 31
    core._processing_warning_sent = False
    core._terminal_turn_ids = set()
    core._failed_turn_ids = set()
    core._audio_drop_count = 0
    core._audio_ingestion_failed_reason = None
    core._last_turn_completed_ts = None
    core._last_turn_completed_wall_ts = None
    core._response_done_ts = None
    core._stop_pressed_ts = None
    core.recording_mode = RecordingMode.OPTIMIZED
    core._start_requested = True
    core._server_connected_event = asyncio.Event()
    core._server_connected = True
    core.transcript = "slow final"
    core._active_turn_provider = "openai-transcription"
    core._active_turn_model = "gpt-4o-transcribe"
    core._active_turn_backend = BACKEND_WHISPER
    core._archive_recent_turn_audio = lambda _outcome: True
    core._archive_failed_turn_audio = lambda _reason: None
    core._play_sound = lambda _name: None
    core.on_transcript = None
    core.on_transcript_complete = None
    core.config = SimpleNamespace(
        processing_timeout_sec=30,
        processing_hard_timeout_sec=120,
    )
    warnings = []
    pastes = []
    core.on_processing_delay = lambda turn_id, elapsed: warnings.append((turn_id, elapsed))

    def set_state(state):
        core.state = state

    async def input_text(text):
        pastes.append(text)

    core._set_state = set_state
    core._input_text = input_text

    terminal_failure = await core._check_processing_timeout_async()
    assert terminal_failure is False
    assert core.state == IMEState.PROCESSING
    assert core._processing_warning_sent is True
    assert warnings and warnings[0][0] == 9

    await core._handle_message(
        {"type": "status", "status": "idle", "turn_id": 9},
        connection_generation=4,
    )
    assert pastes == ["slow final"]
    assert core.state == IMEState.IDLE
    assert 9 in core._terminal_turn_ids


@pytest.mark.asyncio
async def test_processing_hard_timeout_is_explicit_failure_not_fake_success():
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core.state = IMEState.PROCESSING
    core._active_turn_id = 10
    core._processing_entered_ts = time.perf_counter() - 121
    core._processing_warning_sent = False
    core._failed_turn_ids = set()
    core.config = SimpleNamespace(
        processing_timeout_sec=30,
        processing_hard_timeout_sec=120,
    )
    core.on_processing_delay = None
    outcomes = []
    core._archive_failed_turn_audio = lambda reason: outcomes.append(("failed", reason)) or "/tmp/f.wav"
    core._archive_recent_turn_audio = lambda reason: outcomes.append(("recent", reason)) or True
    core._play_sound = lambda _name: None

    async def disconnect():
        core.state = IMEState.DISCONNECTED

    core.disconnect_websocket = disconnect
    assert await core._check_processing_timeout_async() is True
    assert 10 in core._failed_turn_ids
    assert core.state == IMEState.DISCONNECTED
    assert outcomes == [
        ("failed", "processing_hard_timeout"),
        ("recent", "processing_hard_timeout"),
    ]


@pytest.mark.asyncio
async def test_processing_connect_preserves_turn_and_rejects_unscoped_bootstrap_idle(
    monkeypatch,
):
    """R1-S4: socket bootstrap cannot acquire terminal ownership of a turn."""

    class FakeWebSocket:
        async def close(self):
            pass

    async def connect(_uri):
        return FakeWebSocket()

    monkeypatch.setattr(ime_menubar.websockets, "connect", connect)
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core.config = Config(server_host="127.0.0.1", server_port=32123)
    core.state = IMEState.PROCESSING
    core.recording_mode = RecordingMode.OPTIMIZED
    core._active_turn_id = 21
    core._connection_generation = 4
    core._receive_task = asyncio.current_task()
    core._ensure_audio_pipeline = lambda: None
    core._ws_connected_wall_ts = None
    core._terminal_turn_ids = set()
    core._failed_turn_ids = set()
    core._audio_drop_count = 0
    core._audio_ingestion_failed_reason = None
    core._last_turn_completed_ts = None
    core._last_turn_completed_wall_ts = None
    core._response_done_ts = None
    core._stop_pressed_ts = None
    core._processing_entered_ts = time.perf_counter()
    core._processing_warning_sent = False
    core._start_requested = True
    core._server_connected_event = asyncio.Event()
    core._server_connected = False
    core.transcript = "matching terminal"
    core._active_turn_provider = "openai-transcription"
    core._active_turn_model = "gpt-4o-transcribe"
    core._active_turn_backend = BACKEND_WHISPER
    core._archive_recent_turn_audio = lambda _outcome: True
    core._archive_failed_turn_audio = lambda _reason: None
    core._play_sound = lambda _name: None
    core.on_state_change = None
    core.on_transcript = None
    core.on_transcript_complete = None
    pastes = []

    async def input_text(text):
        pastes.append(text)

    core._input_text = input_text

    assert await core.connect_websocket() is True
    assert core.state == IMEState.PROCESSING
    await core._handle_message(
        {"type": "status", "status": "connected"},
        connection_generation=5,
    )
    assert core._server_connected is False
    await core._handle_message(
        {"type": "status", "status": "connected", "turn_id": 21},
        connection_generation=5,
    )
    assert core._server_connected is True
    await core._handle_message(
        {"type": "status", "status": "idle"},
        connection_generation=5,
    )
    assert core.state == IMEState.PROCESSING
    assert pastes == []

    await core._handle_message(
        {"type": "status", "status": "idle", "turn_id": 21},
        connection_generation=5,
    )
    assert core.state == IMEState.IDLE
    assert pastes == ["matching terminal"]


@pytest.mark.asyncio
async def test_processing_transport_close_fails_and_archives_before_reconnect_bootstrap(
    monkeypatch,
):
    """R1-S4: terminal-channel loss is explicit failure, never silent success."""

    class ClosingWebSocket:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionClosedOK(None, None)

        async def close(self):
            pass

    class BootstrapWebSocket:
        async def close(self):
            pass

    async def reconnect(_uri):
        return BootstrapWebSocket()

    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core.config = Config(server_host="127.0.0.1", server_port=32124)
    core.state = IMEState.PROCESSING
    core.recording_mode = RecordingMode.OPTIMIZED
    core._active_turn_id = 22
    core._connection_generation = 8
    core._receive_task = asyncio.current_task()
    core.ws = ClosingWebSocket()
    core.ws_connected = True
    core._ws_connected_wall_ts = time.time()
    core._start_requested = True
    core._server_connected_event = asyncio.Event()
    core._server_connected = True
    core._failed_turn_ids = set()
    core._terminal_turn_ids = set()
    core._ensure_audio_pipeline = lambda: None
    core._stop_and_drain_audio_capture = lambda: asyncio.sleep(0)
    archives = []
    core._archive_failed_turn_audio = lambda reason: archives.append(("failed", reason)) or "/tmp/f.wav"
    core._archive_recent_turn_audio = lambda reason: archives.append(("recent", reason)) or True
    core._play_sound = lambda _name: None
    core.on_state_change = None
    core.transcript = "must not paste"
    pastes = []

    async def input_text(text):
        pastes.append(text)

    core._input_text = input_text

    await core.receive_messages(core.ws, generation=8)
    assert core.state == IMEState.DISCONNECTED
    assert core._failed_turn_ids == {22}
    assert archives == [
        ("failed", "transport_closed_before_terminal"),
        ("recent", "transport_closed_before_terminal"),
    ]

    monkeypatch.setattr(ime_menubar.websockets, "connect", reconnect)
    core._receive_task = asyncio.current_task()
    assert await core.connect_websocket() is True
    await core._handle_message({"type": "status", "status": "idle"})
    await core._handle_message(
        {"type": "status", "status": "idle", "turn_id": 22}
    )
    assert core.state == IMEState.IDLE
    assert pastes == []
    assert core._terminal_turn_ids == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["close", "error"])
async def test_recording_transport_loss_preserves_capture_and_recovers_same_turn(
    failure_kind,
    monkeypatch,
):
    """R2-S4-1: recording transport churn rebuilds without aborting capture."""

    class FailingWebSocket:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if failure_kind == "close":
                raise ConnectionClosedOK(None, None)
            raise RuntimeError("synthetic receive failure")

        async def close(self):
            pass

    sent = []

    class ReconnectedWebSocket:
        async def send(self, message):
            sent.append(message)

        async def close(self):
            pass

    async def reconnect(_uri):
        return ReconnectedWebSocket()

    monkeypatch.setattr(ime_menubar.websockets, "connect", reconnect)
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core.config = Config(server_host="127.0.0.1", server_port=32125)
    core.state = IMEState.RECORDING
    core.recording_mode = RecordingMode.OPTIMIZED
    core._active_turn_id = 23
    core._connection_generation = 12
    core._receive_task = asyncio.current_task()
    capture = object()
    core.audio_stream = capture
    core.ws = FailingWebSocket()
    core.ws_connected = True
    core._ws_connected_wall_ts = time.time()
    core._start_requested = True
    core._server_connected_event = asyncio.Event()
    core._server_connected = True
    core._failed_turn_ids = set()
    core._terminal_turn_ids = set()
    core._audio_drop_count = 0
    core._audio_ingestion_failed_reason = None
    core._ensure_audio_pipeline = lambda: None
    core._session_prompt_mode = "optimize"
    core._force_ws_refresh_before_turn = False
    core._session_start_attempts = 0
    core._max_session_start_attempts = 3
    core._session_start_backoff_base_sec = 0
    core._session_start_backoff_cap_sec = 0
    core._active_recording_token = (1, 23)
    core._processing_entered_ts = None
    core._processing_warning_sent = False
    core._last_turn_completed_ts = None
    core._last_turn_completed_wall_ts = None
    core._response_done_ts = None
    core._stop_pressed_ts = None
    core._active_turn_provider = "openai-transcription"
    core._active_turn_model = "gpt-4o-transcribe"
    core._active_turn_backend = BACKEND_WHISPER
    core.transcript = "must not paste before stop"
    core.on_state_change = None
    core.on_transcript = None
    core.on_transcript_complete = None
    capture_closes = []
    archives = []
    recovery_starts = []
    pastes = []

    async def close_capture():
        capture_closes.append(core.audio_stream)

    async def input_text(text):
        pastes.append(text)

    core._stop_and_drain_audio_capture = close_capture
    core._archive_failed_turn_audio = lambda reason: archives.append(("failed", reason))
    core._archive_recent_turn_audio = lambda reason: archives.append(("recent", reason))
    core._play_sound = lambda _name: None
    core._input_text = input_text
    core._start_session_task = lambda: recovery_starts.append(
        (core._active_turn_id, core.state, core.audio_stream)
    )

    await core.receive_messages(core.ws, generation=12)

    assert core.state == IMEState.RECORDING
    assert core._active_turn_id == 23
    assert core.audio_stream is capture
    assert capture_closes == []
    assert archives == []
    assert pastes == []
    assert core._failed_turn_ids == set()
    assert recovery_starts == [(23, IMEState.RECORDING, capture)]
    assert core.ws is None
    assert core.ws_connected is False
    assert core._start_requested is False
    assert core._server_connected is False

    core._receive_task = asyncio.current_task()
    assert await core.connect_websocket() is True
    generation = core._connection_generation
    await core._ensure_session_started("optimize")
    start_payload = json.loads(sent[-1])
    assert start_payload["type"] == "start_recording"
    assert start_payload["turn_id"] == 23

    await core._handle_message(
        {"type": "status", "status": "idle"},
        connection_generation=generation,
    )
    await core._handle_message(
        {"type": "status", "status": "idle", "turn_id": 23},
        connection_generation=generation,
    )
    assert core.state == IMEState.RECORDING
    assert pastes == []
    assert core._terminal_turn_ids == set()

    async def finish_stop_without_transport_io():
        pass

    core._async_stop = finish_stop_without_transport_io
    await core._stop_recording()
    assert core.state == IMEState.PROCESSING
    await core._handle_message(
        {
            "type": "text",
            "content": "matching continued terminal",
            "isNewResponse": True,
            "turn_id": 23,
        },
        connection_generation=generation,
    )
    await core._handle_message(
        {"type": "status", "status": "idle", "turn_id": 23},
        connection_generation=generation,
    )
    assert core.state == IMEState.IDLE
    assert pastes == ["matching continued terminal"]
    assert core._terminal_turn_ids == {23}
    assert core._failed_turn_ids == set()


@pytest.mark.asyncio
async def test_capture_queue_success_has_zero_drop_and_saturation_fails_explicitly():
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core._audio_queue = queue.Queue(maxsize=1)
    core._audio_failure_lock = threading.RLock()
    core._audio_ingestion_failed_reason = None
    core._audio_drop_count = 0
    core._audio_backpressure_failure_count = 0
    core._audio_abort_task = None
    core.loop = asyncio.get_running_loop()
    aborts = []

    async def abort(crossing_item, reason):
        aborts.append((crossing_item, reason))

    core._abort_audio_ingestion = abort
    first = (b"first", 1.0, 1)
    second = (b"second", 2.0, 1)
    assert core._enqueue_audio(*first) is True
    assert core._audio_drop_count == 0
    assert core._enqueue_audio(*second) is False
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert core._audio_ingestion_failed_reason == "capture_backpressure_exhausted"
    assert core._audio_backpressure_failure_count == 1
    assert core._audio_drop_count == 0
    assert aborts == [(second, "capture_backpressure_exhausted")]


@pytest.mark.asyncio
async def test_failed_ingestion_drains_accepted_frames_to_spool_without_buffer_growth():
    class _IdentityProcessor:
        @staticmethod
        def resample(chunk):
            return chunk

    class _Spool:
        def __init__(self):
            self.chunks = []

        def append(self, chunk):
            self.chunks.append(chunk)

    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core._audio_queue = queue.Queue()
    core._audio_queue.put((b"accepted-1", 1.0, 3))
    core._audio_queue.put((b"accepted-2", 2.0, 3))
    core._audio_queue.put(None)
    core._audio_drained = asyncio.Event()
    core._audio_consumer_paused = False
    core._active_turn_id = 3
    core._last_audio_callback_ts = None
    core._audio_ingestion_failed_reason = "capture_backpressure_exhausted"
    core.audio_processor = _IdentityProcessor()
    core._turn_audio_spool = _Spool()

    def reject_buffer_growth(_chunk):
        raise AssertionError("failed turn must not grow upload backlog")

    core._append_audio_buffer = reject_buffer_growth

    consumer = asyncio.create_task(core._audio_consumer_loop())
    await asyncio.wait_for(core._audio_drained.wait(), timeout=1)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert core._turn_audio_spool.chunks == [b"accepted-1", b"accepted-2"]


@pytest.mark.asyncio
async def test_delayed_audio_open_after_stop_closes_candidate_and_never_starts_session(
    tmp_path,
    monkeypatch,
):
    """R1-S5: a stale executor result cannot resurrect microphone ownership."""

    open_entered = threading.Event()
    release_open = threading.Event()

    class FakeStream:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.closed = False

        def start_stream(self):
            self.started = True

        def stop_stream(self):
            self.stopped = True

        def close(self):
            self.closed = True

    stream = FakeStream()

    class BlockingPyAudio:
        def open(self, **_kwargs):
            open_entered.set()
            if not release_open.wait(timeout=3):
                raise TimeoutError("test did not release blocking open")
            return stream

        def terminate(self):
            pass

    fake_audio = BlockingPyAudio()
    monkeypatch.setattr(ime_menubar.pyaudio, "PyAudio", lambda: fake_audio)
    core = BrainwaveIMECore(Config())
    core.loop = asyncio.get_running_loop()
    core.state = IMEState.IDLE
    core._audio_spool_dir = str(tmp_path / "spool")
    core._recent_audio_cache_enabled = False
    core.config.stop_tail_wait_min_ms = 0
    core.config.stop_tail_wait_max_ms = 0
    core.config.stop_tail_wait_guard_ms = 0
    core._stop_session_connect_deadline_sec = 0
    core._play_sound = lambda _name: None
    session_starts = []
    core._start_session_task = lambda: session_starts.append(core.state)

    start_task = asyncio.create_task(core._start_recording())
    assert await asyncio.to_thread(open_entered.wait, 2)
    turn_id = core._active_turn_id
    assert core.state == IMEState.RECORDING

    await core._stop_recording()
    assert core.state == IMEState.IDLE
    release_open.set()
    await asyncio.wait_for(start_task, timeout=2)

    assert stream.closed is True
    assert core.audio_stream is None
    assert session_starts == []
    assert core._failed_turn_ids == {turn_id}
    assert core._terminal_turn_ids == set()

    if core._audio_consumer_task and not core._audio_consumer_task.done():
        core._audio_consumer_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await core._audio_consumer_task
    core._cleanup_turn_spool()
    core._pyaudio_executor.shutdown(wait=True, cancel_futures=True)


def test_history_entry_records_actual_backend_model_and_provider():
    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._pending_transcripts = queue.Queue()
    app._on_transcript_complete(
        "hello",
        RecordingMode.OPTIMIZED,
        "openai-transcription",
        "gpt-4o-transcribe",
        BACKEND_WHISPER,
    )
    entry = app._pending_transcripts.get_nowait()
    assert entry["backend"] == BACKEND_WHISPER
    assert entry["provider"] == "openai-transcription"
    assert entry["model"] == "gpt-4o-transcribe"


@pytest.mark.asyncio
async def test_failed_target_connect_restores_real_core_and_keeps_ui_selection_aligned(
    tmp_path,
    monkeypatch,
):
    """R1-S2: exercise the mutating BrainwaveIMECore repoint path, not a no-op mock."""

    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_port=32201,
        whisper_port=32202,
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    supervisor = _LifecycleSupervisor(descriptors)
    store = BackendSelectionStore(tmp_path / "selection.json")
    store.save(BACKEND_ECHO)
    coordinator = BackendCoordinator(supervisor, store)

    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core.state = IMEState.IDLE
    core.config = Config()
    core.apply_backend_descriptor(descriptors[BACKEND_ECHO])
    connection_attempts = []

    async def disconnect():
        core.state = IMEState.DISCONNECTED

    async def connect():
        connection_attempts.append(core.config.backend_key)
        if core.config.backend_key == BACKEND_WHISPER:
            core.state = IMEState.DISCONNECTED
            return False
        core.state = IMEState.IDLE
        return True

    core.disconnect_websocket = disconnect
    core.connect_websocket = connect

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = supervisor
    app._backend_coordinator = coordinator
    app._backend_events = queue.Queue()
    app._current_backend = BACKEND_ECHO
    app._selected_provider = descriptors[BACKEND_ECHO].provider
    app._selected_model = descriptors[BACKEND_ECHO].model
    app.core = core
    _prime_lifecycle(app)
    _prime_backend_presentation(app)
    notifications = []
    monkeypatch.setattr(
        ime_menubar.rumps,
        "notification",
        lambda *args: notifications.append(args),
    )

    await app._switch_backend_async(BACKEND_WHISPER)
    app._drain_backend_events()

    assert store.load() == BACKEND_ECHO
    assert app._current_backend == BACKEND_ECHO
    assert app._selected_provider == descriptors[BACKEND_ECHO].provider
    assert app._selected_model == descriptors[BACKEND_ECHO].model
    assert core.config.backend_key == BACKEND_ECHO
    assert core.config.server_port == descriptors[BACKEND_ECHO].port
    assert core.config.provider == descriptors[BACKEND_ECHO].provider
    assert core.config.model == descriptors[BACKEND_ECHO].model
    assert core.state == IMEState.IDLE
    assert connection_attempts == [
        BACKEND_WHISPER,
        BACKEND_ECHO,
        BACKEND_ECHO,
    ]
    assert supervisor.stopped == [BACKEND_WHISPER]
    assert app.backend_header_item.title == "Using: EchoWave"
    assert _checked_lane(app) == BACKEND_ECHO
    assert app.backend_row_items[BACKEND_ECHO]._menuitem.enabled is False
    assert app.backend_row_items[BACKEND_WHISPER]._menuitem.enabled is True
    assert notifications and notifications[0][1] == "Backend operation failed"


@pytest.mark.asyncio
async def test_old_cleanup_failure_keeps_core_selection_and_ui_committed_to_target(
    tmp_path,
    monkeypatch,
):
    """R1-S3: UI commit follows persisted/core target despite cleanup warning."""

    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_port=32301,
        whisper_port=32302,
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    supervisor = _LifecycleSupervisor(descriptors)
    supervisor.stop_failure = OSError("injected old-child stop failure")
    store = BackendSelectionStore(tmp_path / "selection.json")
    store.save(BACKEND_ECHO)

    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core.state = IMEState.IDLE
    core.config = Config()
    core.apply_backend_descriptor(descriptors[BACKEND_ECHO])

    async def disconnect():
        core.state = IMEState.DISCONNECTED

    async def connect():
        core.state = IMEState.IDLE
        return True

    core.disconnect_websocket = disconnect
    core.connect_websocket = connect

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = supervisor
    app._backend_coordinator = BackendCoordinator(supervisor, store)
    app._backend_events = queue.Queue()
    app._current_backend = BACKEND_ECHO
    app._selected_provider = descriptors[BACKEND_ECHO].provider
    app._selected_model = descriptors[BACKEND_ECHO].model
    app.core = core
    _prime_lifecycle(app)
    _prime_backend_presentation(app)
    notifications = []
    monkeypatch.setattr(
        ime_menubar.rumps,
        "notification",
        lambda *args: notifications.append(args),
    )

    await app._switch_backend_async(BACKEND_WHISPER)

    assert store.load() == BACKEND_WHISPER
    assert app._current_backend == BACKEND_WHISPER
    assert app._selected_provider == descriptors[BACKEND_WHISPER].provider
    assert app._selected_model == descriptors[BACKEND_WHISPER].model
    assert core.config.backend_key == BACKEND_WHISPER
    event = app._backend_events.queue[0]
    assert event[0] == "switched"
    assert event[1] == BACKEND_WHISPER
    assert "injected old-child stop failure" in event[2]
    app._drain_backend_events()
    assert app.backend_header_item.title == "Using: WhisperWave"
    assert _checked_lane(app) == BACKEND_WHISPER
    assert app.backend_row_items[BACKEND_WHISPER]._menuitem.enabled is False
    assert app.backend_row_items[BACKEND_ECHO]._menuitem.enabled is True
    assert {
        app.backend_header_item._menuitem.title,
        f"Using: {supervisor.descriptor(app._current_backend).label}",
        f"Using: {supervisor.descriptor(core.config.backend_key).label}",
        f"Using: {supervisor.descriptor(store.load()).label}",
    } == {"Using: WhisperWave"}
    assert notifications and notifications[0][1] == "Backend switched with cleanup warning"


@pytest.mark.asyncio
async def test_reconnect_targets_only_the_current_backend(tmp_path):
    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    supervisor = _LifecycleSupervisor(descriptors)
    repointed = []

    async def repoint(descriptor):
        repointed.append(descriptor.key)
        return True

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = supervisor
    app._current_backend = BACKEND_WHISPER
    app._backend_events = queue.Queue()
    app.core = SimpleNamespace(
        state=IMEState.DISCONNECTED,
        repoint_backend=repoint,
    )
    _prime_lifecycle(app)

    await app._reconnect_current_backend_async()

    assert supervisor.started == [BACKEND_WHISPER]
    assert repointed == [BACKEND_WHISPER]
    assert app._backend_events.get_nowait()[:3] == (
        "reconnected",
        BACKEND_WHISPER,
        None,
    )


@pytest.mark.asyncio
async def test_restart_targets_only_the_current_backend_and_repoints(tmp_path):
    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    supervisor = _LifecycleSupervisor(descriptors)
    repointed = []
    disconnected = []
    audio_rebuilt = []
    call_order = []

    async def disconnect():
        call_order.append("disconnect")
        disconnected.append(True)

    async def repoint(descriptor):
        repointed.append(descriptor.key)
        return True

    async def rebuild_audio_device_table():
        call_order.append("audio-rebuild")
        audio_rebuilt.append(True)
        return True

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = supervisor
    app._current_backend = BACKEND_WHISPER
    app._backend_events = queue.Queue()
    app.core = SimpleNamespace(
        state=IMEState.IDLE,
        disconnect_websocket=disconnect,
        rebuild_audio_device_table=rebuild_audio_device_table,
        repoint_backend=repoint,
    )
    _prime_lifecycle(app)

    await app._restart_service_async()

    assert disconnected == [True]
    assert audio_rebuilt == [True]
    assert call_order == ["audio-rebuild", "disconnect"]
    assert supervisor.restarted == [BACKEND_WHISPER]
    assert repointed == [BACKEND_WHISPER]
    assert app._backend_events.get_nowait()[:3] == (
        "restarted",
        BACKEND_WHISPER,
        None,
    )


@pytest.mark.asyncio
async def test_restart_continues_when_audio_rebuild_raises(tmp_path):
    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    supervisor = _LifecycleSupervisor(descriptors)
    call_order = []
    repointed = []

    async def rebuild_audio_device_table():
        call_order.append("audio-rebuild")
        raise RuntimeError("device layer unavailable")

    async def disconnect():
        call_order.append("disconnect")

    async def repoint(descriptor):
        repointed.append(descriptor.key)
        return True

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = supervisor
    app._current_backend = BACKEND_WHISPER
    app._backend_events = queue.Queue()
    app.core = SimpleNamespace(
        state=IMEState.IDLE,
        disconnect_websocket=disconnect,
        rebuild_audio_device_table=rebuild_audio_device_table,
        repoint_backend=repoint,
    )
    _prime_lifecycle(app)

    assert await app._restart_service_async() is True

    assert call_order == ["audio-rebuild", "disconnect"]
    assert supervisor.restarted == [BACKEND_WHISPER]
    assert repointed == [BACKEND_WHISPER]
    assert app._backend_events.get_nowait()[:3] == (
        "restarted",
        BACKEND_WHISPER,
        None,
    )


@pytest.mark.asyncio
async def test_switch_barrier_rejects_reconnect_restart_and_stale_lifecycle_event(tmp_path):
    """R1-M3: all lifecycle commands serialize behind one operation token."""

    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    supervisor = _LifecycleSupervisor(descriptors)
    entered = asyncio.Event()
    release = asyncio.Event()

    class BarrierCoordinator:
        async def switch(self, **_kwargs):
            app.core.state = IMEState.DISCONNECTED
            entered.set()
            await release.wait()
            app.core.config.backend_key = BACKEND_WHISPER
            app.core.config.provider = descriptors[BACKEND_WHISPER].provider
            app.core.config.model = descriptors[BACKEND_WHISPER].model
            return SimpleNamespace(
                effective_backend=BACKEND_WHISPER,
                cleanup_warning=None,
            )

    scheduled = []

    class FakeLoop:
        def call_soon_threadsafe(self, callback):
            scheduled.append(callback)

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = supervisor
    app._backend_coordinator = BarrierCoordinator()
    app._backend_events = queue.Queue()
    app._current_backend = BACKEND_ECHO
    app._selected_provider = descriptors[BACKEND_ECHO].provider
    app._selected_model = descriptors[BACKEND_ECHO].model
    app.model_labels = {
        descriptors[BACKEND_ECHO].model: "Echo model",
        descriptors[BACKEND_WHISPER].model: "Whisper model",
    }
    app.status_item = _MenuItem()
    _prime_backend_rows(app, descriptors)
    app.reconnect_item = _MenuItem()
    app.restart_item = _MenuItem()
    app.loop = FakeLoop()
    app.core = SimpleNamespace(
        state=IMEState.IDLE,
        config=SimpleNamespace(
            backend_key=BACKEND_ECHO,
            provider=descriptors[BACKEND_ECHO].provider,
            model=descriptors[BACKEND_ECHO].model,
        ),
        repoint_backend=lambda _descriptor: None,
        disconnect_websocket=lambda: None,
    )
    _prime_lifecycle(app)

    switch_task = asyncio.create_task(app._switch_backend_async(BACKEND_WHISPER))
    await asyncio.wait_for(entered.wait(), timeout=1)
    first_token = app._lifecycle_operation_generation
    app._sync_status_menu(IMEState.DISCONNECTED)
    app._sync_backend_menu(IMEState.DISCONNECTED)
    assert app.backend_header_item.title == "Using: EchoWave"
    assert _checked_lane(app) == BACKEND_ECHO
    for key, item in app.backend_row_items.items():
        assert item._menuitem.enabled is False, key
    assert app.reconnect_item._menuitem.enabled is False
    assert app.restart_item._menuitem.enabled is False

    app.reconnect(None)
    app._restart_service(None)
    assert scheduled == []
    assert await app._reconnect_current_backend_async() is False
    assert await app._restart_service_async() is False
    assert supervisor.started == []
    assert supervisor.restarted == []

    release.set()
    await asyncio.wait_for(switch_task, timeout=1)
    assert app._current_backend == BACKEND_WHISPER
    assert app._active_lifecycle_operation is None
    app._sync_provider_menu = lambda: None
    app._sync_model_menu = lambda: None
    app._drain_backend_events()
    app._sync_status_menu(IMEState.DISCONNECTED)
    app._sync_backend_menu(IMEState.DISCONNECTED)
    assert app.backend_header_item.title == "Using: WhisperWave"
    assert _checked_lane(app) == BACKEND_WHISPER
    assert app.backend_row_items[BACKEND_ECHO]._menuitem.enabled is True
    assert app.reconnect_item._menuitem.enabled is True
    assert app.restart_item._menuitem.enabled is True

    second_token = app._begin_lifecycle_operation("reconnect")
    assert second_token is not None and second_token > first_token
    app._sync_backend_menu(IMEState.DISCONNECTED)
    # Reproduce the live symptom: a committed worker updated authority, but an
    # older header and check mark are still on screen when its lifecycle event
    # goes stale.
    app.backend_header_item.title = "Using: EchoWave"
    app.backend_row_items[BACKEND_ECHO].state = 1
    app.backend_row_items[BACKEND_WHISPER].state = 0
    app._backend_events.put(("switched", BACKEND_ECHO, None, first_token))
    app._drain_backend_events()
    assert app._current_backend == BACKEND_WHISPER
    assert app.core.config.backend_key == BACKEND_WHISPER
    assert app.backend_header_item.title == "Using: WhisperWave"
    assert app.backend_header_item._menuitem.title == "Using: WhisperWave"
    assert _checked_lane(app) == BACKEND_WHISPER
    app._finish_lifecycle_operation(second_token)

    restart_token = app._begin_lifecycle_operation("restart")
    app.core.state = IMEState.PROCESSING
    assert await app._restart_service_async(restart_token) is False
    assert app._active_lifecycle_operation is None


@pytest.mark.asyncio
async def test_recording_state_accepts_streaming_deltas_after_segment_rollover():
    """300s segment rollover streams provider deltas while the IME turn is
    still RECORDING locally; dropping them lost the pre-rollover transcript
    on long recordings (2026-09-13 T31 regression)."""

    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core._connection_generation = 1
    core.state = IMEState.RECORDING
    core._active_turn_id = 31
    core._terminal_turn_ids = set()
    core._failed_turn_ids = set()
    core.transcript = "rollover前已提交文本"
    core.on_transcript = None

    # is_new baseline (rollover display reset) lands while RECORDING
    await core._handle_message(
        {
            "type": "text",
            "content": "rollover前已提交文本",
            "isNewResponse": True,
            "turn_id": 31,
        },
        connection_generation=1,
    )
    assert core.transcript == "rollover前已提交文本"

    # post-rollover streaming deltas must NOT be dropped while RECORDING
    await core._handle_message(
        {"type": "text", "content": "rollover", "turn_id": 31},
        connection_generation=1,
    )
    await core._handle_message(
        {"type": "text", "content": "后段", "turn_id": 31},
        connection_generation=1,
    )
    assert core.transcript == "rollover前已提交文本rollover后段"


    # unrelated turn deltas still ignored
    await core._handle_message(
        {"type": "text", "content": "X", "turn_id": 32},
        connection_generation=1,
    )
    assert core.transcript == "rollover前已提交文本rollover后段"


def _transcript_delivery_core():
    """Logical desktop owner with every audio/clipboard/history sink replaced."""
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core.config = SimpleNamespace(stop_sound="none")
    core._connection_generation = 1
    core.state = IMEState.RECORDING
    core._active_turn_id = 71
    core._terminal_turn_ids = set()
    core._failed_turn_ids = set()
    core._audio_drop_count = 0
    core._audio_ingestion_failed_reason = None
    core._stop_pressed_ts = None
    core._server_connected_event = asyncio.Event()
    core._start_requested = True
    core.recording_mode = RecordingMode.OPTIMIZED
    core._active_turn_provider = "openai-transcription"
    core._active_turn_model = "gpt-4o-transcribe"
    core._active_turn_backend = BACKEND_WHISPER
    core.transcript = ""
    core.on_transcript = None
    core._invalidate_recording_token = lambda: None
    core._archive_recent_turn_audio = lambda _outcome: True
    core._archive_failed_turn_audio = lambda _reason: None
    core._play_sound = lambda _name: None
    core._set_state = lambda state: setattr(core, "state", state)
    core.pastes = []
    core.completions = []
    core.on_transcript_complete = lambda *args: core.completions.append(args)

    async def input_text(text):
        core.pastes.append(text)

    async def async_stop():
        pass

    core._input_text = input_text
    core._async_stop = async_stop
    return core


@pytest.mark.asyncio
async def test_stop_preserves_streamed_turn_until_matching_terminal():
    core = _transcript_delivery_core()
    final = "A completed prefix followed by the unique ending."
    await core._handle_message({"type": "text", "content": final, "turn_id": 71})
    await core._stop_recording()
    await core._handle_message({"type": "status", "status": "idle", "turn_id": 71})

    assert core.pastes == [final]
    assert [entry[0] for entry in core.completions] == [final]


@pytest.mark.asyncio
async def test_whisper_shell_preserves_repeated_delta_content_verbatim():
    core = _transcript_delivery_core()
    phrase = "Please repeat this. "
    for _ in range(2):
        await core._handle_message(
            {"type": "text", "content": phrase, "turn_id": 71}
        )
    await core._handle_message(
        {"type": "text", "content": "A unique ending.", "turn_id": 71}
    )
    assert core.transcript == phrase * 2 + "A unique ending."


def test_event_tap_consumes_hotkey_autorepeat_without_scheduling_another_toggle(
    monkeypatch,
):
    from Quartz import kCGKeyboardEventAutorepeat

    callbacks = []
    scheduled = []
    nullified = []

    class ImmediateThread:
        def __init__(self, target, **kwargs):
            self.target = target

        def start(self):
            self.target()

    def create_tap(*args):
        callbacks.append(args[4])
        return object()

    monkeypatch.setattr(ime_menubar.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(ime_menubar, "CGEventTapCreate", create_tap)
    monkeypatch.setattr(ime_menubar, "CGEventTapEnable", lambda *_args: None)
    monkeypatch.setattr(ime_menubar, "CFMachPortCreateRunLoopSource", lambda *_args: object())
    monkeypatch.setattr(ime_menubar, "CFRunLoopGetCurrent", lambda: object())
    monkeypatch.setattr(ime_menubar, "CFRunLoopAddSource", lambda *_args: None)
    monkeypatch.setattr(ime_menubar, "CFRunLoopRun", lambda: None)
    monkeypatch.setattr(ime_menubar, "CGEventGetIntegerValueField", lambda event, field: event[field])
    monkeypatch.setattr(ime_menubar, "CGEventGetFlags", lambda event: event["flags"])
    monkeypatch.setattr(ime_menubar, "CGEventSetType", lambda event, _kind: nullified.append(event))
    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app.config = Config()
    app.loop = SimpleNamespace(call_soon_threadsafe=scheduled.append)
    app._start_event_tap()
    assert len(callbacks) == 1

    def press(repeat):
        event = {
            ime_menubar.kCGKeyboardEventKeycode: app.config.hotkey_keycode,
            kCGKeyboardEventAutorepeat: repeat,
            "flags": ime_menubar.kCGEventFlagMaskCommand,
        }
        assert callbacks[0](None, ime_menubar.kCGEventKeyDown, event, None) is event

    press(0)
    for _ in range(10):
        press(1)
    assert len(scheduled) == 1
    assert len(nullified) == 11
    press(0)  # A released-and-pressed key is the next intentional toggle.
    assert len(scheduled) == 2


@pytest.mark.parametrize("segment_count", [1, 3])
@pytest.mark.parametrize("with_deltas", [False, True])
@pytest.mark.asyncio
async def test_whisper_segments_reach_desktop_as_one_complete_paste_and_history(
    segment_count, with_deltas
):
    """Real sidecar rollover/stop output crosses the real desktop message path."""
    from starlette.websockets import WebSocketState
    from micwave_core.whisper_server import (
        TranscriptionTurnSession,
        TranscriptionTurnSessionConfig,
    )

    core = _transcript_delivery_core()

    class DesktopSocket:
        client_state = WebSocketState.CONNECTED

        def __init__(self):
            self.sent = []

        async def send_text(self, raw):
            payload = json.loads(raw)
            self.sent.append(payload)
            await core._handle_message(payload, connection_generation=1)

    class Provider:
        def __init__(self):
            self.handlers = {}
            self.sent_audio = []
            self.committed = asyncio.Event()

        def register_handler(self, name, callback):
            self.handlers[name] = callback

        def set_on_disconnect(self, callback):
            self.on_disconnect = callback

        def _is_ws_open(self):
            return True

        async def send_audio(self, chunk):
            self.sent_audio.append(chunk)
            return True

        async def commit_audio(self):
            self.committed.set()

        async def clear_audio_buffer(self):
            pass

        async def close(self):
            pass

    socket = DesktopSocket()
    frame_bytes = 4_800
    config = TranscriptionTurnSessionConfig(
        keep_provider_session=True,
        provider_session_max_turns=8,
        provider_session_max_age_sec=3300,
        provider_init_max_attempts=1,
        provider_init_retry_delay_sec=0,
        transcription_finalize_timeout_sec=5,
        transcription_ack_grace_sec=2,
        transcription_failure_rotate_threshold=2,
        default_source_sample_rate=24000,
        max_turn_audio_bytes=frame_bytes * 2,
    )
    session = TranscriptionTurnSession(socket, config)
    providers = [Provider() for _ in range(segment_count)]
    replacements = iter(providers)

    async def init_provider(**kwargs):
        session._client = next(replacements)
        session._provider_generation += 1
        session._provider_audio_bytes = 0
        session._begin_provider_epoch_locked()
        session._register_client_handlers(session._client)
        return True

    session._init_or_reuse_client = init_provider
    await session._handle_start_recording({"turn_id": 71, "input_sample_rate": 24000})
    frames = [bytes([i + 1]) * frame_bytes for i in range(segment_count * 2)]
    segment_texts = [f"Sentence for segment {i}. " for i in range(segment_count)]
    segment_texts[-1] += "A unique ending."
    cursor = 0

    async def emit_segment(index):
        provider = providers[index]
        await asyncio.wait_for(provider.committed.wait(), 1)
        item = f"delivery-item-{index}"
        await provider.handlers["input_audio_buffer.committed"]({"item_id": item})
        if with_deltas:
            await provider.handlers["conversation.item.input_audio_transcription.delta"](
                {"item_id": item, "event_id": f"delivery-event-{index}", "delta": segment_texts[index]}
            )
        await provider.handlers["conversation.item.input_audio_transcription.completed"](
            {"item_id": item, "transcript": segment_texts[index]}
        )

    for index in range(segment_count - 1):
        crossing = (index + 1) * 2
        while cursor <= crossing:
            await session._handle_audio_bytes(frames[cursor])
            cursor += 1
        await emit_segment(index)
        rollover = session._segment_rollover_task
        assert rollover is not None
        await asyncio.wait_for(asyncio.shield(rollover), 1)
        assert core.state == IMEState.RECORDING
        assert core.pastes == [] and core.completions == []

    while cursor < len(frames):
        await session._handle_audio_bytes(frames[cursor])
        cursor += 1
    stop_task = None

    async def async_stop():
        nonlocal stop_task
        stop_task = asyncio.create_task(session._handle_stop_recording({"turn_id": 71}))
        await asyncio.sleep(0)

    core._async_stop = async_stop
    await core._stop_recording()
    await emit_segment(segment_count - 1)
    await asyncio.wait_for(stop_task, 1)
    await core._handle_message({"type": "status", "status": "idle", "turn_id": 71})

    expected = "".join(segment_texts)
    assert [chunk for provider in providers for chunk in provider.sent_audio] == frames
    assert core.pastes == [expected]
    assert [entry[0] for entry in core.completions] == [expected]
    assert core.state == IMEState.IDLE
    assert len([p for p in socket.sent if p.get("status") == "idle"]) == 1
    assert session._emitted_text == expected


@pytest.mark.parametrize("stop_during_retry", [False, True])
@pytest.mark.parametrize("retry_succeeds", [False, True])
@pytest.mark.asyncio
async def test_rollover_repetition_retry_has_one_terminal_desktop_owner(
    stop_during_retry, retry_succeeds
):
    from dataclasses import replace
    from starlette.websockets import WebSocketState
    from micwave_core.whisper_server import TranscriptionTurnSession, TranscriptionTurnSessionConfig

    core = _transcript_delivery_core()
    archives, drains, notifications, restarts = [], [], [], []
    core._archive_failed_turn_audio = lambda reason: archives.append(("failed", reason))
    core._archive_recent_turn_audio = lambda reason: archives.append(("recent", reason))
    core.on_transcription_error = notifications.append
    core._start_session_task = lambda: restarts.append(True)

    async def drain_capture():
        drains.append(True)

    core._stop_and_drain_audio_capture = drain_capture

    class Socket:
        client_state = WebSocketState.CONNECTED

        def __init__(self):
            self.sent = []

        async def send_text(self, raw):
            payload = json.loads(raw)
            self.sent.append(payload)
            await core._handle_message(payload, connection_generation=1)

    class Provider:
        def __init__(self):
            self.handlers = {}
            self.sent_audio = []
            self.committed = asyncio.Event()

        def register_handler(self, name, handler):
            self.handlers[name] = handler

        def set_on_disconnect(self, callback):
            self.on_disconnect = callback

        def _is_ws_open(self):
            return True

        async def send_audio(self, frame):
            self.sent_audio.append(frame)
            return True

        async def commit_audio(self):
            self.committed.set()

        async def clear_audio_buffer(self):
            pass

        async def close(self):
            pass

    config = replace(
        TranscriptionTurnSessionConfig.from_env(),
        max_turn_audio_bytes=9_600,
        provider_init_max_attempts=1,
        provider_init_retry_delay_sec=0,
        transcription_finalize_timeout_sec=5,
    )
    socket = Socket()
    session = TranscriptionTurnSession(socket, config)
    providers = [Provider() for _ in range(3)]
    rebuild_started, release_rebuild = asyncio.Event(), asyncio.Event()
    created = []

    async def init_provider(**kwargs):
        index = len(created)
        if index == 1:
            rebuild_started.set()
            await release_rebuild.wait()
        provider = providers[index]
        created.append(provider)
        session._client = provider
        session._provider_generation += 1
        session._provider_audio_bytes = 0
        session._begin_provider_epoch_locked()
        session._register_client_handlers(provider)
        return True

    session._init_or_reuse_client = init_provider
    await session._handle_start_recording({"turn_id": 71, "input_sample_rate": 24000})
    frames = [bytes([i]) * 4_800 for i in (41, 42, 43)]
    for frame in frames:
        await session._handle_audio_bytes(frame)
    await asyncio.wait_for(providers[0].committed.wait(), 1)
    unit = "".join(f"{index:03d} " for index in range(45))
    loop = "Unique preface. " + unit * 31 + "Unique ending."

    async def complete(index, text):
        provider = providers[index]
        item = f"guard-item-{index}"
        await provider.handlers["input_audio_buffer.committed"]({"item_id": item})
        await provider.handlers["conversation.item.input_audio_transcription.delta"](
            {"item_id": item, "event_id": f"guard-delta-{index}", "delta": text}
        )
        await provider.handlers["conversation.item.input_audio_transcription.completed"](
            {"item_id": item, "transcript": text}
        )

    await complete(0, loop)
    await asyncio.wait_for(rebuild_started.wait(), 1)
    stop_task = None

    async def async_stop():
        nonlocal stop_task
        stop_task = asyncio.create_task(session._handle_stop_recording({"turn_id": 71}))
        await asyncio.sleep(0)

    core._async_stop = async_stop
    if stop_during_retry:
        await core._stop_recording()
    retry_task = session._finalize_retry_task
    release_rebuild.set()
    await asyncio.wait_for(asyncio.shield(retry_task), 1)
    await asyncio.wait_for(providers[1].committed.wait(), 1)
    assert providers[0].sent_audio == providers[1].sent_audio == frames[:2]
    assert core.pastes == [] and core.completions == []
    recovered_prefix = "The recovered prefix. "
    await complete(1, recovered_prefix if retry_succeeds else loop)

    if retry_succeeds:
        rollover = session._segment_rollover_task
        assert rollover is not None
        await asyncio.wait_for(asyncio.shield(rollover), 1)
        if not stop_during_retry:
            await core._stop_recording()
        await asyncio.wait_for(providers[2].committed.wait(), 1)
        tail = "The unique final tail."
        await complete(2, tail)
        await asyncio.wait_for(stop_task, 1)
        expected = recovered_prefix + tail
        assert core.pastes == [expected]
        assert [entry[0] for entry in core.completions] == [expected]
        assert notifications == [] and restarts == []
        assert providers[2].sent_audio == frames[2:]
        assert len(created) == 3
    else:
        await asyncio.wait_for(session._turn_done.wait(), 1)
        if stop_task is not None:
            await asyncio.wait_for(stop_task, 1)
        assert core.pastes == [] and core.completions == []
        assert 71 in core._failed_turn_ids
        assert archives == [("failed", "repeated_transcript"), ("recent", "repeated_transcript")]
        assert len(drains) == (0 if stop_during_retry else 1)
        assert len(notifications) == 1 and "重复" in notifications[0]
        assert restarts == [] and len(created) == 2
    assert core.state == IMEState.IDLE
    assert len([p for p in socket.sent if p.get("status") == "idle"]) == 1


def test_terminal_transcription_error_is_displayed_on_main_ui_queue(monkeypatch):
    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_events = queue.Queue()
    app._current_backend = BACKEND_WHISPER
    app.core = SimpleNamespace(state=IMEState.IDLE)
    app._lifecycle_event_is_current = lambda _token: True
    app._sync_backend_menu = lambda _state: None
    app._sync_provider_menu = lambda: None
    app._sync_model_menu = lambda: None
    app._sync_status_menu = lambda _state: None
    notifications = []
    monkeypatch.setattr(ime_menubar.rumps, "notification", lambda *args: notifications.append(args))
    app._on_transcription_error("转写重复，录音已保存。")
    app._drain_backend_events()
    assert notifications == [("EchoWave IME", "转写失败", "转写重复，录音已保存。")]


@pytest.mark.parametrize("drain_raises", [False, True])
@pytest.mark.parametrize("callback_raises", [False, True])
@pytest.mark.asyncio
async def test_terminal_error_remains_failed_when_capture_or_notification_raises(
    drain_raises, callback_raises
):
    core = _transcript_delivery_core()
    archives = []

    async def drain_capture():
        if drain_raises:
            raise RuntimeError("synthetic drain failure")

    def notify(message):
        assert "重复" in message
        if callback_raises:
            raise RuntimeError("synthetic notification failure")

    core._stop_and_drain_audio_capture = drain_capture
    core.on_transcription_error = notify
    core._archive_failed_turn_audio = lambda reason: archives.append(("failed", reason))
    core._archive_recent_turn_audio = lambda reason: archives.append(("recent", reason))
    core.transcript = "A known-bad provisional transcript."
    await core._handle_message(
        {"type": "error", "terminal": True, "reason": "repeated_transcript", "content": "转写重复。", "turn_id": 71}
    )
    await core._handle_message({"type": "status", "status": "idle", "turn_id": 71})
    assert core.state == IMEState.IDLE
    assert 71 in core._failed_turn_ids
    assert core.pastes == [] and core.completions == []
    assert archives == [("failed", "repeated_transcript"), ("recent", "repeated_transcript")]
