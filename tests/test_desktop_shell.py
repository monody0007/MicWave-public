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


class _BackendSwitchViewDouble:
    def __init__(self):
        self.label = ""
        self.enabled = None
        self.dismiss_on_exit_armed = False

    def setBackendLabel_enabled_(self, label, enabled):
        self.label = str(label)
        self.enabled = bool(enabled)

    def setDismissOnExitArmed_(self, armed):
        self.dismiss_on_exit_armed = bool(armed)


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


def _prime_backend_presentation(app):
    app.backend_switch_item = _MenuItem()
    app.backend_switch_view = _BackendSwitchViewDouble()
    app._sync_provider_menu = lambda: None
    app._sync_model_menu = lambda: None
    app._sync_status_menu = lambda _state: None


def test_icons_hotkey_and_single_backend_switch_menu_state(tmp_path):
    assert BrainwaveIMEApp.STATE_ICONS == {
        IMEState.IDLE: "🌊",
        IMEState.RECORDING: "🔵",
        IMEState.PROCESSING: "🌀",
        IMEState.DISCONNECTED: "⚫",
    }
    assert Config().hotkey_keycode == 50

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = _DescriptorSupervisor(
        build_default_descriptors(
            tmp_path,
            app_support_dir=tmp_path / "runtime",
            log_dir=tmp_path / "logs",
            echo_python="/echo/python",
            whisper_python="/whisper/python",
        )
    )
    app._current_backend = BACKEND_ECHO
    app._switch_in_progress = False
    app.backend_switch_item = _MenuItem()
    app.backend_switch_view = _BackendSwitchViewDouble()

    app._sync_backend_menu(IMEState.IDLE)
    assert app.backend_switch_item.title == "EchoWave"
    assert app.backend_switch_item._menuitem.title == "EchoWave"
    assert app.backend_switch_view.label == "EchoWave"
    assert app.backend_switch_item.state == 0
    assert app.backend_switch_item._menuitem.state == 0
    assert app.backend_switch_item._menuitem.enabled is True
    assert app.backend_switch_view.enabled is True

    app._current_backend = BACKEND_WHISPER
    app._sync_backend_menu(IMEState.DISCONNECTED)
    assert app.backend_switch_item.title == "WhisperWave"
    assert app.backend_switch_item._menuitem.title == "WhisperWave"
    assert app.backend_switch_view.label == "WhisperWave"
    assert app.backend_switch_item.state == 0
    assert app.backend_switch_item._menuitem.state == 0
    assert app.backend_switch_item._menuitem.enabled is True

    app._sync_backend_menu(IMEState.RECORDING)
    assert app.backend_switch_item._menuitem.enabled is False
    assert app.backend_switch_view.label == "WhisperWave"
    assert app.backend_switch_view.enabled is False
    app._sync_backend_menu(IMEState.PROCESSING)
    assert app.backend_switch_item._menuitem.enabled is False
    assert app.backend_switch_view.label == "WhisperWave"
    assert app.backend_switch_view.enabled is False


def test_app_installs_backend_custom_view_as_first_menu_row_and_reuses_timer(
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
    assert native_menu.itemAtIndex_(0) is app.backend_switch_item._menuitem
    assert native_menu.itemAtIndex_(1) is app.status_item._menuitem
    assert app.backend_switch_item.title == "EchoWave"
    assert app.backend_switch_item._menuitem.title() == "EchoWave"
    assert app.backend_switch_item._menuitem.state() == 0
    assert app.backend_switch_item._menuitem.action() is None
    assert app.backend_switch_item._menuitem.view() is app.backend_switch_view
    assert app.backend_switch_view._label.stringValue() == "EchoWave"
    assert app._state_timer.start_calls == 1
    assert registered_timers == [app._state_timer]


def test_backend_custom_view_click_schedules_one_switch_and_never_cancels_tracking(
    tmp_path,
):
    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_python="/echo/python",
        whisper_python="/whisper/python",
    )
    scheduled = []
    cancellations = []

    class FakeLoop:
        def call_soon_threadsafe(self, callback):
            scheduled.append(callback)

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = _DescriptorSupervisor(descriptors)
    app._current_backend = BACKEND_ECHO
    app.backend_switch_item = ime_menubar.rumps.MenuItem("EchoWave", callback=None)
    app.core = SimpleNamespace(state=IMEState.IDLE)
    app.loop = FakeLoop()
    app._sync_status_menu = lambda _state: None
    app._cancel_backend_menu_tracking = lambda: cancellations.append(True)
    _prime_lifecycle(app)
    app._attach_backend_switch_view()
    app._sync_backend_menu(IMEState.IDLE)

    app.backend_switch_view.mouseUp_(None)
    app.backend_switch_view.mouseUp_(None)

    assert len(scheduled) == 1
    assert app._active_lifecycle_kind == "switch"
    assert app._current_backend == BACKEND_ECHO
    assert app.backend_switch_item.title == "EchoWave"
    assert app.backend_switch_view._label.stringValue() == "EchoWave"
    assert app.backend_switch_view._interaction_enabled is False
    assert cancellations == []


def test_backend_custom_view_mouse_exit_dismisses_once_only_after_arm():
    class FakeNativeMenu:
        def __init__(self):
            self.cancel_calls = 0

        def cancelTrackingWithoutAnimation(self):
            self.cancel_calls += 1

    native_menu = FakeNativeMenu()
    native_item = SimpleNamespace(menu=lambda: native_menu)
    owner = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    owner.backend_switch_item = SimpleNamespace(_menuitem=native_item)
    owner._backend_switch_selected = lambda _sender: None
    view = ime_menubar.BackendSwitchMenuView.alloc().initWithFrame_(
        ((0.0, 0.0), (240.0, 24.0))
    )
    view.configureWithOwner_(owner)

    view.mouseExited_(None)
    assert native_menu.cancel_calls == 0

    view.setDismissOnExitArmed_(True)
    view.mouseExited_(None)
    view.mouseExited_(None)
    assert native_menu.cancel_calls == 1
    assert view._dismiss_on_exit_armed is False

    view.setDismissOnExitArmed_(False)
    view.mouseExited_(None)
    assert native_menu.cancel_calls == 1

    view.setDismissOnExitArmed_(True)
    view.viewDidMoveToWindow()
    assert view._dismiss_on_exit_armed is False


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
    assert app.backend_switch_item.title == "EchoWave"
    assert app.backend_switch_item._menuitem.title == "EchoWave"
    assert app.backend_switch_view.label == "EchoWave"
    assert app.backend_switch_item._menuitem.enabled is True
    assert app.backend_switch_view.enabled is True
    assert app.backend_switch_view.dismiss_on_exit_armed is False
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
    assert app.backend_switch_item.title == "WhisperWave"
    assert app.backend_switch_item._menuitem.title == "WhisperWave"
    assert app.backend_switch_view.label == "WhisperWave"
    assert app.backend_switch_item.state == 0
    assert app.backend_switch_item._menuitem.enabled is True
    assert app.backend_switch_view.enabled is True
    assert app.backend_switch_view.dismiss_on_exit_armed is True
    assert {
        app.backend_switch_view.label,
        app.backend_switch_item._menuitem.title,
        supervisor.descriptor(app._current_backend).label,
        supervisor.descriptor(core.config.backend_key).label,
        supervisor.descriptor(store.load()).label,
    } == {"WhisperWave"}
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
    app.backend_switch_item = _MenuItem()
    app.backend_switch_view = _BackendSwitchViewDouble()
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
    assert app.backend_switch_item.title == "EchoWave"
    assert app.backend_switch_view.label == "EchoWave"
    assert app.backend_switch_item._menuitem.enabled is False
    assert app.backend_switch_view.enabled is False
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
    assert app.backend_switch_item.title == "WhisperWave"
    assert app.backend_switch_view.label == "WhisperWave"
    assert app.backend_switch_view.dismiss_on_exit_armed is True
    assert app.backend_switch_item._menuitem.enabled is True
    assert app.reconnect_item._menuitem.enabled is True
    assert app.restart_item._menuitem.enabled is True

    second_token = app._begin_lifecycle_operation("reconnect")
    assert second_token is not None and second_token > first_token
    app._set_backend_switch_dismiss_armed(False)
    app._sync_backend_menu(IMEState.DISCONNECTED)
    # Reproduce the live symptom: a committed worker updated authority, but an
    # older native label is still on screen when its lifecycle event goes stale.
    app.backend_switch_item.title = "EchoWave"
    app.backend_switch_view.label = "EchoWave"
    app._backend_events.put(("switched", BACKEND_ECHO, None, first_token))
    app._drain_backend_events()
    assert app._current_backend == BACKEND_WHISPER
    assert app.core.config.backend_key == BACKEND_WHISPER
    assert app.backend_switch_item.title == "WhisperWave"
    assert app.backend_switch_item._menuitem.title == "WhisperWave"
    assert app.backend_switch_view.label == "WhisperWave"
    assert app.backend_switch_view.dismiss_on_exit_armed is False
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
