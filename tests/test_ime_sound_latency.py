"""Immediate IME sound cues without an audio device or a running sound process."""
import asyncio
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ime_menubar
from ime_menubar import BrainwaveIMECore, Config, IMEState, RecordingMode


@pytest.fixture
def sounds(monkeypatch):
    class FakeSound:
        objects = []
        creation = "ok"
        failure = None
        play_result = True

        @classmethod
        def alloc(cls):
            return cls()

        def initWithContentsOfFile_byReference_(self, path, by_reference):
            assert by_reference is False
            if self.creation == "raise":
                raise RuntimeError("cannot decode")
            if self.creation == "nil":
                return None
            self.path = path
            self.events = []
            self.objects.append(self)
            return self

        def _record(self, event):
            self.events.append(event)
            if event == self.failure:
                raise RuntimeError("audio unavailable")

        def stop(self):
            self._record("stop")

        def play(self):
            self._record("play")
            return self.play_result

        def setVolume_(self, volume):
            self._record(("volume", volume))

    monkeypatch.setattr(ime_menubar, "NSSound", FakeSound)
    monkeypatch.setattr(ime_menubar.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(ime_menubar.time, "sleep", lambda _seconds: None)
    return FakeSound


@pytest.fixture
def core():
    core = BrainwaveIMECore.__new__(BrainwaveIMECore)
    core._server_connected_event = asyncio.Event()
    core.config = Config(
        start_sound="Tink", stop_sound="Morse",
        complete_sound="Bottle", error_sound="Basso",
    )
    return core


def test_sound_cache_restarts_without_loop_or_subprocess(core, sounds, monkeypatch):
    spawn = Mock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(ime_menubar.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(ime_menubar.subprocess, "Popen", spawn)
    monkeypatch.setattr(ime_menubar.asyncio, "ensure_future", spawn)
    core._play_sound("Tink")
    core._play_sound("Tink")
    assert len(sounds.objects) == 1
    assert core._sound_objects["Tink"] is sounds.objects[0]
    assert sounds.objects[0].path == "/System/Library/Sounds/Tink.aiff"
    assert sounds.objects[0].events == ["stop", "play", "stop", "play"]
    spawn.assert_not_called()


def test_concurrent_first_plays_share_cache(core, sounds):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(core._play_sound, ["Tink"] * 12))
    assert len(sounds.objects) == 1
    assert sounds.objects[0].events == ["stop", "play"] * 12


@pytest.mark.parametrize("name", [None, "", "none", " NoNe ", "   "])
@pytest.mark.parametrize("native", [True, False])
def test_disabled_sounds_do_nothing(core, sounds, monkeypatch, capsys, name, native):
    if not native:
        monkeypatch.setattr(ime_menubar, "NSSound", None)
    core._play_sound(name)
    assert not sounds.objects
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure", ["missing", "nil", "raise"])
def test_unavailable_sound_warns_once(core, sounds, monkeypatch, capsys, failure):
    if failure == "missing":
        monkeypatch.setattr(ime_menubar.os.path, "isfile", lambda _path: False)
    else:
        sounds.creation = failure
    core._play_sound("Missing")
    assert capsys.readouterr().out.splitlines() == [
        "[IME] Sound unavailable (Missing); skipping"
    ]
    assert core._sound_objects == {}


@pytest.mark.parametrize("failure", ["stop", "play", "false"])
def test_play_failure_is_contained(core, sounds, capsys, failure):
    sounds.failure = failure
    sounds.play_result = failure != "false"
    core._play_sound("Tink")
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 1
    assert output[0].startswith("[IME] Sound play error (Tink): ")


@pytest.mark.asyncio
async def test_no_nssound_schedules_existing_afplay(core, monkeypatch, capsys):
    monkeypatch.setattr(ime_menubar, "NSSound", None)
    core._play_sound_async = AsyncMock()
    core._play_sound("Tink")
    await asyncio.sleep(0)
    core._play_sound_async.assert_awaited_once_with("Tink")
    assert "NSSound unavailable; using afplay" in capsys.readouterr().out


def test_no_nssound_without_loop_logs_and_returns(core, monkeypatch, capsys):
    monkeypatch.setattr(ime_menubar, "NSSound", None)
    core.loop = None
    core._play_sound("Tink")
    assert "asyncio loop is not running" in capsys.readouterr().out


def test_warmup_uses_separate_muted_object(core, sounds):
    core._play_sound("Tink")
    cached = core._sound_objects["Tink"]
    core.warm_up_sounds()
    assert core._sound_objects == {"Tink": cached}
    assert len(sounds.objects) == 2
    assert cached.events == ["stop", "play"]
    assert sounds.objects[1].events == [("volume", 0.0), "play", "stop", ("volume", 1.0)]


@pytest.mark.parametrize("failure", ["play", "stop", ("volume", 0.0), ("volume", 1.0)])
def test_warmup_swallows_errors_and_restores_volume(core, sounds, failure):
    sounds.failure = failure
    core.warm_up_sounds()
    assert sounds.objects[0].events[-1] == ("volume", 1.0)
    assert not hasattr(core, "_sound_objects")


@pytest.mark.parametrize("failure", ["nil", "raise", "missing", "disabled", "no_appkit"])
def test_warmup_unavailable_is_noop(core, sounds, monkeypatch, failure):
    if failure == "missing":
        monkeypatch.setattr(ime_menubar.os.path, "isfile", lambda _path: False)
    elif failure == "disabled":
        core.config.start_sound = "none"
    elif failure == "no_appkit":
        monkeypatch.setattr(ime_menubar, "NSSound", None)
    else:
        sounds.creation = failure
    core.warm_up_sounds()
    assert not sounds.objects
    assert not hasattr(core, "_sound_objects")


def test_auto_connect_starts_daemon_warmup_without_waiting(core, monkeypatch):
    events = []
    worker = Mock()
    worker.start.side_effect = lambda: events.append("warmup_started")
    thread = Mock(return_value=worker)
    monkeypatch.setattr(ime_menubar.threading, "Thread", thread)
    app = SimpleNamespace(
        core=core,
        _connect=lambda: events.append("connect"),
        _startup_decision=SimpleNamespace(diagnostic=None),
        _start_event_tap=lambda: events.append("event_tap"),
    )
    ime_menubar.BrainwaveIMEApp._auto_connect(app)
    thread.assert_called_once_with(target=core.warm_up_sounds, daemon=True)
    worker.join.assert_not_called()
    assert events == ["warmup_started", "connect", "event_tap"]


@pytest.mark.asyncio
async def test_stop_cue_precedes_invalidation_and_async_stop_once(core, sounds):
    core.config.stop_sound = "CustomStop"
    core.state = IMEState.RECORDING
    events = []

    def invalidate():
        assert sounds.objects[0].path.endswith("/CustomStop.aiff")
        assert sounds.objects[0].events == ["stop", "play"]
        events.append("invalidate")

    async def stop():
        events.append("async_stop")
        assert core.state == IMEState.PROCESSING
        assert sounds.objects[0].events == ["stop", "play"]

    core._invalidate_recording_token = invalidate
    core._set_state = lambda state: setattr(core, "state", state)
    core._async_stop = stop
    await core._stop_recording()
    assert events == ["invalidate", "async_stop"]
    assert len(sounds.objects) == 1
    assert sounds.objects[0].events == ["stop", "play"]


@pytest.mark.asyncio
async def test_stop_sound_error_does_not_change_stop_state_machine(core, sounds):
    sounds.failure = "play"
    core.state = IMEState.RECORDING
    core._invalidate_recording_token = Mock()
    core._set_state = lambda state: setattr(core, "state", state)
    core._async_stop = AsyncMock()
    await core._stop_recording()
    core._invalidate_recording_token.assert_called_once()
    core._async_stop.assert_awaited_once()
    assert core.state == IMEState.PROCESSING


@pytest.mark.asyncio
async def test_start_cue_precedes_audio_open(core, sounds):
    core.config.start_sound = "CustomStart"
    core.state = IMEState.IDLE
    core._audio_rebuild_in_progress = False
    core._turn_id = 0
    core._ensure_audio_pipeline = Mock()
    core._allocate_recording_token = lambda: 1
    core._should_refresh_ws_before_turn = lambda: False
    core._begin_turn_spool = Mock()
    core._clear_audio_buffer = Mock()
    core._audio_drained = asyncio.Event()
    core._upload_chunk_samples = 100
    core._chunk_size_frames = 100
    core._enqueue_audio = Mock()
    core._pyaudio_executor = None
    core.loop = asyncio.get_running_loop()
    core._set_state = lambda state: setattr(core, "state", state)
    core._finish_abandoned_audio_start = Mock()

    def open_audio(*_args):
        assert sounds.objects[0].path.endswith("/CustomStart.aiff")
        assert sounds.objects[0].events == ["stop", "play"]
        assert core.state == IMEState.RECORDING
        return None

    core._open_audio_stream_with_recovery_blocking = open_audio
    await core._start_recording()
    core._finish_abandoned_audio_start.assert_called_once()
    assert len(sounds.objects) == 1


@pytest.mark.asyncio
async def test_idle_completion_uses_complete_sound_before_paste(core, sounds):
    core.config.complete_sound = "CustomComplete"
    core.state = IMEState.PROCESSING
    core._is_message_for_active_turn = lambda *_args: True
    core._active_turn_id = 1
    core._terminal_turn_ids = set()
    core._failed_turn_ids = set()
    core._audio_drop_count = 0
    core._audio_ingestion_failed_reason = None
    core._stop_pressed_ts = None
    core.recording_mode = RecordingMode.OPTIMIZED
    core._archive_recent_turn_audio = Mock()
    core._set_state = lambda state: setattr(core, "state", state)
    core.transcript = "transcript"
    core.on_transcript_complete = None

    async def paste(text):
        assert text == "transcript"
        assert sounds.objects[0].path.endswith("/CustomComplete.aiff")
        assert sounds.objects[0].events == ["stop", "play"]

    core._input_text = AsyncMock(side_effect=paste)
    await core._handle_message({"type": "status", "status": "idle", "turn_id": 1})
    core._input_text.assert_awaited_once_with("transcript")
    assert core.state == IMEState.IDLE
