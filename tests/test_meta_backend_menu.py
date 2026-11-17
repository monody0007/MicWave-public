"""W0347: EchoWave-Meta sibling probing retirement & two-lane desktop picker contract.

EchoWave-Meta has been retired and archived. Even when a sibling EchoWave-Meta
directory exists on disk, EchoWave must register ONLY its two owned lanes
(EchoWave + WhisperWave), while gracefully downgrading any legacy persisted
"meta" / "meta_echo" selection in backend-selection.json to FALLBACK_BACKEND.
"""

import asyncio
import json
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

import ime_menubar
from backend_runtime import (
    BACKEND_ECHO,
    BACKEND_WHISPER,
    FALLBACK_BACKEND,
    REQUIRED_BACKENDS,
    RETIRED_BACKENDS,
    VALID_BACKENDS,
    BackendCoordinator,
    BackendProbeError,
    BackendSelectionStore,
    BackendSupervisor,
    BackendSwitchError,
    SelectionStoreError,
    build_default_descriptors,
    probe_backend,
)
from ime_menubar import BrainwaveIMEApp, IMEState


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "EchoWave"
    project.mkdir(parents=True, exist_ok=True)
    return project


def _make_legacy_sibling_fork(tmp_path: Path) -> Path:
    """Create a sibling EchoWave-Meta checkout to prove probing is retired."""
    fork = tmp_path / "EchoWave-Meta"
    fork.mkdir(parents=True, exist_ok=True)
    (fork / "echo_backend_entry.py").write_text("", encoding="utf-8")
    for rel in ("whisper_backend/venv/bin/python", "venv/bin/python"):
        target = fork / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("", encoding="utf-8")
    return fork


def _descriptors(tmp_path: Path, **kwargs) -> dict:
    return build_default_descriptors(
        _make_project(tmp_path),
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_python="/echo/python",
        whisper_python="/whisper/python",
        **kwargs,
    )


class _RecordingSupervisor:
    def __init__(self, descriptors, *, failing=()):
        self.descriptors = dict(descriptors)
        self.started = []
        self._failing = set(failing)

    def descriptor(self, key):
        return self.descriptors[key]

    def start(self, key, *, timeout=None):
        self.started.append(key)
        if key in self._failing:
            raise ime_menubar.BackendRuntimeError(f"{key} refused to start")


def test_only_two_owned_backend_keys_are_valid_and_meta_is_retired():
    assert VALID_BACKENDS == {BACKEND_ECHO, BACKEND_WHISPER}
    assert REQUIRED_BACKENDS == {BACKEND_ECHO, BACKEND_WHISPER}
    assert RETIRED_BACKENDS == {"meta", "meta_echo"}
    assert FALLBACK_BACKEND == BACKEND_WHISPER
    assert not hasattr(ime_menubar, "BACKEND_META")
    assert not hasattr(ime_menubar, "BACKEND_META_ECHO")


def test_default_descriptors_ignore_sibling_echowave_meta_checkout(tmp_path):
    fork = _make_legacy_sibling_fork(tmp_path)
    assert fork.is_dir()
    descriptors = _descriptors(tmp_path)
    assert set(descriptors) == {BACKEND_ECHO, BACKEND_WHISPER}
    assert descriptors[BACKEND_ECHO].port == 23456
    assert descriptors[BACKEND_WHISPER].port == 23459
    assert descriptors[BACKEND_ECHO].inherited_env_prefixes == ("BRAINWAVE_",)
    assert descriptors[BACKEND_WHISPER].inherited_env_prefixes == ("WHISPERWAVE_",)


def test_supervisor_requires_both_owned_lanes_and_rejects_retired_or_unknown_keys(
    tmp_path,
):
    two = _descriptors(tmp_path)
    BackendSupervisor(two)

    for dropped in (BACKEND_ECHO, BACKEND_WHISPER):
        broken = {k: v for k, v in two.items() if k != dropped}
        with pytest.raises(ValueError) as excinfo:
            BackendSupervisor(broken)
        assert dropped in str(excinfo.value)

    for invalid_key in ("meta", "meta_echo", "not_a_backend"):
        extra = dict(two)
        extra[invalid_key] = two[BACKEND_ECHO]
        with pytest.raises(ValueError) as excinfo:
            BackendSupervisor(extra)
        assert invalid_key in str(excinfo.value)


@pytest.mark.parametrize("backend", [BACKEND_ECHO, BACKEND_WHISPER])
def test_selection_store_round_trips_owned_backends(tmp_path, backend):
    store = BackendSelectionStore(tmp_path / "selection.json")
    store.save(backend)
    assert store.load() == backend


@pytest.mark.parametrize("retired", ["meta", "meta_echo"])
def test_selection_store_rejects_saving_retired_meta_but_loads_legacy_file_for_downgrade(
    tmp_path,
    retired,
):
    path = tmp_path / "selection.json"
    store = BackendSelectionStore(path)
    with pytest.raises(SelectionStoreError):
        store.save(retired)

    path.write_text(
        json.dumps({"version": 1, "selected_backend": retired}),
        encoding="utf-8",
    )
    assert store.load() == retired

    supervisor = _RecordingSupervisor(_descriptors(tmp_path))
    logs = []
    decision = BackendCoordinator(supervisor, store, log=logs.append).start_initial()
    assert decision.effective_backend == BACKEND_WHISPER
    assert decision.requested_backend == retired
    assert retired in decision.diagnostic
    assert supervisor.started == [BACKEND_WHISPER]
    assert any(retired in line for line in logs)


def test_legacy_meta_fallback_lands_on_echo_when_whisper_fails(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps({"version": 1, "selected_backend": "meta"}),
        encoding="utf-8",
    )
    store = BackendSelectionStore(path)
    supervisor = _RecordingSupervisor(
        _descriptors(tmp_path),
        failing=(BACKEND_WHISPER,),
    )
    decision = BackendCoordinator(supervisor, store, log=lambda _line: None).start_initial()
    assert decision.effective_backend == BACKEND_ECHO
    assert decision.requested_backend == "meta"
    assert supervisor.started == [BACKEND_WHISPER, BACKEND_ECHO]


def test_selection_store_refuses_unknown_backends(tmp_path):
    store = BackendSelectionStore(tmp_path / "selection.json")
    with pytest.raises(SelectionStoreError):
        store.save("meta_typo")
    (tmp_path / "selection.json").write_text(
        json.dumps({"version": 1, "selected_backend": "meta_typo"}),
        encoding="utf-8",
    )
    with pytest.raises(SelectionStoreError):
        store.load()


async def _always_repoints(_descriptor):
    return True


@pytest.mark.parametrize("bad", ["meta", "meta_echo", "meta_typo", "Meta", ""])
def test_coordinator_switch_rejects_retired_and_unknown_backends(tmp_path, bad):
    descriptors = _descriptors(tmp_path)
    supervisor = SimpleNamespace(
        descriptors=descriptors,
        descriptor=lambda key: descriptors[key],
    )
    coordinator = BackendCoordinator(
        supervisor,
        BackendSelectionStore(tmp_path / "selection.json"),
    )
    with pytest.raises(BackendSwitchError):
        asyncio.run(
            coordinator.switch(
                current_backend=BACKEND_ECHO,
                target_backend=bad,
                state="idle",
                repoint=_always_repoints,
            )
        )


class _NativeItemDouble:
    def __init__(self):
        self.enabled = None
        self.title = ""
        self.state = 0

    def setEnabled_(self, enabled):
        self.enabled = bool(enabled)

    def setTitle_(self, title):
        self.title = str(title)

    def setState_(self, state):
        self.state = int(state)


class _MenuItemDouble:
    def __init__(self, title="", callback=None):
        self._menuitem = _NativeItemDouble()
        self._menuitem.setTitle_(title)
        self.callback = callback

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


def _rows(current, descriptors, *, idle=True, lifecycle_active=False):
    return ime_menubar.backend_menu_rows(
        current,
        descriptors,
        idle=idle,
        lifecycle_active=lifecycle_active,
    )


def test_lane_rows_are_strictly_two_owned_lanes_in_fixed_order(tmp_path):
    _make_legacy_sibling_fork(tmp_path)
    descriptors = _descriptors(tmp_path)
    shuffled = {
        key: descriptors[key]
        for key in (BACKEND_WHISPER, BACKEND_ECHO)
    }
    assert list(shuffled) != list(ime_menubar.BACKEND_MENU_ORDER)

    for mapping in (descriptors, shuffled):
        rows = _rows(BACKEND_ECHO, mapping)
        assert [key for key, _title, _enabled, _checked in rows] == [
            BACKEND_ECHO,
            BACKEND_WHISPER,
        ]
        assert [title for _key, title, _enabled, _checked in rows] == [
            "EchoWave",
            "WhisperWave",
        ]


@pytest.mark.parametrize("current", [BACKEND_ECHO, BACKEND_WHISPER])
def test_lane_in_use_is_checked_and_non_clickable(tmp_path, current):
    rows = _rows(current, _descriptors(tmp_path))
    by_key = {key: (enabled, is_checked) for key, _t, enabled, is_checked in rows}
    assert by_key[current] == (False, True)
    for key, (enabled, is_checked) in by_key.items():
        if key != current:
            assert (enabled, is_checked) == (True, False)


def _app_with_lane_rows(tmp_path, *, current=BACKEND_ECHO, scheduled=None):
    descriptors = _descriptors(tmp_path)

    class _Supervisor:
        def __init__(self):
            self.descriptors = descriptors

        def descriptor(self, key):
            return descriptors[key]

    class _Loop:
        def call_soon_threadsafe(self, callback):
            if scheduled is not None:
                scheduled.append(callback)

    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._backend_supervisor = _Supervisor()
    app._current_backend = current
    app.backend_header_item = _MenuItemDouble()
    app.core = SimpleNamespace(state=IMEState.IDLE)
    app.loop = _Loop()
    app._sync_status_menu = lambda _state: None
    app._lifecycle_state_lock = threading.Lock()
    app._lifecycle_operation_generation = 0
    app._active_lifecycle_operation = None
    app._active_lifecycle_kind = None
    app._switch_in_progress = False
    app._pending_switch_target = None
    app._backend_events = queue.Queue()
    app.backend_row_items = {
        key: _MenuItemDouble() for key in ime_menubar.BACKEND_MENU_ORDER
    }
    for key, item in app.backend_row_items.items():
        item._backend_key = key
    return app


def test_sync_and_row_selection_operate_on_two_owned_lanes(tmp_path):
    scheduled = []
    app = _app_with_lane_rows(tmp_path, current=BACKEND_ECHO, scheduled=scheduled)
    app._sync_backend_menu(IMEState.IDLE)
    assert app.backend_header_item.title == "Using: EchoWave"
    assert [item.title for item in app.backend_row_items.values()] == [
        "EchoWave",
        "WhisperWave",
    ]

    app._backend_row_selected(app.backend_row_items[BACKEND_WHISPER])
    assert len(scheduled) == 1
    assert app.backend_header_item.title == "Switching to WhisperWave..."


def test_echo_provider_and_model_menus_disabled_on_whisper_and_enabled_on_echo():
    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app.provider_items = {"openai": _MenuItemDouble("OpenAI")}
    app.model_items = {"gpt-realtime": _MenuItemDouble("GPT Realtime")}
    app.provider_menu = _MenuItemDouble("Provider")
    app.model_menu = _MenuItemDouble("Model")
    app._selected_provider = "openai"
    app._selected_model = "gpt-realtime"

    app._current_backend = BACKEND_WHISPER
    app._sync_provider_menu()
    app._sync_model_menu()
    assert app.provider_menu._menuitem.enabled is False
    assert app.model_menu._menuitem.enabled is False

    app._current_backend = BACKEND_ECHO
    app._sync_provider_menu()
    app._sync_model_menu()
    assert app.provider_menu._menuitem.enabled is True
    assert app.model_menu._menuitem.enabled is True


_HEALTH_SERVER = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKEND = os.environ.get("WHISPERWAVE_BACKEND_KEY") or "whisper"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/health", "/api/v1/desktop-contract"):
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(
            {
                "status": "ok",
                "backend": BACKEND,
                "contract": {"name": "echowave-pcm-websocket", "version": 1},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


def _wait_for_health(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.build_opener(
                urllib.request.ProxyHandler({})
            ).open(f"http://127.0.0.1:{port}/health", timeout=0.3):
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    raise AssertionError(f"health endpoint on {port} never came up")


def test_probe_matches_backend_identity_and_rejects_mismatch(tmp_path):
    port = _free_port()
    descriptor = _descriptors(tmp_path, whisper_port=port)[BACKEND_WHISPER]
    script = tmp_path / "health_server.py"
    script.write_text(_HEALTH_SERVER, encoding="utf-8")

    for env_overlay, expectation in (
        ({}, "match"),
        ({"WHISPERWAVE_BACKEND_KEY": "echo"}, "mismatch"),
    ):
        env = {"PATH": "/usr/bin:/bin", **env_overlay}
        child = subprocess.Popen(
            [sys.executable, str(script), str(port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            _wait_for_health(port)
            if expectation == "match":
                probe = probe_backend(descriptor, timeout=2.0)
                assert probe.backend == BACKEND_WHISPER
                assert probe.contract_name == "echowave-pcm-websocket"
            else:
                with pytest.raises(BackendProbeError) as excinfo:
                    probe_backend(descriptor, timeout=2.0)
                assert "echo" in str(excinfo.value)
        finally:
            child.terminate()
            child.wait(timeout=5)
