import json
import re
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from backend_runtime import (
    BACKEND_ECHO,
    BACKEND_START_TIMEOUT_ENV,
    BACKEND_WHISPER,
    DEFAULT_BACKEND_START_TIMEOUT_SEC,
    WIRE_CONTRACT_NAME,
    WIRE_CONTRACT_VERSION,
    BackendCoordinator,
    BackendDescriptor,
    BackendProbeError,
    BackendRuntimeError,
    BackendSelectionStore,
    BackendSwitchError,
    HealthProbe,
    SelectionStoreError,
    StartupDecision,
    SwitchNotAllowed,
    BackendSupervisor,
    build_default_descriptors,
    probe_backend,
    resolve_backend_start_timeout,
)


def _descriptor(tmp_path: Path, key: str, port: int) -> BackendDescriptor:
    return BackendDescriptor(
        key=key,
        label="EchoWave" if key == BACKEND_ECHO else "WhisperWave",
        host="127.0.0.1",
        port=port,
        command=("/usr/bin/true",),
        cwd=str(tmp_path),
        pid_file=str(tmp_path / f"{key}.pid"),
        log_file=str(tmp_path / f"{key}.log"),
        provider="test",
        model=f"{key}-model",
    )


def _probe(descriptor: BackendDescriptor) -> HealthProbe:
    return HealthProbe(
        backend=descriptor.key,
        contract_name=WIRE_CONTRACT_NAME,
        contract_version=WIRE_CONTRACT_VERSION,
        endpoint=descriptor.websocket_uri,
        payload={"status": "ok"},
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_contract_server(
    tmp_path: Path,
    *,
    bind_delay_sec: float = 0.0,
    bind_delay_file: Path | None = None,
) -> Path:
    """Contract server script; ``bind_delay_sec`` models a slow import phase.

    ``bind_delay_file``, when given and present at spawn time, overrides the
    delay so a test can run the same script slow first and fast afterwards.
    """

    delay_file_repr = repr(str(bind_delay_file)) if bind_delay_file is not None else "None"
    script = tmp_path / "contract_server.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            BACKEND = sys.argv[2]
            DELAY = {float(bind_delay_sec)!r}
            DELAY_FILE = {delay_file_repr}
            if DELAY_FILE:
                try:
                    with open(DELAY_FILE, encoding="ascii") as handle:
                        DELAY = float(handle.read().strip() or DELAY)
                except FileNotFoundError:
                    pass
            time.sleep(DELAY)
            PAYLOAD = {{
                "status": "ok",
                "backend": BACKEND,
                "contract": {{
                    "name": {WIRE_CONTRACT_NAME!r},
                    "version": {WIRE_CONTRACT_VERSION},
                }},
            }}

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path not in ("/health", "/api/v1/desktop-contract"):
                        self.send_response(404)
                        self.end_headers()
                        return
                    body = json.dumps(PAYLOAD).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *_args):
                    pass

            ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
            """
        ),
        encoding="utf-8",
    )
    return script


def _process_descriptor(
    tmp_path: Path,
    key: str,
    port: int,
    script: Path,
) -> BackendDescriptor:
    return BackendDescriptor(
        key=key,
        label="EchoWave" if key == BACKEND_ECHO else "WhisperWave",
        host="127.0.0.1",
        port=port,
        command=(sys.executable, str(script), str(port), key),
        cwd=str(tmp_path),
        pid_file=str(tmp_path / f"{key}.pid"),
        log_file=str(tmp_path / f"{key}.log"),
        provider="test",
        model=f"{key}-model",
        require_remote_contract=True,
    )


def _wait_for_contract(descriptor: BackendDescriptor, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            from backend_runtime import probe_backend

            probe_backend(descriptor, timeout=0.1)
            return
        except BackendProbeError as exc:
            last_error = exc
            time.sleep(0.02)
    raise AssertionError(f"contract server did not start: {last_error}")


class FakeSupervisor:
    def __init__(self, tmp_path: Path):
        self.descriptors = {
            BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, 31001),
            BACKEND_WHISPER: _descriptor(tmp_path, BACKEND_WHISPER, 31002),
        }
        self.started = []
        self.stopped = []
        self.start_timeouts = []
        self.stop_timeouts = []
        self.fail = set()

    def descriptor(self, key):
        return self.descriptors[key]

    def start(self, key, *, timeout=None):
        self.started.append(key)
        self.start_timeouts.append(timeout)
        if key in self.fail:
            raise BackendProbeError(f"{key} unavailable")
        return _probe(self.descriptors[key])

    def stop(self, key, *, timeout=5.0):
        self.stopped.append(key)
        self.stop_timeouts.append(timeout)


def test_selection_store_default_atomic_persist_and_restart(tmp_path):
    path = tmp_path / "selection.json"
    first = BackendSelectionStore(path)
    assert first.load() == BACKEND_ECHO
    first.save(BACKEND_WHISPER)
    assert not list(tmp_path.glob("*.tmp"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"selected_backend": BACKEND_WHISPER, "version": 1}
    assert BackendSelectionStore(path).load() == BACKEND_WHISPER


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "[]",
        '{"version": 999, "selected_backend": "echo"}',
        '{"version": 1, "selected_backend": "mystery"}',
    ],
)
def test_selection_store_rejects_corrupt_or_unknown_values(tmp_path, payload):
    path = tmp_path / "selection.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(SelectionStoreError):
        BackendSelectionStore(path).load()


def test_cold_start_whisper_failure_is_explicit_and_does_not_rewrite_request(tmp_path):
    path = tmp_path / "selection.json"
    store = BackendSelectionStore(path)
    store.save(BACKEND_WHISPER)
    supervisor = FakeSupervisor(tmp_path)
    supervisor.fail.add(BACKEND_WHISPER)
    decision = BackendCoordinator(supervisor, store).start_initial()
    assert decision == StartupDecision(
        requested_backend=BACKEND_WHISPER,
        effective_backend=BACKEND_ECHO,
        diagnostic=decision.diagnostic,
    )
    assert "explicit fallback" in decision.diagnostic
    assert supervisor.started == [BACKEND_WHISPER, BACKEND_ECHO]
    assert store.load() == BACKEND_WHISPER


def test_cold_start_corrupt_selection_surfaces_diagnostic_and_uses_echo(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text("{broken", encoding="utf-8")
    store = BackendSelectionStore(path)
    supervisor = FakeSupervisor(tmp_path)
    decision = BackendCoordinator(supervisor, store).start_initial()
    assert decision.requested_backend == "invalid"
    assert decision.effective_backend == BACKEND_ECHO
    assert "invalid backend selection JSON" in decision.diagnostic
    assert supervisor.started == [BACKEND_ECHO]


@pytest.mark.asyncio
async def test_switch_transaction_idle_success_orders_repoint_persist_stop(tmp_path):
    events = []
    supervisor = FakeSupervisor(tmp_path)
    store = BackendSelectionStore(tmp_path / "selection.json")

    original_start = supervisor.start
    original_stop = supervisor.stop
    original_save = store.save

    def start(key, *, timeout=10.0):
        events.append(("start+probe", key))
        return original_start(key, timeout=timeout)

    def stop(key, *, timeout=5.0):
        events.append(("stop", key))
        return original_stop(key, timeout=timeout)

    def save(key):
        events.append(("persist", key))
        return original_save(key)

    supervisor.start = start
    supervisor.stop = stop
    store.save = save

    async def repoint(descriptor):
        events.append(("repoint", descriptor.key))
        return True

    result = await BackendCoordinator(supervisor, store).switch(
        current_backend=BACKEND_ECHO,
        target_backend=BACKEND_WHISPER,
        state="idle",
        repoint=repoint,
    )
    assert result.effective_backend == BACKEND_WHISPER
    assert events == [
        ("start+probe", BACKEND_WHISPER),
        ("repoint", BACKEND_WHISPER),
        ("persist", BACKEND_WHISPER),
        ("stop", BACKEND_ECHO),
    ]
    assert store.load() == BACKEND_WHISPER


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["recording", "processing"])
async def test_switch_rejected_during_active_turn(tmp_path, state):
    supervisor = FakeSupervisor(tmp_path)
    store = BackendSelectionStore(tmp_path / "selection.json")

    async def repoint(_descriptor):
        raise AssertionError("must not repoint")

    with pytest.raises(SwitchNotAllowed):
        await BackendCoordinator(supervisor, store).switch(
            current_backend=BACKEND_ECHO,
            target_backend=BACKEND_WHISPER,
            state=state,
            repoint=repoint,
        )
    assert supervisor.started == []


@pytest.mark.asyncio
async def test_switch_target_health_failure_keeps_old_selection_and_endpoint(tmp_path):
    supervisor = FakeSupervisor(tmp_path)
    supervisor.fail.add(BACKEND_WHISPER)
    store = BackendSelectionStore(tmp_path / "selection.json")
    store.save(BACKEND_ECHO)
    repoints = []

    async def repoint(descriptor):
        repoints.append(descriptor.key)
        return True

    with pytest.raises(BackendSwitchError, match="keeping EchoWave"):
        await BackendCoordinator(supervisor, store).switch(
            current_backend=BACKEND_ECHO,
            target_backend=BACKEND_WHISPER,
            state="idle",
            repoint=repoint,
        )
    assert repoints == []
    assert supervisor.stopped == []
    assert store.load() == BACKEND_ECHO


@pytest.mark.asyncio
async def test_switch_repoint_failure_stops_only_target_and_keeps_persisted_old(tmp_path):
    supervisor = FakeSupervisor(tmp_path)
    store = BackendSelectionStore(tmp_path / "selection.json")
    store.save(BACKEND_ECHO)

    async def repoint(_descriptor):
        return False

    with pytest.raises(BackendSwitchError, match="could not connect"):
        await BackendCoordinator(supervisor, store).switch(
            current_backend=BACKEND_ECHO,
            target_backend=BACKEND_WHISPER,
            state="disconnected",
            repoint=repoint,
        )
    assert supervisor.stopped == [BACKEND_WHISPER]
    assert store.load() == BACKEND_ECHO


def test_supervisor_child_cleanup_is_exact_and_does_not_touch_other_backend(tmp_path):
    descriptors = {
        BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, 31001),
        BACKEND_WHISPER: _descriptor(tmp_path, BACKEND_WHISPER, 31002),
    }
    supervisor = BackendSupervisor(descriptors)

    class FakeChild:
        def __init__(self, pid):
            self.pid = pid
            self.returncode = None
            self.terminated = 0
            self.killed = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated += 1

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def kill(self):
            self.killed += 1

    class FakeLog:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    echo_child = FakeChild(61001)
    whisper_child = FakeChild(61002)
    echo_log = FakeLog()
    whisper_log = FakeLog()
    supervisor._children = {
        BACKEND_ECHO: echo_child,
        BACKEND_WHISPER: whisper_child,
    }
    supervisor._log_handles = {
        BACKEND_ECHO: echo_log,
        BACKEND_WHISPER: whisper_log,
    }

    supervisor.stop(BACKEND_ECHO)
    assert echo_child.terminated == 1
    assert echo_child.killed == 0
    assert echo_log.closed is True
    assert whisper_child.terminated == 0
    assert whisper_log.closed is False


def test_supervisor_fails_closed_when_unknown_listener_already_serves_valid_contract(tmp_path):
    """R1-S1: a valid contract on the port is not proof of child ownership."""

    port = _free_port()
    listener_script = _write_contract_server(tmp_path)
    listener_descriptor = _process_descriptor(
        tmp_path,
        BACKEND_ECHO,
        port,
        listener_script,
    )
    listener = subprocess.Popen(
        list(listener_descriptor.command),
        cwd=listener_descriptor.cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    spawned = []

    def candidate_that_would_exit(*args, **kwargs):
        spawned.append((args, kwargs))
        return subprocess.Popen(["/bin/sh", "-c", "exit 1"])

    candidate = BackendDescriptor(
        **{
            **listener_descriptor.__dict__,
            "command": ("/bin/sh", "-c", "exit 1"),
        }
    )
    other = _descriptor(tmp_path, BACKEND_WHISPER, _free_port())
    supervisor = BackendSupervisor(
        {BACKEND_ECHO: candidate, BACKEND_WHISPER: other},
        popen_factory=candidate_that_would_exit,
    )
    try:
        _wait_for_contract(listener_descriptor)
        with pytest.raises(BackendRuntimeError, match="port .* already in use"):
            supervisor.start(BACKEND_ECHO, timeout=0.5)
        assert spawned == []
        assert not Path(candidate.pid_file).exists()
        assert listener.poll() is None
    finally:
        listener.terminate()
        listener.wait(timeout=3)


def test_supervisor_discards_dead_pid_record_before_starting_real_child(tmp_path):
    """R1-S1: a stale numeric PID record cannot block or own the new child."""

    script = _write_contract_server(tmp_path)
    echo = _process_descriptor(tmp_path, BACKEND_ECHO, _free_port(), script)
    other = _descriptor(tmp_path, BACKEND_WHISPER, _free_port())
    Path(echo.pid_file).write_text("999999999\n", encoding="ascii")
    supervisor = BackendSupervisor({BACKEND_ECHO: echo, BACKEND_WHISPER: other})
    try:
        probe = supervisor.start(BACKEND_ECHO, timeout=3.0)
        child = supervisor._children[BACKEND_ECHO]
        assert probe.backend == BACKEND_ECHO
        assert int(Path(echo.pid_file).read_text(encoding="ascii")) == child.pid
        assert child.poll() is None
    finally:
        supervisor.stop(BACKEND_ECHO)
    assert not Path(echo.pid_file).exists()


def test_supervisor_reaps_exited_candidate_without_committing_dead_pid(tmp_path):
    """R1-S1: startup failure never turns a dead candidate into ownership."""

    echo = BackendDescriptor(
        key=BACKEND_ECHO,
        label="EchoWave",
        host="127.0.0.1",
        port=_free_port(),
        command=("/bin/sh", "-c", "exit 1"),
        cwd=str(tmp_path),
        pid_file=str(tmp_path / "echo.pid"),
        log_file=str(tmp_path / "echo.log"),
        provider="test",
        model="test",
    )
    other = _descriptor(tmp_path, BACKEND_WHISPER, _free_port())
    supervisor = BackendSupervisor({BACKEND_ECHO: echo, BACKEND_WHISPER: other})

    with pytest.raises(BackendRuntimeError, match="exited during startup"):
        supervisor.start(BACKEND_ECHO, timeout=0.5)

    assert BACKEND_ECHO not in supervisor._children
    assert not Path(echo.pid_file).exists()
    BackendSupervisor._assert_port_available(echo)


def test_supervisor_reaps_exact_pid_owned_orphan_before_starting_replacement(tmp_path):
    """R1-S1: a shell-crash orphan is recoverable only through exact PID+argv."""

    script = _write_contract_server(tmp_path)
    echo = _process_descriptor(tmp_path, BACKEND_ECHO, _free_port(), script)
    other = _descriptor(tmp_path, BACKEND_WHISPER, _free_port())
    orphan = subprocess.Popen(
        list(echo.command),
        cwd=echo.cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    Path(echo.pid_file).write_text(f"{orphan.pid}\n", encoding="ascii")
    supervisor = BackendSupervisor({BACKEND_ECHO: echo, BACKEND_WHISPER: other})
    try:
        _wait_for_contract(echo)
        probe = supervisor.start(BACKEND_ECHO, timeout=3.0)
        replacement = supervisor._children[BACKEND_ECHO]
        orphan.wait(timeout=3)
        assert probe.backend == BACKEND_ECHO
        assert replacement.pid != orphan.pid
        assert replacement.poll() is None
        assert int(Path(echo.pid_file).read_text(encoding="ascii")) == replacement.pid
    finally:
        try:
            supervisor.stop(BACKEND_ECHO)
        finally:
            if orphan.poll() is None:
                orphan.terminate()
                orphan.wait(timeout=3)


@pytest.mark.asyncio
async def test_stop_old_failure_commits_target_with_cleanup_warning(tmp_path):
    """R1-S3: post-persist cleanup failure is degraded commit, not rollback."""

    supervisor = FakeSupervisor(tmp_path)
    store = BackendSelectionStore(tmp_path / "selection.json")
    store.save(BACKEND_ECHO)

    def stop(key, *, timeout=5.0):
        supervisor.stopped.append(key)
        if key == BACKEND_ECHO:
            raise OSError("injected old-child stop failure")

    supervisor.stop = stop

    async def repoint(_descriptor):
        return True

    result = await BackendCoordinator(supervisor, store).switch(
        current_backend=BACKEND_ECHO,
        target_backend=BACKEND_WHISPER,
        state="idle",
        repoint=repoint,
    )
    assert result.effective_backend == BACKEND_WHISPER
    assert "injected old-child stop failure" in result.cleanup_warning
    assert store.load() == BACKEND_WHISPER
    assert supervisor.stopped == [BACKEND_ECHO]


def test_backend_children_receive_descriptor_specific_minimal_environments(
    tmp_path,
    monkeypatch,
):
    """R1-M2: Echo and Whisper receive separate, testable env contracts."""

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "test-ca.pem"))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:18080")
    monkeypatch.setenv("OPENAI_API_KEY", "non-secret-test-sentinel")
    monkeypatch.setenv("BRAINWAVE_TEST_SETTING", "echo-only")
    monkeypatch.setenv("WHISPERWAVE_TEST_SETTING", "whisper-only")
    monkeypatch.setenv("GOOGLE_API_KEY", "non-secret-test-google")
    monkeypatch.setenv("OPENAI_REALTIME_MODEL", "echo-model")
    monkeypatch.setenv("PYTHONPATH", "/must/not/inherit")

    descriptors = build_default_descriptors(
        tmp_path,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
        echo_port=_free_port(),
        whisper_port=_free_port(),
        echo_python="/usr/bin/true",
        whisper_python="/usr/bin/true",
    )
    captured = []

    class FakeChild:
        _next_pid = 70000

        def __init__(self):
            self.pid = type(self)._next_pid
            type(self)._next_pid += 1
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    def popen(command, **kwargs):
        captured.append((tuple(command), dict(kwargs["env"])))
        return FakeChild()

    supervisor = BackendSupervisor(
        descriptors,
        popen_factory=popen,
        probe_func=lambda descriptor, _timeout: _probe(descriptor),
    )
    supervisor.start(BACKEND_WHISPER)
    supervisor.stop(BACKEND_WHISPER)
    supervisor.start(BACKEND_ECHO)
    supervisor.stop(BACKEND_ECHO)

    whisper_env = captured[0][1]
    echo_env = captured[1][1]
    assert whisper_env["HOME"] == str(tmp_path / "home")
    assert whisper_env["PATH"] == "/usr/bin:/bin"
    assert whisper_env["LANG"] == "en_US.UTF-8"
    assert whisper_env["TMPDIR"] == str(tmp_path / "tmp")
    assert whisper_env["SSL_CERT_FILE"] == str(tmp_path / "test-ca.pem")
    assert whisper_env["HTTPS_PROXY"] == "http://127.0.0.1:18080"
    assert whisper_env["OPENAI_API_KEY"] == "non-secret-test-sentinel"
    assert whisper_env["WHISPERWAVE_TEST_SETTING"] == "whisper-only"
    assert whisper_env["WHISPERWAVE_PORT"] == str(descriptors[BACKEND_WHISPER].port)
    assert "BRAINWAVE_TEST_SETTING" not in whisper_env
    assert "GOOGLE_API_KEY" not in whisper_env
    assert "OPENAI_REALTIME_MODEL" not in whisper_env
    assert "PYTHONPATH" not in whisper_env

    assert echo_env["OPENAI_API_KEY"] == "non-secret-test-sentinel"
    assert echo_env["BRAINWAVE_TEST_SETTING"] == "echo-only"
    assert echo_env["GOOGLE_API_KEY"] == "non-secret-test-google"
    assert echo_env["OPENAI_REALTIME_MODEL"] == "echo-model"
    assert "WHISPERWAVE_TEST_SETTING" not in echo_env
    assert "PYTHONPATH" not in echo_env


# ── task 0607: startup gate distinguishes alive-but-slow from dead/hung ──────


class _NeverListeningChild:
    """Fake Popen that stays alive and never serves; models a slow import."""

    _next_pid = 71000

    def __init__(self):
        self.pid = type(self)._next_pid
        type(self)._next_pid += 1
        self.returncode = None
        self.terminated = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def _never_listening_supervisor(tmp_path: Path, *, log=None, progress_interval=5.0):
    descriptors = {
        BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, _free_port()),
        BACKEND_WHISPER: _descriptor(tmp_path, BACKEND_WHISPER, _free_port()),
    }
    children = []

    def popen(_command, **_kwargs):
        child = _NeverListeningChild()
        children.append(child)
        return child

    def probe(descriptor, _timeout):
        raise BackendProbeError(f"{descriptor.label} health probe failed: Connection refused")

    kwargs = {"popen_factory": popen, "probe_func": probe, "progress_interval": progress_interval}
    if log is not None:
        kwargs["log"] = log
    return BackendSupervisor(descriptors, **kwargs), children


def test_supervisor_waits_for_alive_child_that_binds_late(tmp_path):
    """S2/I1 (a): a live child that binds after 2s is adopted, never killed as dead."""

    script = _write_contract_server(tmp_path, bind_delay_sec=2.0)
    echo = _process_descriptor(tmp_path, BACKEND_ECHO, _free_port(), script)
    other = _descriptor(tmp_path, BACKEND_WHISPER, _free_port())
    supervisor = BackendSupervisor({BACKEND_ECHO: echo, BACKEND_WHISPER: other})
    started = time.monotonic()
    try:
        probe = supervisor.start(BACKEND_ECHO, timeout=6.0)
        elapsed = time.monotonic() - started
        child = supervisor._children[BACKEND_ECHO]
        assert probe.backend == BACKEND_ECHO
        assert elapsed >= 1.9
        assert child.poll() is None
        assert int(Path(echo.pid_file).read_text(encoding="ascii")) == child.pid
    finally:
        supervisor.stop(BACKEND_ECHO)
    assert not Path(echo.pid_file).exists()


def test_supervisor_reports_hung_child_with_elapsed_pid_and_last_probe(tmp_path):
    """S2/I2 (a): at the limit an alive child is stopped as hung, with diagnostics."""

    script = _write_contract_server(tmp_path, bind_delay_sec=2.0)
    echo = _process_descriptor(tmp_path, BACKEND_ECHO, _free_port(), script)
    other = _descriptor(tmp_path, BACKEND_WHISPER, _free_port())
    spawned = []

    def popen(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        spawned.append(child)
        return child

    supervisor = BackendSupervisor(
        {BACKEND_ECHO: echo, BACKEND_WHISPER: other},
        popen_factory=popen,
    )
    started = time.monotonic()
    with pytest.raises(BackendProbeError) as excinfo:
        supervisor.start(BACKEND_ECHO, timeout=0.5)
    elapsed = time.monotonic() - started
    message = str(excinfo.value)
    assert len(spawned) == 1
    assert re.search(r"within \d+\.\ds", message), message
    assert f"pid {spawned[0].pid}" in message
    assert "Connection refused" in message
    assert BACKEND_START_TIMEOUT_ENV in message
    assert elapsed < 4.0
    assert spawned[0].poll() is not None
    assert BACKEND_ECHO not in supervisor._children
    assert not Path(echo.pid_file).exists()


def test_supervisor_fails_fast_on_exited_child_without_waiting_for_limit(tmp_path):
    """S2 (b): child exit is reported at once with its code, even with a 10s limit."""

    echo = BackendDescriptor(
        key=BACKEND_ECHO,
        label="EchoWave",
        host="127.0.0.1",
        port=_free_port(),
        command=("/bin/sh", "-c", "exit 3"),
        cwd=str(tmp_path),
        pid_file=str(tmp_path / "echo.pid"),
        log_file=str(tmp_path / "echo.log"),
        provider="test",
        model="test",
    )
    other = _descriptor(tmp_path, BACKEND_WHISPER, _free_port())
    supervisor = BackendSupervisor({BACKEND_ECHO: echo, BACKEND_WHISPER: other})
    started = time.monotonic()
    with pytest.raises(BackendRuntimeError) as excinfo:
        supervisor.start(BACKEND_ECHO, timeout=10.0)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert not isinstance(excinfo.value, BackendProbeError)
    assert "exited during startup with code 3" in str(excinfo.value)
    assert BACKEND_ECHO not in supervisor._children
    assert not Path(echo.pid_file).exists()


@pytest.mark.parametrize("raw", ["abc", "0", "-3", "nan", "inf"])
def test_start_timeout_env_invalid_values_fall_back_with_one_warning(monkeypatch, raw):
    """S1 (c): a bad knob warns once and uses the default; it never raises."""

    logs = []
    monkeypatch.setenv(BACKEND_START_TIMEOUT_ENV, raw)
    assert resolve_backend_start_timeout(log=logs.append) == DEFAULT_BACKEND_START_TIMEOUT_SEC
    assert DEFAULT_BACKEND_START_TIMEOUT_SEC == 90.0
    assert len(logs) == 1
    assert BACKEND_START_TIMEOUT_ENV in logs[0]
    assert raw in logs[0]


def test_start_timeout_env_valid_blank_and_unset(monkeypatch):
    """S1 (c): ``2.5`` parses; blank or unset means the default without noise."""

    logs = []
    monkeypatch.setenv(BACKEND_START_TIMEOUT_ENV, "2.5")
    assert resolve_backend_start_timeout(log=logs.append) == 2.5
    monkeypatch.setenv(BACKEND_START_TIMEOUT_ENV, "")
    assert resolve_backend_start_timeout(log=logs.append) == DEFAULT_BACKEND_START_TIMEOUT_SEC
    monkeypatch.delenv(BACKEND_START_TIMEOUT_ENV)
    assert resolve_backend_start_timeout(log=logs.append) == DEFAULT_BACKEND_START_TIMEOUT_SEC
    assert resolve_backend_start_timeout("7", log=logs.append) == 7.0
    assert logs == []


def test_supervisor_start_without_timeout_reads_env_limit(tmp_path, monkeypatch):
    """S1/S4 (c): ``timeout=None`` resolves the env limit inside start()."""

    monkeypatch.setenv(BACKEND_START_TIMEOUT_ENV, "0.3")
    logs = []
    supervisor, children = _never_listening_supervisor(tmp_path, log=logs.append)
    started = time.monotonic()
    with pytest.raises(BackendProbeError, match=r"within 0\.\ds"):
        supervisor.start(BACKEND_ECHO)
    elapsed = time.monotonic() - started
    assert 0.3 <= elapsed < 2.0
    assert children[0].terminated == 1
    assert not any("invalid" in line for line in logs)


def test_supervisor_explicit_timeout_ignores_env_entirely(tmp_path, monkeypatch):
    """S1 (c): an explicit value is used as given and a bad env value is not even read."""

    monkeypatch.setenv(BACKEND_START_TIMEOUT_ENV, "abc")
    logs = []
    supervisor, children = _never_listening_supervisor(tmp_path, log=logs.append)
    started = time.monotonic()
    with pytest.raises(BackendProbeError):
        supervisor.start(BACKEND_ECHO, timeout=0.2)
    elapsed = time.monotonic() - started
    assert 0.2 <= elapsed < 2.0
    assert not any("invalid" in line for line in logs)
    assert children[0].terminated == 1


def test_probe_backend_ignores_proxy_environment(tmp_path, monkeypatch):
    """S5/I3 (d): loopback probes go direct even when HTTP(S)_PROXY points at a dead proxy."""

    port = _free_port()
    script = _write_contract_server(tmp_path)
    descriptor = _process_descriptor(tmp_path, BACKEND_ECHO, port, script)
    listener = subprocess.Popen(
        list(descriptor.command),
        cwd=descriptor.cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_contract(descriptor)
        dead_proxy = f"http://127.0.0.1:{_free_port()}"
        for key in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(key, raising=False)
        for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
            monkeypatch.setenv(key, dead_proxy)

        # Control: a default opener built under this env honours the proxy and
        # cannot reach the listener, so the assertion below is meaningful.
        with pytest.raises(urllib.error.URLError):
            urllib.request.build_opener().open(descriptor.health_uri, timeout=1.0)

        # ``urllib.request.urlopen`` reuses a module-level opener that earlier
        # probes built under a proxy-free env; drop it so a regression to
        # ``urlopen`` really sees HTTP_PROXY and fails below.
        monkeypatch.setattr(urllib.request, "_opener", None)
        probe = probe_backend(descriptor, timeout=1.0)
        assert probe.backend == BACKEND_ECHO
        assert probe.payload["status"] == "ok"
    finally:
        listener.terminate()
        listener.wait(timeout=3)


def test_supervisor_logs_progress_while_child_is_still_starting(tmp_path):
    """S3 (e): periodic ``still starting`` lines and a final error that carries elapsed."""

    logs = []
    supervisor, children = _never_listening_supervisor(
        tmp_path,
        log=logs.append,
        progress_interval=0.1,
    )
    with pytest.raises(BackendProbeError) as excinfo:
        supervisor.start(BACKEND_ECHO, timeout=0.45)
    pid = children[0].pid
    progress = [line for line in logs if "still starting" in line]
    assert len(progress) >= 2, logs
    for line in progress:
        assert line.startswith("[Backend] EchoWave still starting: ")
        assert re.search(r"still starting: \d+\.\ds elapsed", line)
        assert f"pid {pid}" in line
        assert "last probe: " in line and "Connection refused" in line
    message = str(excinfo.value)
    assert re.search(r"within \d+\.\ds", message)
    assert f"pid {pid}" in message
    assert "Connection refused" in message
    assert children[0].terminated == 1


def test_supervisor_logs_healthy_line_with_elapsed_and_pid(tmp_path):
    """S3 (e): success prints one ``healthy after`` line."""

    logs = []
    descriptors = {
        BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, _free_port()),
        BACKEND_WHISPER: _descriptor(tmp_path, BACKEND_WHISPER, _free_port()),
    }
    children = []

    def popen(_command, **_kwargs):
        child = _NeverListeningChild()
        children.append(child)
        return child

    supervisor = BackendSupervisor(
        descriptors,
        popen_factory=popen,
        probe_func=lambda descriptor, _timeout: _probe(descriptor),
        log=logs.append,
    )
    supervisor.start(BACKEND_WHISPER, timeout=1.0)
    supervisor.stop(BACKEND_WHISPER)
    healthy = [line for line in logs if "healthy after" in line]
    assert len(healthy) == 1
    assert re.fullmatch(
        rf"\[Backend\] WhisperWave healthy after \d+\.\ds \(pid {children[0].pid}\)",
        healthy[0],
    )
    assert not any("still starting" in line for line in logs)


def test_start_initial_passes_none_timeout_so_supervisor_is_single_source(tmp_path):
    """S4 (f): the coordinator does not pick a limit; ``None`` reaches the supervisor."""

    store = BackendSelectionStore(tmp_path / "selection.json")
    supervisor = FakeSupervisor(tmp_path)
    coordinator = BackendCoordinator(supervisor, store)
    coordinator.start_initial()
    assert supervisor.start_timeouts == [None]
    coordinator.start_initial(timeout=3.0)
    assert supervisor.start_timeouts == [None, 3.0]


@pytest.mark.asyncio
async def test_switch_passes_none_timeout_to_supervisor_start(tmp_path):
    """S4 (f): the switch transaction also leaves the limit to the supervisor."""

    store = BackendSelectionStore(tmp_path / "selection.json")
    supervisor = FakeSupervisor(tmp_path)

    async def repoint(_descriptor):
        return True

    result = await BackendCoordinator(supervisor, store).switch(
        current_backend=BACKEND_ECHO,
        target_backend=BACKEND_WHISPER,
        state="idle",
        repoint=repoint,
    )
    assert result.effective_backend == BACKEND_WHISPER
    assert supervisor.start_timeouts == [None]


class _ExitsAfterPollsChild:
    """Fake Popen that is alive for the first ``polls_alive`` polls, then exited.

    ``first_poll_delay`` stalls the first poll, so the deadline is already past
    when that poll has answered "alive": the exact window between the in-loop
    poll and the deadline check in which a real child can die (review F3).
    """

    _next_pid = 72000

    def __init__(self, *, polls_alive: int, exit_code: int, first_poll_delay: float = 0.0):
        self.pid = type(self)._next_pid
        type(self)._next_pid += 1
        self._polls_alive = polls_alive
        self._exit_code = exit_code
        self._first_poll_delay = first_poll_delay
        self.polls = 0
        self.terminated = 0

    def poll(self):
        self.polls += 1
        if self.polls == 1 and self._first_poll_delay:
            time.sleep(self._first_poll_delay)
        if self.polls <= self._polls_alive:
            return None
        return self._exit_code

    def terminate(self):
        self.terminated += 1

    def wait(self, timeout=None):
        return self._exit_code

    def kill(self):
        pass


def _refusing_probe(descriptor, _timeout):
    raise BackendProbeError(f"{descriptor.label} health probe failed: Connection refused")


def _slow_process_supervisor(tmp_path: Path, key: str):
    """Real child that binds after 5s (or after ``bind_delay`` file seconds)."""

    delay_file = tmp_path / "bind_delay"
    delay_file.write_text("5", encoding="ascii")
    script = _write_contract_server(tmp_path, bind_delay_sec=5.0, bind_delay_file=delay_file)
    slow = _process_descriptor(tmp_path, key, _free_port(), script)
    other_key = BACKEND_WHISPER if key == BACKEND_ECHO else BACKEND_ECHO
    other = _descriptor(tmp_path, other_key, _free_port())
    spawned = []

    def popen(*args, **kwargs):
        child = subprocess.Popen(*args, **kwargs)
        spawned.append(child)
        return child

    supervisor = BackendSupervisor({key: slow, other_key: other}, popen_factory=popen)
    return supervisor, slow, spawned, delay_file


def _start_in_thread(supervisor, key: str, spawned: list, *, timeout: float = 30.0):
    """Run start() on a worker; return (thread, outcome) once the child is spawned."""

    outcome = {}

    def run():
        started = time.monotonic()
        try:
            outcome["probe"] = supervisor.start(key, timeout=timeout)
        except BaseException as exc:  # noqa: BLE001 - recorded for the test thread
            outcome["error"] = exc
        outcome["elapsed"] = time.monotonic() - started

    worker = threading.Thread(target=run, name=f"start-{key}")
    worker.start()
    deadline = time.monotonic() + 5.0
    while not spawned and time.monotonic() < deadline:
        time.sleep(0.01)
    assert spawned, "start() never spawned the child"
    return worker, outcome


def _kill_leftovers(spawned: list) -> None:
    for child in spawned:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


def _assert_aborted(outcome, child) -> None:
    error = outcome["error"]
    assert isinstance(error, BackendRuntimeError), error
    assert not isinstance(error, BackendProbeError), error
    assert re.search(r"startup aborted by stop request after \d+\.\ds", str(error)), error
    assert f"pid {child.pid}" in str(error)
    assert outcome["elapsed"] < 3.0, outcome
    assert child.poll() is not None


def test_stop_from_another_thread_aborts_slow_start_and_next_start_succeeds(tmp_path):
    """F2: stop() during a slow start yields within ~1s; the flag does not leak."""

    supervisor, echo, spawned, delay_file = _slow_process_supervisor(tmp_path, BACKEND_ECHO)
    worker, outcome = _start_in_thread(supervisor, BACKEND_ECHO, spawned)
    try:
        time.sleep(0.5)
        stop_requested = time.monotonic()
        supervisor.stop(BACKEND_ECHO)
        stop_elapsed = time.monotonic() - stop_requested
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert stop_elapsed < 1.5, stop_elapsed
        _assert_aborted(outcome, spawned[0])
        assert BACKEND_ECHO not in supervisor._children
        assert not Path(echo.pid_file).exists()
        assert not supervisor._abort_events[BACKEND_ECHO].is_set()

        delay_file.write_text("0", encoding="ascii")
        probe = supervisor.start(BACKEND_ECHO, timeout=10.0)
        assert probe.backend == BACKEND_ECHO
        assert len(spawned) == 2
        assert spawned[1].poll() is None
        assert int(Path(echo.pid_file).read_text(encoding="ascii")) == spawned[1].pid
    finally:
        supervisor.stop(BACKEND_ECHO)
        _kill_leftovers(spawned)
    assert not Path(echo.pid_file).exists()


def test_stop_all_from_another_thread_aborts_slow_start_promptly(tmp_path):
    """F2: quit/SIGTERM path. stop_all() returns in ~1s, not after the 30s limit."""

    supervisor, whisper, spawned, delay_file = _slow_process_supervisor(tmp_path, BACKEND_WHISPER)
    worker, outcome = _start_in_thread(supervisor, BACKEND_WHISPER, spawned)
    try:
        time.sleep(0.5)
        stop_requested = time.monotonic()
        supervisor.stop_all()
        stop_elapsed = time.monotonic() - stop_requested
        worker.join(timeout=5.0)
        assert not worker.is_alive()
        assert stop_elapsed < 1.5, stop_elapsed
        _assert_aborted(outcome, spawned[0])
        assert supervisor._children == {}
        assert not Path(whisper.pid_file).exists()
        assert not any(event.is_set() for event in supervisor._abort_events.values())

        delay_file.write_text("0", encoding="ascii")
        probe = supervisor.start(BACKEND_WHISPER, timeout=10.0)
        assert probe.backend == BACKEND_WHISPER
        assert spawned[1].poll() is None
    finally:
        supervisor.stop_all()
        _kill_leftovers(spawned)


def test_restart_does_not_abort_its_own_start(tmp_path):
    """F2: restart() is stop-then-start on one thread; the flag must not carry over."""

    descriptors = {
        BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, _free_port()),
        BACKEND_WHISPER: _descriptor(tmp_path, BACKEND_WHISPER, _free_port()),
    }
    children = []

    def popen(_command, **_kwargs):
        child = _NeverListeningChild()
        children.append(child)
        return child

    supervisor = BackendSupervisor(
        descriptors,
        popen_factory=popen,
        probe_func=lambda descriptor, _timeout: _probe(descriptor),
        log=lambda _line: None,
    )
    supervisor.start(BACKEND_ECHO, timeout=1.0)
    probe = supervisor.restart(BACKEND_ECHO, timeout=1.0)
    assert probe.backend == BACKEND_ECHO
    assert [child.terminated for child in children] == [1, 0]
    assert supervisor._children[BACKEND_ECHO] is children[1]
    assert not supervisor._abort_events[BACKEND_ECHO].is_set()
    supervisor.stop(BACKEND_ECHO)


def test_stop_reentered_on_the_start_thread_does_not_deadlock(tmp_path):
    """F2: a signal handler on the launcher thread calls stop() inside start().

    The re-entrant stop reaps the child itself; start() then reports the exit
    instead of waiting out the limit, and no flag is left behind.
    """

    descriptors = {
        BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, _free_port()),
        BACKEND_WHISPER: _descriptor(tmp_path, BACKEND_WHISPER, _free_port()),
    }
    children = []

    def popen(_command, **_kwargs):
        child = _NeverListeningChild()
        children.append(child)
        return child

    probes = []

    def probe(descriptor, _timeout):
        probes.append(descriptor.key)
        if len(probes) == 1:
            supervisor.stop_all()
        raise BackendProbeError(f"{descriptor.label} health probe failed: Connection refused")

    supervisor = BackendSupervisor(descriptors, popen_factory=popen, probe_func=probe)
    started = time.monotonic()
    with pytest.raises(BackendRuntimeError) as excinfo:
        supervisor.start(BACKEND_ECHO, timeout=30.0)
    assert time.monotonic() - started < 2.0
    assert not isinstance(excinfo.value, BackendProbeError)
    assert children[0].terminated == 1
    assert supervisor._children == {}
    assert not any(event.is_set() for event in supervisor._abort_events.values())


def test_supervisor_reports_exit_code_when_child_exits_between_poll_and_deadline(tmp_path):
    """F3/I2: an exit in the poll-to-deadline window is reported with its code, not as hung."""

    descriptors = {
        BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, _free_port()),
        BACKEND_WHISPER: _descriptor(tmp_path, BACKEND_WHISPER, _free_port()),
    }
    children = []

    def popen(_command, **_kwargs):
        child = _ExitsAfterPollsChild(polls_alive=1, exit_code=7, first_poll_delay=0.2)
        children.append(child)
        return child

    logs = []
    supervisor = BackendSupervisor(
        descriptors,
        popen_factory=popen,
        probe_func=_refusing_probe,
        log=logs.append,
    )
    with pytest.raises(BackendRuntimeError) as excinfo:
        supervisor.start(BACKEND_ECHO, timeout=0.05)
    message = str(excinfo.value)
    assert not isinstance(excinfo.value, BackendProbeError), message
    assert "exited during startup with code 7" in message
    assert "hung" not in message
    assert f"pid {children[0].pid}" in message
    # Loop poll (alive), post-deadline poll (exited), reap's own poll: no hung path.
    assert children[0].polls == 3
    assert children[0].terminated == 0
    assert BACKEND_ECHO not in supervisor._children
    assert not Path(descriptors[BACKEND_ECHO].pid_file).exists()


def test_supervisor_warns_once_per_invalid_env_value_across_starts(tmp_path, monkeypatch):
    """F4: three start() calls under one bad value print one warning; a new value warns again."""

    descriptors = {
        BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, _free_port()),
        BACKEND_WHISPER: _descriptor(tmp_path, BACKEND_WHISPER, _free_port()),
    }
    logs = []
    supervisor = BackendSupervisor(
        descriptors,
        popen_factory=lambda _command, **_kwargs: _NeverListeningChild(),
        probe_func=lambda descriptor, _timeout: _probe(descriptor),
        log=logs.append,
    )

    def warnings():
        return [line for line in logs if "ignoring invalid" in line]

    monkeypatch.setenv(BACKEND_START_TIMEOUT_ENV, "abc")
    for _ in range(3):
        supervisor.start(BACKEND_ECHO)
        supervisor.stop(BACKEND_ECHO)
    assert len(warnings()) == 1
    assert "'abc'" in warnings()[0]

    monkeypatch.setenv(BACKEND_START_TIMEOUT_ENV, "-3")
    supervisor.start(BACKEND_ECHO)
    supervisor.stop(BACKEND_ECHO)
    assert len(warnings()) == 2
    assert "'-3'" in warnings()[1]

    monkeypatch.setenv(BACKEND_START_TIMEOUT_ENV, "abc")
    supervisor.start(BACKEND_ECHO)
    supervisor.stop(BACKEND_ECHO)
    assert len(warnings()) == 2
    assert resolve_backend_start_timeout(log=logs.append) == DEFAULT_BACKEND_START_TIMEOUT_SEC
    assert len(warnings()) == 3, "the pure resolver still warns every call"


def test_stop_all_failure_on_one_backend_does_not_leak_abort_flag_to_the_other(tmp_path):
    """review r2 N1: a stop() that raises must not skip the other backend's stop/clear."""
    script = _write_contract_server(tmp_path)
    whisper_port = _free_port()
    supervisor = BackendSupervisor(
        {
            BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, _free_port()),
            BACKEND_WHISPER: _process_descriptor(tmp_path, BACKEND_WHISPER, whisper_port, script),
        }
    )
    # Corrupt echo PID record: stop(echo) raises before whisper is reached.
    Path(supervisor.descriptor(BACKEND_ECHO).pid_file).write_text("not-a-pid\n", encoding="ascii")
    with pytest.raises(BackendRuntimeError, match="invalid EchoWave PID record"):
        supervisor.stop_all()
    assert not supervisor._abort_events[BACKEND_WHISPER].is_set()
    assert not supervisor._abort_events[BACKEND_ECHO].is_set()
    try:
        probe = supervisor.start(BACKEND_WHISPER, timeout=10.0)
        assert probe.backend == BACKEND_WHISPER
        assert Path(supervisor.descriptor(BACKEND_WHISPER).pid_file).exists()
    finally:
        supervisor.stop(BACKEND_WHISPER)


def test_start_aborts_before_spawn_when_stop_is_already_pending(tmp_path):
    """review r2 N2: a pending stop must not cost a wasted spawn."""
    spawned = []

    def popen_factory(*args, **kwargs):
        spawned.append(args)
        raise AssertionError("must not spawn while a stop is pending")

    supervisor = BackendSupervisor(
        {
            BACKEND_ECHO: _descriptor(tmp_path, BACKEND_ECHO, _free_port()),
            BACKEND_WHISPER: _descriptor(tmp_path, BACKEND_WHISPER, _free_port()),
        },
        popen_factory=popen_factory,
        probe_func=lambda descriptor, _timeout: _probe(descriptor),
    )
    supervisor._abort_events[BACKEND_WHISPER].set()
    with pytest.raises(BackendRuntimeError, match="aborted by stop request before spawn"):
        supervisor.start(BACKEND_WHISPER, timeout=1.0)
    assert spawned == []
    supervisor.stop(BACKEND_WHISPER)
    assert not supervisor._abort_events[BACKEND_WHISPER].is_set()
