"""Desktop-shell ownership for EchoWave's two isolated local backends.

This module deliberately contains no provider implementation.  It owns only
backend descriptors, exact child-process lifecycle, the versioned wire-contract
probe, the atomic desktop selection, and the switch transaction.  Echo and
Whisper remain separate interpreters and communicate with the menu shell only
through the PCM + WebSocket protocol.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Optional


BACKEND_ECHO = "echo"
BACKEND_WHISPER = "whisper"
VALID_BACKENDS = frozenset({BACKEND_ECHO, BACKEND_WHISPER})
SELECTION_SCHEMA_VERSION = 1
WIRE_CONTRACT_NAME = "echowave-pcm-websocket"
WIRE_CONTRACT_VERSION = 1

_COMMON_CHILD_ENV_KEYS = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)

# Startup gate (task 0607). A child that is alive but not yet listening is
# "slow", which is recoverable; only exit, contract mismatch, or exceeding this
# limit count as failure. 90s absorbs cold page cache, swap pressure, and a
# saturated CPU; a real hang is still reported, with diagnostics.
BACKEND_START_TIMEOUT_ENV = "ECHOWAVE_BACKEND_START_TIMEOUT_SEC"
DEFAULT_BACKEND_START_TIMEOUT_SEC = 90.0
DEFAULT_START_PROGRESS_INTERVAL_SEC = 5.0


def resolve_backend_start_timeout(
    raw: Optional[str] = None,
    *,
    log: Callable[[str], None] = print,
) -> float:
    """Return the startup wait limit in seconds.

    ``raw`` defaults to ``ECHOWAVE_BACKEND_START_TIMEOUT_SEC``. Unset or blank
    means the default. Anything that is not a finite number greater than zero
    is reported once through ``log`` and replaced by the default: a bad knob
    must never turn into a failed startup.
    """

    if raw is None:
        raw = os.environ.get(BACKEND_START_TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_BACKEND_START_TIMEOUT_SEC
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        log(
            f"[Backend] ignoring invalid {BACKEND_START_TIMEOUT_ENV}={raw!r}; "
            f"using default {DEFAULT_BACKEND_START_TIMEOUT_SEC:g}s"
        )
        return DEFAULT_BACKEND_START_TIMEOUT_SEC
    return value


class BackendRuntimeError(RuntimeError):
    """Base error for backend lifecycle and routing failures."""


class SelectionStoreError(BackendRuntimeError):
    """The persisted selection is malformed, unknown, or could not be saved."""


class BackendProbeError(BackendRuntimeError):
    """A process did not expose the expected health/contract surface."""


class BackendSwitchError(BackendRuntimeError):
    """A switch transaction failed and the old backend remained selected."""


class SwitchNotAllowed(BackendSwitchError):
    """A switch was requested while a turn owned the capture/transport path."""


class BackendSelectionStore:
    """Atomic shell-owned JSON selection under Application Support."""

    def __init__(self, path: Optional[os.PathLike[str] | str] = None):
        default = Path.home() / "Library" / "Application Support" / "EchoWave" / "backend-selection.json"
        self.path = Path(path).expanduser() if path is not None else default

    def load(self) -> str:
        """Return the selected backend; a missing file means stable Echo.

        Existing but malformed data is never guessed.  The caller must surface
        the diagnostic and make an explicit startup fallback decision.
        """

        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return BACKEND_ECHO
        except OSError as exc:
            raise SelectionStoreError(f"cannot read backend selection {self.path}: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SelectionStoreError(
                f"invalid backend selection JSON at {self.path}: line {exc.lineno} column {exc.colno}"
            ) from exc
        if not isinstance(payload, dict):
            raise SelectionStoreError(f"backend selection at {self.path} must be a JSON object")
        if payload.get("version") != SELECTION_SCHEMA_VERSION:
            raise SelectionStoreError(
                f"unsupported backend selection version at {self.path}: {payload.get('version')!r}"
            )
        selected = payload.get("selected_backend")
        if selected not in VALID_BACKENDS:
            raise SelectionStoreError(
                f"unknown selected_backend at {self.path}: {selected!r}"
            )
        return str(selected)

    def save(self, selected_backend: str) -> None:
        if selected_backend not in VALID_BACKENDS:
            raise SelectionStoreError(f"refusing to persist unknown backend: {selected_backend!r}")
        payload = {
            "version": SELECTION_SCHEMA_VERSION,
            "selected_backend": selected_backend,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            try:
                dir_fd = os.open(self.path.parent, os.O_RDONLY)
            except OSError:
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        except Exception as exc:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            if isinstance(exc, SelectionStoreError):
                raise
            raise SelectionStoreError(f"cannot persist backend selection {self.path}: {exc}") from exc


@dataclass(frozen=True)
class BackendDescriptor:
    key: str
    label: str
    host: str
    port: int
    command: tuple[str, ...]
    cwd: str
    pid_file: str
    log_file: str
    provider: str
    model: str
    environment: Mapping[str, str] = field(default_factory=dict)
    inherited_env_keys: tuple[str, ...] = field(default_factory=tuple)
    inherited_env_prefixes: tuple[str, ...] = field(default_factory=tuple)
    require_remote_contract: bool = False

    @property
    def websocket_uri(self) -> str:
        return f"ws://{self.host}:{self.port}/api/v1/ws"

    @property
    def health_uri(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    @property
    def contract_uri(self) -> str:
        return f"http://{self.host}:{self.port}/api/v1/desktop-contract"


@dataclass(frozen=True)
class HealthProbe:
    backend: str
    contract_name: str
    contract_version: int
    endpoint: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class StartupDecision:
    requested_backend: str
    effective_backend: str
    diagnostic: Optional[str] = None


@dataclass(frozen=True)
class SwitchResult:
    previous_backend: str
    effective_backend: str
    probe: HealthProbe
    cleanup_warning: Optional[str] = None


def build_default_descriptors(
    project_dir: os.PathLike[str] | str,
    *,
    app_support_dir: Optional[os.PathLike[str] | str] = None,
    log_dir: Optional[os.PathLike[str] | str] = None,
    echo_port: int = 23456,
    whisper_port: int = 23459,
    echo_python: Optional[os.PathLike[str] | str] = None,
    whisper_python: Optional[os.PathLike[str] | str] = None,
) -> dict[str, BackendDescriptor]:
    project = Path(project_dir).resolve()
    support = (
        Path(app_support_dir).expanduser()
        if app_support_dir is not None
        else Path.home() / "Library" / "Application Support" / "EchoWave" / "backends"
    )
    logs = (
        Path(log_dir).expanduser()
        if log_dir is not None
        else Path.home() / "Library" / "Logs" / "EchoWave IME"
    )
    echo_python_path = Path(echo_python) if echo_python is not None else project / "venv" / "bin" / "python"
    whisper_python_path = (
        Path(whisper_python)
        if whisper_python is not None
        else project / "whisper_backend" / "venv" / "bin" / "python"
    )
    return {
        BACKEND_ECHO: BackendDescriptor(
            key=BACKEND_ECHO,
            label="EchoWave",
            host="127.0.0.1",
            port=int(echo_port),
            command=(str(echo_python_path), str(project / "echo_backend_entry.py")),
            cwd=str(project),
            pid_file=str(support / "echo.pid"),
            log_file=str(logs / "echo-backend.log"),
            provider="openai",
            model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1-mini"),
            environment={"ECHOWAVE_ECHO_PORT": str(int(echo_port))},
            inherited_env_keys=_COMMON_CHILD_ENV_KEYS
            + (
                "OPENAI_API_KEY",
                "GOOGLE_API_KEY",
                "GOOGLE_APPLICATION_CREDENTIALS",
                "OPENAI_REALTIME_MODEL",
                "OPENAI_REALTIME_MODALITIES",
            ),
            inherited_env_prefixes=("BRAINWAVE_",),
            require_remote_contract=True,
        ),
        BACKEND_WHISPER: BackendDescriptor(
            key=BACKEND_WHISPER,
            label="WhisperWave",
            host="127.0.0.1",
            port=int(whisper_port),
            command=(str(whisper_python_path), "-m", "whisper_backend"),
            cwd=str(project),
            pid_file=str(support / "whisper.pid"),
            log_file=str(logs / "whisper-backend.log"),
            provider="openai-transcription",
            model=os.getenv("WHISPERWAVE_MODEL", "gpt-4o-transcribe"),
            environment={"WHISPERWAVE_PORT": str(int(whisper_port))},
            inherited_env_keys=_COMMON_CHILD_ENV_KEYS + ("OPENAI_API_KEY",),
            inherited_env_prefixes=("WHISPERWAVE_",),
            require_remote_contract=True,
        ),
    }


def probe_backend(descriptor: BackendDescriptor, timeout: float = 1.0) -> HealthProbe:
    # Loopback probes go direct. An empty ProxyHandler keeps HTTP(S)_PROXY and
    # the macOS system proxy out of the path; the default opener would route
    # 127.0.0.1 through them and report the proxy's failure as the backend's.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def fetch_json(uri: str, surface: str) -> Mapping[str, object]:
        request = urllib.request.Request(
            uri,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with opener.open(request, timeout=timeout) as response:
                content_type = response.headers.get_content_type()
                if content_type != "application/json":
                    raise BackendProbeError(
                        f"{descriptor.label} {surface} returned content-type {content_type!r}"
                    )
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise BackendProbeError(
                f"{descriptor.label} {surface} probe failed: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise BackendProbeError(f"{descriptor.label} {surface} payload is not an object")
        return result

    payload = fetch_json(descriptor.health_uri, "health")
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise BackendProbeError(f"{descriptor.label} health payload is not ok")

    remote_backend = payload.get("backend")
    remote_contract = payload.get("contract")
    if descriptor.require_remote_contract:
        if remote_backend is None or remote_contract is None:
            contract_payload = fetch_json(descriptor.contract_uri, "contract")
            remote_backend = contract_payload.get("backend")
            remote_contract = contract_payload.get("contract")
        expected = {"name": WIRE_CONTRACT_NAME, "version": WIRE_CONTRACT_VERSION}
        if remote_backend != descriptor.key or remote_contract != expected:
            raise BackendProbeError(
                f"{descriptor.label} contract mismatch: backend={remote_backend!r}, contract={remote_contract!r}"
            )
    elif remote_backend is not None and remote_backend != descriptor.key:
        raise BackendProbeError(
            f"{descriptor.label} backend identity mismatch: {remote_backend!r}"
        )
    return HealthProbe(
        backend=descriptor.key,
        contract_name=WIRE_CONTRACT_NAME,
        contract_version=WIRE_CONTRACT_VERSION,
        endpoint=descriptor.websocket_uri,
        payload=payload,
    )


class BackendSupervisor:
    """Own exact backend child handles/PIDs; never scan by basename or pkill."""

    def __init__(
        self,
        descriptors: Mapping[str, BackendDescriptor],
        *,
        popen_factory=subprocess.Popen,
        probe_func: Callable[[BackendDescriptor, float], HealthProbe] = probe_backend,
        log: Callable[[str], None] = print,
        progress_interval: float = DEFAULT_START_PROGRESS_INTERVAL_SEC,
    ):
        self.descriptors = dict(descriptors)
        if set(self.descriptors) != VALID_BACKENDS:
            raise ValueError(f"descriptors must define exactly {sorted(VALID_BACKENDS)}")
        self._popen_factory = popen_factory
        self._probe_func = probe_func
        self._log = log
        self._progress_interval = max(0.01, float(progress_interval))
        self._children: dict[str, subprocess.Popen] = {}
        self._log_handles: dict[str, object] = {}
        self._lock = threading.RLock()
        # Cooperative cancel: stop()/stop_all() raise a backend's flag before
        # they block on the lock, so a start() holding it for up to the startup
        # limit yields within one loop turn instead of after the limit.
        self._abort_events: dict[str, threading.Event] = {
            key: threading.Event() for key in self.descriptors
        }
        # Invalid-env warnings already printed by this supervisor, keyed by the
        # raw value; restart/reconnect call start() repeatedly.
        self._warned_invalid_timeouts: set[str] = set()

    def descriptor(self, backend: str) -> BackendDescriptor:
        try:
            return self.descriptors[backend]
        except KeyError as exc:
            raise BackendRuntimeError(f"unknown backend: {backend!r}") from exc

    def _resolve_start_timeout(self) -> float:
        """Resolve the env limit; warn once per distinct invalid raw value."""

        raw = os.environ.get(BACKEND_START_TIMEOUT_ENV)
        warnings: list[str] = []
        timeout = resolve_backend_start_timeout(raw, log=warnings.append)
        if warnings and raw not in self._warned_invalid_timeouts:
            self._warned_invalid_timeouts.add(raw)
            for line in warnings:
                self._log(line)
        return timeout

    def start(self, backend: str, *, timeout: Optional[float] = None) -> HealthProbe:
        """Spawn the exact child and wait until it serves the contract.

        ``timeout=None`` resolves the limit from ``ECHOWAVE_BACKEND_START_TIMEOUT_SEC``
        (default 90s); an explicit value is used as given. While the child is
        alive the supervisor keeps waiting up to that limit. Child exit fails
        immediately with ``BackendRuntimeError``; a child still alive at the
        limit is stopped and reported as hung via ``BackendProbeError``. A
        ``stop()``/``stop_all()`` issued from another thread while waiting
        aborts the wait, stops the child, and raises ``BackendRuntimeError``.
        """

        descriptor = self.descriptor(backend)
        abort = self._abort_events[backend]
        if timeout is None:
            timeout = self._resolve_start_timeout()
        else:
            timeout = float(timeout)
        with self._lock:
            existing = self._children.get(backend)
            if existing is not None and existing.poll() is None:
                try:
                    probe = self._probe_func(descriptor, min(1.0, timeout))
                except BackendProbeError:
                    self._stop_in_memory_child(backend, timeout=min(5.0, timeout))
                else:
                    if existing.poll() is None:
                        try:
                            self._write_pid_file(descriptor, int(existing.pid))
                        except Exception:
                            self._stop_in_memory_child(backend, timeout=min(5.0, timeout))
                            raise
                        return probe
            elif existing is not None:
                self._stop_in_memory_child(backend, timeout=min(5.0, timeout))

            recovered_owned_pid = self._reconcile_pid_file_before_start(
                descriptor,
                timeout=min(5.0, max(0.1, timeout)),
            )
            if recovered_owned_pid:
                self._wait_for_port_available(
                    descriptor,
                    timeout=min(2.0, max(0.1, timeout)),
                )
            else:
                self._assert_port_available(descriptor)

            executable = Path(descriptor.command[0])
            if not executable.is_file():
                raise BackendRuntimeError(
                    f"{descriptor.label} runtime is missing: {executable}; install its isolated requirements first"
                )
            if abort.is_set():
                # A stop() is already queued behind this lock: do not spend a
                # spawn only to SIGTERM it on the first loop turn (review r2 N2).
                raise BackendRuntimeError(
                    f"{descriptor.label} startup aborted by stop request before spawn"
                )
            Path(descriptor.pid_file).parent.mkdir(parents=True, exist_ok=True)
            Path(descriptor.log_file).parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(descriptor.log_file, "a", encoding="utf-8")
            env = self._build_child_env(descriptor)
            try:
                child = self._popen_factory(
                    list(descriptor.command),
                    cwd=descriptor.cwd,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=False,
                )
            except Exception:
                log_handle.close()
                raise
            self._children[backend] = child
            self._log_handles[backend] = log_handle

            pid = int(child.pid)
            started_at = time.monotonic()
            deadline = started_at + max(0.05, timeout)
            next_progress_at = started_at + self._progress_interval
            last_error: Optional[Exception] = None

            def fail_exited(exit_code: int) -> None:
                # Dead is a different failure mode from slow: reap now, never
                # sit out the deadline for a process that is gone.
                elapsed = time.monotonic() - started_at
                self._stop_in_memory_child(backend, timeout=min(5.0, timeout))
                raise BackendRuntimeError(
                    f"{descriptor.label} exited during startup with code {exit_code} "
                    f"after {elapsed:.1f}s (pid {pid}); last probe: "
                    f"{last_error or 'none'}"
                )

            while True:
                exit_code = child.poll()
                if exit_code is not None:
                    fail_exited(exit_code)
                if abort.is_set():
                    # A stop()/stop_all() is queued behind our lock: hand the
                    # child over now. The stop clears the flag once it gets in.
                    elapsed = time.monotonic() - started_at
                    self._stop_in_memory_child(backend, timeout=min(5.0, timeout))
                    raise BackendRuntimeError(
                        f"{descriptor.label} startup aborted by stop request after "
                        f"{elapsed:.1f}s (pid {pid})"
                    )
                now = time.monotonic()
                if now >= deadline:
                    break
                try:
                    probe = self._probe_func(
                        descriptor,
                        min(0.5, max(0.05, deadline - now)),
                    )
                except BackendProbeError as exc:
                    last_error = exc
                    now = time.monotonic()
                    if now >= next_progress_at:
                        self._log(
                            f"[Backend] {descriptor.label} still starting: "
                            f"{now - started_at:.1f}s elapsed, pid {pid}, last probe: {exc}"
                        )
                        while next_progress_at <= now:
                            next_progress_at += self._progress_interval
                    abort.wait(0.05)
                    continue
                exit_code = child.poll()
                if exit_code is not None:
                    elapsed = time.monotonic() - started_at
                    self._stop_in_memory_child(backend, timeout=min(5.0, timeout))
                    raise BackendRuntimeError(
                        f"{descriptor.label} exited immediately after health probe "
                        f"with code {exit_code} after {elapsed:.1f}s (pid {pid})"
                    )
                try:
                    # The stable ownership record is committed only after the
                    # candidate is both healthy and still alive. A listener that
                    # existed before spawn can therefore never be attributed to
                    # this PID.
                    self._write_pid_file(descriptor, pid)
                except Exception:
                    self._stop_in_memory_child(backend, timeout=min(5.0, timeout))
                    raise
                self._log(
                    f"[Backend] {descriptor.label} healthy after "
                    f"{time.monotonic() - started_at:.1f}s (pid {pid})"
                )
                return probe

            # The child may have exited between the poll at the top of the last
            # turn and the deadline check; its exit code beats a "hung" verdict.
            exit_code = child.poll()
            if exit_code is not None:
                fail_exited(exit_code)

            # Deadline reached with the child still alive: hung, not dead. Only
            # now is it stopped, and the message carries everything needed to
            # tell "slow host" from "broken backend" in the IME log.
            elapsed = time.monotonic() - started_at
            self._stop_in_memory_child(backend, timeout=min(5.0, timeout))
            raise BackendProbeError(
                f"{descriptor.label} did not pass health/contract probe within "
                f"{elapsed:.1f}s (pid {pid} was still alive at the limit and was "
                f"stopped as hung; raise {BACKEND_START_TIMEOUT_ENV} if the host is "
                f"just slow); last probe: {last_error or 'no probe attempted'}"
            )

    def stop(self, backend: str, *, timeout: float = 5.0) -> None:
        descriptor = self.descriptor(backend)
        abort = self._abort_events[backend]
        # Raise the flag before blocking: a start() holding the lock checks it
        # every loop turn. Clear it once inside so it cannot leak into the next
        # start() (restart() is stop-then-start on the same thread).
        abort.set()
        with self._lock:
            abort.clear()
            child = self._children.get(backend)
            if child is not None:
                child_pid = int(child.pid)
                self._stop_in_memory_child(backend, timeout=timeout)
                self._remove_pid_file_if_matches(descriptor, child_pid)
                return
            stopped_pid = self._stop_exact_pid_file_process(
                descriptor,
                timeout=timeout,
            )
            if stopped_pid is not None:
                self._remove_pid_file_if_matches(descriptor, stopped_pid)

    def restart(self, backend: str, *, timeout: Optional[float] = None) -> HealthProbe:
        self.stop(backend)
        return self.start(backend, timeout=timeout)

    def stop_all(self) -> None:
        backends = tuple(self.descriptors)
        # Raise every flag first: whichever backend a start() is waiting on
        # must yield before the per-backend stops queue behind its lock.
        for backend in backends:
            self._abort_events[backend].set()
        # Every backend gets its stop() (which clears its own flag inside the
        # lock) even when an earlier one raises; a skipped backend would keep a
        # set flag and have every later start() abort at once (review r2 N1).
        errors: list[Exception] = []
        for backend in backends:
            try:
                self.stop(backend)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    @staticmethod
    def _build_child_env(descriptor: BackendDescriptor) -> dict[str, str]:
        env: dict[str, str] = {}
        for key in descriptor.inherited_env_keys:
            if key in os.environ:
                env[key] = os.environ[key]
        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in descriptor.inherited_env_prefixes):
                env[key] = value
        env.update({key: str(value) for key, value in descriptor.environment.items()})
        return env

    @staticmethod
    def _assert_port_available(descriptor: BackendDescriptor) -> None:
        family = socket.AF_INET6 if ":" in descriptor.host else socket.AF_INET
        address = (descriptor.host, descriptor.port)
        with socket.socket(family, socket.SOCK_STREAM) as probe_sock:
            probe_sock.settimeout(0.1)
            if probe_sock.connect_ex(address) == 0:
                raise BackendRuntimeError(
                    f"{descriptor.label} port {descriptor.host}:{descriptor.port} "
                    "is already in use by an unknown or non-owned listener"
                )
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            # Backend servers use normal restart-safe SO_REUSEADDR behavior.
            # Matching it avoids mistaking TIME_WAIT from the final health probe
            # for a live listener while connect_ex above still rejects one.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(address)
            except OSError as exc:
                raise BackendRuntimeError(
                    f"{descriptor.label} port {descriptor.host}:{descriptor.port} "
                    "is already in use by an unknown or non-owned listener"
                ) from exc

    @classmethod
    def _wait_for_port_available(
        cls,
        descriptor: BackendDescriptor,
        *,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        last_error = None
        while True:
            try:
                cls._assert_port_available(descriptor)
                return
            except BackendRuntimeError as exc:
                last_error = exc
            if time.monotonic() >= deadline:
                raise BackendRuntimeError(
                    f"{descriptor.label} exact owned PID exited but port "
                    f"{descriptor.host}:{descriptor.port} did not become available"
                ) from last_error
            time.sleep(0.02)

    def _stop_in_memory_child(self, backend: str, *, timeout: float) -> None:
        child = self._children.pop(backend, None)
        log_handle = self._log_handles.pop(backend, None)
        try:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=max(0.1, timeout))
        finally:
            if log_handle is not None:
                log_handle.close()

    @staticmethod
    def _read_pid_file(descriptor: BackendDescriptor) -> Optional[int]:
        try:
            raw_pid = Path(descriptor.pid_file).read_text(encoding="ascii").strip()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise BackendRuntimeError(
                f"cannot read {descriptor.label} PID record {descriptor.pid_file}: {exc}"
            ) from exc
        try:
            pid = int(raw_pid)
        except ValueError as exc:
            raise BackendRuntimeError(
                f"invalid {descriptor.label} PID record at {descriptor.pid_file}"
            ) from exc
        if pid <= 0:
            raise BackendRuntimeError(
                f"invalid {descriptor.label} PID record at {descriptor.pid_file}"
            )
        return pid

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _remove_pid_file_if_matches(descriptor: BackendDescriptor, pid: int) -> None:
        path = Path(descriptor.pid_file)
        try:
            recorded_pid = int(path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, OSError, ValueError):
            return
        if recorded_pid != pid:
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _reconcile_pid_file_before_start(
        self,
        descriptor: BackendDescriptor,
        *,
        timeout: float,
    ) -> bool:
        pid = self._read_pid_file(descriptor)
        if pid is None:
            return False
        command = self._pid_command(pid)
        if command is None:
            if self._pid_exists(pid):
                raise BackendRuntimeError(
                    f"{descriptor.label} PID {pid} is live but its command identity "
                    "cannot be verified; refusing to signal or overwrite the record"
                )
            self._remove_pid_file_if_matches(descriptor, pid)
            return False
        if not self._command_matches(descriptor, command):
            raise BackendRuntimeError(
                f"{descriptor.label} PID record points to a non-owned live process; "
                "refusing to signal or overwrite it"
            )
        stopped_pid = self._stop_exact_pid_file_process(descriptor, timeout=timeout)
        if stopped_pid is not None:
            self._remove_pid_file_if_matches(descriptor, stopped_pid)
            return True
        return False

    @staticmethod
    def _write_pid_file(descriptor: BackendDescriptor, pid: int) -> None:
        target = Path(descriptor.pid_file)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(f"{pid}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _pid_command(pid: int) -> Optional[str]:
        try:
            output = subprocess.check_output(
                ["/bin/ps", "-p", str(pid), "-o", "command="],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return None
        return output.strip() or None

    @staticmethod
    def _expected_executable_identities(executable: str) -> set[Path]:
        identities = {Path(executable).resolve()}
        executable_path = Path(executable)
        pyvenv_path = executable_path.parent.parent / "pyvenv.cfg"
        try:
            lines = pyvenv_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return identities
        for line in lines:
            key, separator, value = line.partition("=")
            if not separator or key.strip() != "executable":
                continue
            base_executable = Path(value.strip()).resolve()
            identities.add(base_executable)
            version_root = base_executable.parent.parent
            python_app = (
                version_root
                / "Resources"
                / "Python.app"
                / "Contents"
                / "MacOS"
                / "Python"
            )
            if python_app.is_file():
                identities.add(python_app.resolve())
            break
        return identities

    @classmethod
    def _command_matches(cls, descriptor: BackendDescriptor, command: str) -> bool:
        observed_executable, separator, observed_args = command.partition(" ")
        try:
            observed_identity = Path(observed_executable).resolve(strict=True)
        except OSError:
            return False
        executable_matches = observed_identity in cls._expected_executable_identities(
            descriptor.command[0]
        )
        expected_args = " ".join(descriptor.command[1:])
        if not expected_args:
            return executable_matches and not separator
        return executable_matches and bool(separator) and observed_args == expected_args

    def _stop_exact_pid_file_process(
        self,
        descriptor: BackendDescriptor,
        *,
        timeout: float,
    ) -> Optional[int]:
        pid = self._read_pid_file(descriptor)
        if pid is None:
            return None
        command = self._pid_command(pid)
        if command is None:
            if self._pid_exists(pid):
                raise BackendRuntimeError(
                    f"{descriptor.label} PID {pid} command identity cannot be verified; "
                    "refusing to signal it"
                )
            return pid
        if not self._command_matches(descriptor, command):
            raise BackendRuntimeError(
                f"{descriptor.label} PID {pid} is not an exact command match; "
                "refusing to signal it"
            )
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return pid
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            current_command = self._pid_command(pid)
            if (
                current_command is None
                or not self._command_matches(descriptor, current_command)
            ):
                return pid
            time.sleep(0.05)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return pid
        kill_deadline = time.monotonic() + max(0.1, min(1.0, timeout))
        while time.monotonic() < kill_deadline:
            current_command = self._pid_command(pid)
            if (
                current_command is None
                or not self._command_matches(descriptor, current_command)
            ):
                return pid
            time.sleep(0.05)
        raise BackendRuntimeError(
            f"{descriptor.label} exact PID {pid} did not exit after SIGKILL"
        )


class BackendCoordinator:
    """Cold-start selection and idle-only start→probe→repoint→persist→stop."""

    def __init__(self, supervisor: BackendSupervisor, selection_store: BackendSelectionStore):
        self.supervisor = supervisor
        self.selection_store = selection_store

    def start_initial(self, *, timeout: Optional[float] = None) -> StartupDecision:
        # ``None`` is passed through so BackendSupervisor.start() remains the
        # single place that resolves the startup limit (env or default).
        try:
            requested = self.selection_store.load()
        except SelectionStoreError as exc:
            self.supervisor.start(BACKEND_ECHO, timeout=timeout)
            return StartupDecision(
                requested_backend="invalid",
                effective_backend=BACKEND_ECHO,
                diagnostic=str(exc),
            )
        try:
            self.supervisor.start(requested, timeout=timeout)
            return StartupDecision(requested, requested, None)
        except BackendRuntimeError as exc:
            if requested == BACKEND_ECHO:
                raise
            self.supervisor.start(BACKEND_ECHO, timeout=timeout)
            return StartupDecision(
                requested_backend=requested,
                effective_backend=BACKEND_ECHO,
                diagnostic=f"requested {requested} unavailable; explicit fallback to echo: {exc}",
            )

    async def switch(
        self,
        *,
        current_backend: str,
        target_backend: str,
        state: str,
        repoint: Callable[[BackendDescriptor], Awaitable[bool]],
        timeout: Optional[float] = None,
    ) -> SwitchResult:
        if state not in {"idle", "disconnected"}:
            raise SwitchNotAllowed(f"backend switch is disabled while state={state}")
        if current_backend not in VALID_BACKENDS or target_backend not in VALID_BACKENDS:
            raise BackendSwitchError("switch contains an unknown backend")
        if current_backend == target_backend:
            probe = await asyncio.to_thread(
                self.supervisor._probe_func,
                self.supervisor.descriptor(current_backend),
                1.0 if timeout is None else min(1.0, timeout),
            )
            return SwitchResult(current_backend, current_backend, probe)

        target = self.supervisor.descriptor(target_backend)
        previous = self.supervisor.descriptor(current_backend)
        try:
            probe = await asyncio.to_thread(self.supervisor.start, target_backend, timeout=timeout)
        except Exception as exc:
            raise BackendSwitchError(
                f"target {target.label} failed health/contract probe; keeping {previous.label}: {exc}"
            ) from exc

        try:
            if not bool(await repoint(target)):
                raise BackendSwitchError(f"menu client could not connect to {target.label}")
            await asyncio.to_thread(self.selection_store.save, target_backend)
        except Exception as exc:
            rollback_errors = []
            try:
                restored = bool(await repoint(previous))
                if not restored:
                    rollback_errors.append(
                        f"previous endpoint {previous.label} did not reconnect"
                    )
            except Exception as rollback_exc:
                rollback_errors.append(
                    f"previous endpoint restore failed: {rollback_exc}"
                )
            try:
                # save() can raise after mutation (for example after replace but
                # before directory fsync). Explicitly write the previous choice
                # instead of inferring that an exception means no mutation.
                await asyncio.to_thread(
                    self.selection_store.save,
                    current_backend,
                )
            except Exception as rollback_exc:
                rollback_errors.append(
                    f"previous selection restore failed: {rollback_exc}"
                )
            try:
                await asyncio.to_thread(self.supervisor.stop, target_backend)
            except Exception as cleanup_exc:
                rollback_errors.append(
                    f"target cleanup failed: {cleanup_exc}"
                )
            if rollback_errors:
                raise BackendSwitchError(
                    f"switch to {target.label} failed and rollback is degraded: "
                    f"{exc}; {'; '.join(rollback_errors)}"
                ) from exc
            raise BackendSwitchError(
                f"switch to {target.label} rolled back before commit: {exc}"
            ) from exc

        cleanup_warning = None
        try:
            await asyncio.to_thread(self.supervisor.stop, current_backend)
        except Exception as exc:
            # Repoint + persistence are the commit boundary. The selected target
            # remains authoritative; failure to reap the old child is visible
            # degraded cleanup, never an error that tells the UI to stay old.
            cleanup_warning = (
                f"switched to {target.label}, but cleanup of {previous.label} "
                f"failed and requires retry: {exc}"
            )
        return SwitchResult(
            current_backend,
            target_backend,
            probe,
            cleanup_warning=cleanup_warning,
        )
