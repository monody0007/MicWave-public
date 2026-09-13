#!/usr/bin/env python3
"""Single desktop-shell launcher for EchoWave's isolated backend children."""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from backend_runtime import (
    BackendCoordinator,
    BackendSelectionStore,
    BackendSupervisor,
    build_default_descriptors,
)


PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
RESOURCE_DIR = PROJECT_DIR
LEGACY_APP_SUPPORT_DIR = os.path.expanduser(
    "~/Library/Application Support/Brainwave IME"
)
SHELL_STATE_DIR = os.path.expanduser(
    "~/Library/Application Support/EchoWave/desktop-shell"
)
SHELL_PID_FILE = os.path.join(SHELL_STATE_DIR, "shell.pid")

_ECHO_DOTENV_ALLOWED_KEYS = frozenset(
    {
        "BRAINWAVE_ANSWER_GUARD_GRACE_SEC",
        "BRAINWAVE_ANSWER_GUARD_MIN_SIMILARITY",
        "BRAINWAVE_ANSWER_GUARD_NOVEL_MATERIAL_RATIO",
        "BRAINWAVE_ANSWER_GUARD_SIMILARITY_MAX_CHARS",
        "BRAINWAVE_AUDIO_CAPTURE_QUEUE_MAX_FRAMES",
        "BRAINWAVE_AUDIO_SPOOL_DIR",
        "BRAINWAVE_COMPLETE_SOUND",
        "BRAINWAVE_ENFORCE_SINGLE_INSTANCE",
        "BRAINWAVE_ERROR_SOUND",
        "BRAINWAVE_HISTORY_DIR",
        "BRAINWAVE_IDLE_WS_RECONNECT_SEC",
        "BRAINWAVE_INCLUDE_INSTRUCTIONS_EACH_RESPONSE",
        "BRAINWAVE_INPUT_TRANSCRIPT_GRACE_SEC",
        "BRAINWAVE_INPUT_TRANSCRIPT_REPLACEMENT_MIN_DELTA_CHARS",
        "BRAINWAVE_INPUT_TRANSCRIPT_REPLACEMENT_MIN_RATIO",
        "BRAINWAVE_KEEP_PROVIDER_SESSION",
        "BRAINWAVE_LATENCY_PRESET",
        "BRAINWAVE_MAX_OUTPUT_TOKENS",
        "BRAINWAVE_MAX_UPLOAD_BACKLOG_BYTES",
        "BRAINWAVE_NO_SPEECH_FLOOR_MARGIN_DB",
        "BRAINWAVE_NO_SPEECH_GUARD",
        "BRAINWAVE_NO_SPEECH_HB300_MIN_RATIO",
        "BRAINWAVE_NO_SPEECH_MIN_ACTIVE_DBFS",
        "BRAINWAVE_NO_SPEECH_MIN_RUN_FRAMES",
        "BRAINWAVE_NO_SPEECH_NEARFIELD_P90_DBFS",
        "BRAINWAVE_NO_SPEECH_PEAK_FLOOR_DBFS",
        "BRAINWAVE_PASSTHROUGH_WITHOUT_MARKER",
        "BRAINWAVE_PROCESSING_HARD_TIMEOUT_SEC",
        "BRAINWAVE_PROCESSING_TIMEOUT_SEC",
        "BRAINWAVE_PROVIDER_INIT_MAX_ATTEMPTS",
        "BRAINWAVE_PROVIDER_INIT_RETRY_DELAY_SEC",
        "BRAINWAVE_PROVIDER_SESSION_MAX_AGE_SEC",
        "BRAINWAVE_PROVIDER_SESSION_MAX_TURNS",
        "BRAINWAVE_PYAUDIO_CHUNK_SIZE",
        "BRAINWAVE_RECENT_AUDIO_CACHE_DIR",
        "BRAINWAVE_RECENT_AUDIO_CACHE_ENABLED",
        "BRAINWAVE_RECENT_AUDIO_CACHE_LIMIT",
        "BRAINWAVE_RESPONSE_FINALIZE_TIMEOUT_SEC",
        "BRAINWAVE_SESSION_START_BACKOFF_BASE_SEC",
        "BRAINWAVE_SESSION_START_BACKOFF_CAP_SEC",
        "BRAINWAVE_SESSION_START_MAX_ATTEMPTS",
        "BRAINWAVE_START_SOUND",
        "BRAINWAVE_STOP_SESSION_CONNECT_DEADLINE_SEC",
        "BRAINWAVE_STOP_SOUND",
        "BRAINWAVE_STOP_TAIL_WAIT_GUARD_MS",
        "BRAINWAVE_STOP_TAIL_WAIT_MAX_MS",
        "BRAINWAVE_STOP_TAIL_WAIT_MIN_MS",
        "BRAINWAVE_SUSPICIOUS_INPUT_TRANSCRIPT_GRACE_SEC",
        "BRAINWAVE_SUSPICIOUS_MARKER_AUDIO_SEC",
        "BRAINWAVE_SUSPICIOUS_MARKER_EMITTED_CHARS",
        "BRAINWAVE_UPLOAD_CHUNK_MS",
        "BRAINWAVE_VERBOSE_SERVER_LOG",
        "OPENAI_REALTIME_MODALITIES",
        "OPENAI_REALTIME_MODEL",
    }
)
_WHISPER_DOTENV_ALLOWED_KEYS = frozenset(
    {
        "WHISPERWAVE_DEBUG_LOG",
        "WHISPERWAVE_DELAY",
        "WHISPERWAVE_FORCE_RECONNECT_AFTER_MS",
        "WHISPERWAVE_KEEP_PROVIDER_SESSION",
        "WHISPERWAVE_LANGUAGE",
        "WHISPERWAVE_MAX_TURN_AUDIO_BYTES",
        "WHISPERWAVE_MODALITIES",
        "WHISPERWAVE_MODEL",
        "WHISPERWAVE_NOISE_REDUCTION",
        "WHISPERWAVE_PORT",
        "WHISPERWAVE_PROVIDER_EVENT_METRICS",
        "WHISPERWAVE_PROVIDER_INIT_MAX_ATTEMPTS",
        "WHISPERWAVE_PROVIDER_INIT_RETRY_DELAY_SEC",
        "WHISPERWAVE_PROVIDER_SESSION_MAX_AGE_SEC",
        "WHISPERWAVE_PROVIDER_SESSION_MAX_TURNS",
        "WHISPERWAVE_SERVER_AUDIO_CACHE_DIR",
        "WHISPERWAVE_SERVER_AUDIO_CACHE_ENABLED",
        "WHISPERWAVE_SERVER_AUDIO_CACHE_LIMIT",
        "WHISPERWAVE_SERVER_AUDIO_FILENAME_PREFIX",
        "WHISPERWAVE_TRANSCRIPTION_ACK_GRACE_SEC",
        "WHISPERWAVE_TRANSCRIPTION_FAILURE_ROTATE_THRESHOLD",
        "WHISPERWAVE_TRANSCRIPTION_FINALIZE_TIMEOUT_SEC",
        "WHISPERWAVE_VERBOSE_SERVER_LOG",
    }
)
_DOTENV_ALLOWED_KEYS = _ECHO_DOTENV_ALLOWED_KEYS | _WHISPER_DOTENV_ALLOWED_KEYS
_DOTENV_SECRET_MARKERS = (
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "PRIVATE_KEY",
)

_supervisor = None
_cleanup_lock = threading.Lock()
_cleanup_done = False


def _pid_command(pid: int):
    try:
        output = subprocess.check_output(
            ["/bin/ps", "-p", str(pid), "-o", "command="],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.strip() or None


def _command_matches_shell(command: str) -> bool:
    """Only the exact EchoWave launcher may own the shell PID file."""
    return PROJECT_DIR in command and os.path.join(PROJECT_DIR, "launcher.py") in command


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_shell_pid():
    try:
        return int(Path(SHELL_PID_FILE).read_text(encoding="ascii").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _write_shell_pid(pid: int) -> None:
    target = Path(SHELL_PID_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
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


def _remove_own_shell_pid() -> None:
    if _read_shell_pid() != os.getpid():
        return
    try:
        os.unlink(SHELL_PID_FILE)
    except FileNotFoundError:
        pass


def _should_enforce_single_instance() -> bool:
    return os.getenv("BRAINWAVE_ENFORCE_SINGLE_INSTANCE") == "1"


def _stop_previous_shell() -> None:
    """Stop one validated prior shell PID; never scan or match basenames."""
    if not _should_enforce_single_instance():
        return
    pid = _read_shell_pid()
    if pid is None or pid in {os.getpid(), os.getppid()}:
        return
    command = _pid_command(pid)
    if not command or not _command_matches_shell(command):
        print(f"[Launcher] Ignoring stale/unowned shell PID {pid}")
        return
    print(f"[Launcher] Stopping previous desktop shell PID {pid}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            return
        time.sleep(0.05)
    if _pid_is_alive(pid):
        os.kill(pid, signal.SIGKILL)


def _cleanup() -> None:
    global _cleanup_done
    with _cleanup_lock:
        if _cleanup_done:
            return
        _cleanup_done = True
    supervisor = _supervisor
    if supervisor is not None:
        print("[Launcher] Cleaning up exact backend children...")
        supervisor.stop_all()
    _remove_own_shell_pid()


def _signal_handler(signum, _frame):
    sig_name = signal.Signals(signum).name
    print(f"[Launcher] Received {sig_name}, shutting down...")
    _cleanup()
    # rumps' NSApplication loop can swallow SystemExit.
    os._exit(128 + signum)


def _parse_dotenv_assignment(raw_line: str):
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_env(env_paths=None):
    """Load non-secret local config without overriding injected environment."""
    if env_paths is None:
        env_paths = [
            os.path.join(LEGACY_APP_SUPPORT_DIR, ".env"),
            os.path.join(PROJECT_DIR, ".env"),
            os.path.join(RESOURCE_DIR, ".env"),
        ]
    for env_path in env_paths:
        if not os.path.exists(env_path):
            continue
        loaded = 0
        with open(env_path, encoding="utf-8") as handle:
            for raw_line in handle:
                assignment = _parse_dotenv_assignment(raw_line)
                if assignment is None:
                    continue
                key, value = assignment
                if not key or any(marker in key.upper() for marker in _DOTENV_SECRET_MARKERS):
                    continue
                if key not in _DOTENV_ALLOWED_KEYS or key in os.environ:
                    continue
                os.environ[key] = value
                loaded += 1
        print(f"[Launcher] Loaded {loaded} non-secret config keys from {env_path}")
        return
    print("[Launcher] WARNING: No local .env configuration found")


def preflight_check() -> bool:
    """Validate shared secret presence and print only non-secret effective config."""
    ok = True
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key.startswith("your_"):
        print("[Preflight] FATAL: OPENAI_API_KEY is missing or placeholder")
        ok = False

    session_params = {
        "BRAINWAVE_KEEP_PROVIDER_SESSION": ("1", "reuse provider session"),
        "BRAINWAVE_PROVIDER_SESSION_MAX_TURNS": ("8", "session max turns"),
        "BRAINWAVE_PROVIDER_SESSION_MAX_AGE_SEC": ("7200", "session max age"),
        "BRAINWAVE_INCLUDE_INSTRUCTIONS_EACH_RESPONSE": ("1", "per-response instructions"),
        "BRAINWAVE_IDLE_WS_RECONNECT_SEC": (None, "idle reconnect threshold"),
    }
    print("[Preflight] Session configuration")
    for name, (default, description) in session_params.items():
        effective = os.getenv(name, default)
        source = "env" if name in os.environ else "default"
        display = effective if effective is not None else "(inherit MAX_AGE)"
        print(f"[Preflight]   {name} = {display} ({source}) - {description}")

    print(
        "[Preflight]   OPENAI_REALTIME_MODEL = "
        f"{os.getenv('OPENAI_REALTIME_MODEL', 'gpt-realtime-2.1-mini')}"
    )
    print(
        "[Preflight]   WHISPERWAVE_MODEL = "
        f"{os.getenv('WHISPERWAVE_MODEL', 'gpt-4o-transcribe')}"
    )
    print("[Preflight] OK" if ok else "[Preflight] FAILED")
    return ok


def run_server_foreground():
    """Compatibility entry point for the stable Echo backend only."""
    os.chdir(RESOURCE_DIR)
    from realtime_server import app
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=23456)


def run_menubar_directly(supervisor, selection_store, startup_decision):
    if RESOURCE_DIR not in sys.path:
        sys.path.insert(0, RESOURCE_DIR)
    os.chdir(RESOURCE_DIR)
    print(
        "[Launcher] Starting one menubar/hotkey owner with "
        f"requested={startup_decision.requested_backend} "
        f"effective={startup_decision.effective_backend}"
    )
    from ime_menubar import run_menubar

    run_menubar(
        supervisor=supervisor,
        selection_store=selection_store,
        startup_decision=startup_decision,
    )


def main():
    global _supervisor

    print("[Launcher] Starting EchoWave desktop shell...")
    load_env()
    if not preflight_check():
        print("[Launcher] Aborting due to preflight failures")
        raise SystemExit(1)
    if "--server" in sys.argv:
        run_server_foreground()
        return

    _stop_previous_shell()
    _write_shell_pid(os.getpid())
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_cleanup)

    descriptors = build_default_descriptors(PROJECT_DIR)
    _supervisor = BackendSupervisor(descriptors)
    selection_store = BackendSelectionStore()
    coordinator = BackendCoordinator(_supervisor, selection_store)
    startup_decision = coordinator.start_initial()

    try:
        run_menubar_directly(_supervisor, selection_store, startup_decision)
    except KeyboardInterrupt:
        print("[Launcher] Interrupted")
    except Exception as exc:
        print(f"[Launcher] Error: {exc}")
        traceback.print_exc()
        raise
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
