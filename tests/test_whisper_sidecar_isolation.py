import asyncio
from pathlib import Path

from backend_runtime import (
    BACKEND_ECHO,
    BACKEND_WHISPER,
    WIRE_CONTRACT_NAME,
    WIRE_CONTRACT_VERSION,
    build_default_descriptors,
)
from micwave_core import whisper_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_ROOT = PROJECT_ROOT / "whisper_backend"


def test_whisper_sidecar_has_no_desktop_owner_artifacts_or_dependencies():
    assert not (SIDECAR_ROOT / "ime_menubar.py").exists()
    assert not (SIDECAR_ROOT / "launcher.py").exists()
    requirements = (SIDECAR_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for forbidden in ("pyaudio", "rumps", "Quartz", "pynput"):
        assert forbidden not in requirements


def test_default_descriptors_lock_ports_and_separate_runtime_pid_log(tmp_path):
    descriptors = build_default_descriptors(
        PROJECT_ROOT,
        app_support_dir=tmp_path / "runtime",
        log_dir=tmp_path / "logs",
    )
    echo = descriptors[BACKEND_ECHO]
    whisper = descriptors[BACKEND_WHISPER]
    assert echo.port == 23456
    assert whisper.port == 23459
    assert echo.command[0] != whisper.command[0]
    assert echo.pid_file != whisper.pid_file
    assert echo.log_file != whisper.log_file
    assert echo.provider != whisper.provider
    assert echo.model != whisper.model
    assert echo.command[1].endswith("echo_backend_entry.py")
    assert whisper.command[1:] == ("-m", "micwave_core.whisper_server")


def test_whisper_health_is_versioned_and_backend_identified():
    payload = asyncio.run(whisper_server.health_check())
    assert payload["status"] == "ok"
    assert payload["backend"] == BACKEND_WHISPER
    assert payload["contract"] == {
        "name": WIRE_CONTRACT_NAME,
        "version": WIRE_CONTRACT_VERSION,
    }


def test_echo_desktop_contract_is_literal_echo(monkeypatch):
    import echo_backend_entry
    import launcher
    from micwave_core import whisper_adapter

    monkeypatch.setenv("ECHOWAVE_BACKEND_KEY", "retired_override_ignored")
    payload = asyncio.run(echo_backend_entry.desktop_contract())
    assert payload["backend"] == "echo"
    assert launcher._WHISPER_DOTENV_ALLOWED_KEYS is whisper_adapter.DOTENV_ALLOWED_KEYS
