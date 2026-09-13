import ast
from pathlib import Path

from backend_runtime import (
    BACKEND_ECHO,
    BACKEND_WHISPER,
    WIRE_CONTRACT_NAME,
    WIRE_CONTRACT_VERSION,
    build_default_descriptors,
)
from whisper_backend import realtime_server


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_ROOT = PROJECT_ROOT / "whisper_backend"


def test_whisper_sidecar_has_no_desktop_owner_artifacts_or_dependencies():
    assert not (SIDECAR_ROOT / "ime_menubar.py").exists()
    assert not (SIDECAR_ROOT / "launcher.py").exists()
    requirements = (SIDECAR_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for forbidden in ("pyaudio", "rumps", "Quartz", "pynput"):
        assert forbidden not in requirements


def test_whisper_provider_modules_use_package_relative_internal_imports():
    internal = {
        "audio_persistence",
        "config",
        "dotenv_loader",
        "realtime_client_base",
        "realtime_server",
        "realtime_text_utils",
        "transcript_merge",
        "transcription_prompt",
        "whisper_realtime_client",
    }
    for path in SIDECAR_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name in internal for alias in node.names), path
            if isinstance(node, ast.ImportFrom) and node.module in internal:
                assert node.level > 0, path


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


def test_whisper_health_is_versioned_and_backend_identified():
    payload = __import__("asyncio").run(realtime_server.health_check())
    assert payload["status"] == "ok"
    assert payload["backend"] == BACKEND_WHISPER
    assert payload["contract"] == {
        "name": WIRE_CONTRACT_NAME,
        "version": WIRE_CONTRACT_VERSION,
    }
