"""Single desktop-owner and exact PID lifecycle regressions."""

import signal

import launcher
from whisper_backend import dotenv_loader


def test_shell_command_matcher_accepts_only_echowave_launcher_path():
    own = (
        f"{launcher.PROJECT_DIR}/venv/bin/python "
        f"{launcher.PROJECT_DIR}/launcher.py"
    )
    old_whisper_app = (
        "/opt/whisperwave-app/venv/bin/python "
        "/opt/whisperwave-app/launcher.py"
    )
    bare_collision = "python launcher.py"
    assert launcher._command_matches_shell(own) is True
    assert launcher._command_matches_shell(old_whisper_app) is False
    assert launcher._command_matches_shell(bare_collision) is False


def test_single_owner_cleanup_targets_only_validated_pid_file_process(monkeypatch):
    target_pid = 4242
    own = f"{launcher.PROJECT_DIR}/venv/bin/python {launcher.PROJECT_DIR}/launcher.py"
    signals = []
    alive_checks = iter([True, False])

    monkeypatch.setenv("BRAINWAVE_ENFORCE_SINGLE_INSTANCE", "1")
    monkeypatch.setattr(launcher, "_read_shell_pid", lambda: target_pid)
    monkeypatch.setattr(launcher, "_pid_command", lambda pid: own if pid == target_pid else None)
    monkeypatch.setattr(launcher, "_pid_is_alive", lambda pid: next(alive_checks))
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    launcher._stop_previous_shell()
    assert signals == [(target_pid, signal.SIGTERM)]


def test_stale_or_foreign_pid_file_is_never_signalled(monkeypatch):
    signals = []
    monkeypatch.setenv("BRAINWAVE_ENFORCE_SINGLE_INSTANCE", "1")
    monkeypatch.setattr(launcher, "_read_shell_pid", lambda: 4343)
    monkeypatch.setattr(
        launcher,
        "_pid_command",
        lambda _pid: "/other/python /opt/whisperwave-app/launcher.py",
    )
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    launcher._stop_previous_shell()
    assert signals == []


def test_launcher_imports_one_menubar_owner_only():
    source = open(launcher.__file__, encoding="utf-8").read()
    assert source.count("from ime_menubar import run_menubar") == 1
    assert "WhisperWave/ime_menubar.py" not in source
    assert "whisper_backend.ime_menubar" not in source


def test_load_env_preserves_secure_injection_and_loads_only_non_secret_config(
    tmp_path,
    monkeypatch,
):
    """R2-M1-1: launcher parsing is finite, normalized, and non-destructive."""

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=your_placeholder_key",
                "WHISPERWAVE_ACCESS_KEY=must_not_load",
                "BRAINWAVE_AUTH_HEADER=must_not_load",
                "WHISPERWAVE_UNKNOWN_SETTING=must_not_load",
                "BRAINWAVE_UNKNOWN_SETTING=must_not_load",
                "RANDOM_KEY=must_not_load",
                " BRAINWAVE_KEEP_PROVIDER_SESSION = 0 ",
                ' WHISPERWAVE_MODEL = "test-transcribe-model" ',
                " export WHISPERWAVE_PORT = '34567' ",
                ' OPENAI_REALTIME_MODEL = "dotenv-echo-model" ',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secure-injected-sentinel")
    monkeypatch.setenv("OPENAI_REALTIME_MODEL", "existing-echo-model")
    monkeypatch.delenv("WHISPERWAVE_ACCESS_KEY", raising=False)
    monkeypatch.delenv("BRAINWAVE_AUTH_HEADER", raising=False)
    monkeypatch.delenv("WHISPERWAVE_UNKNOWN_SETTING", raising=False)
    monkeypatch.delenv("BRAINWAVE_UNKNOWN_SETTING", raising=False)
    monkeypatch.delenv("RANDOM_KEY", raising=False)
    monkeypatch.delenv("BRAINWAVE_KEEP_PROVIDER_SESSION", raising=False)
    monkeypatch.delenv("WHISPERWAVE_MODEL", raising=False)
    monkeypatch.delenv("WHISPERWAVE_PORT", raising=False)

    launcher.load_env(env_paths=[str(env_path)])

    assert launcher.os.environ["OPENAI_API_KEY"] == "secure-injected-sentinel"
    assert "WHISPERWAVE_ACCESS_KEY" not in launcher.os.environ
    assert "BRAINWAVE_AUTH_HEADER" not in launcher.os.environ
    assert "WHISPERWAVE_UNKNOWN_SETTING" not in launcher.os.environ
    assert "BRAINWAVE_UNKNOWN_SETTING" not in launcher.os.environ
    assert "RANDOM_KEY" not in launcher.os.environ
    assert launcher.os.environ["BRAINWAVE_KEEP_PROVIDER_SESSION"] == "0"
    assert launcher.os.environ["WHISPERWAVE_MODEL"] == "test-transcribe-model"
    assert launcher.os.environ["WHISPERWAVE_PORT"] == "34567"
    assert launcher.os.environ["OPENAI_REALTIME_MODEL"] == "existing-echo-model"


def test_launcher_whisper_dotenv_allowlist_matches_sidecar_contract():
    assert launcher._WHISPER_DOTENV_ALLOWED_KEYS == dotenv_loader.DOTENV_ALLOWED_KEYS
    assert dotenv_loader.DOTENV_ALLOWED_KEYS <= launcher._DOTENV_ALLOWED_KEYS
