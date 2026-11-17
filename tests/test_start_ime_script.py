"""start_ime.sh behaviour in a sandbox: key injection and how micwave-core is provided.

The script is copied into a temp root whose venv interpreters are stubs: dependency import probes pass (so no
dependency install or brew runs), the editable-link probe fails (so the script takes its link path), `launcher.py`
reports which keys it was started with, and pip records its arguments.
LAUNCH_JOB_NAME makes the script `exec` the launcher in the foreground, so nothing is left running.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_STUB_PYTHON = """#!/bin/bash
# Echo only the tests' own values; anything else could be a real key reaching the test output.
show() { case "$1" in "") echo "<unset>" ;; from-*) echo "$1" ;; *) echo "<other>" ;; esac; }
case "$1" in
    -c) case "$2" in "import pyaudio"|"import fastapi"*) exit 0 ;; *) exit 1 ;; esac ;;
    launcher.py) echo "launcher OPENAI=$(show "$OPENAI_API_KEY") GOOGLE=$(show "$GOOGLE_API_KEY")"; exit 0 ;;
esac
exit 0
"""
_STUB_PIP = """#!/bin/bash
echo "pip $*" >> "$PIP_LOG"
"""


def _executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    shutil.copy2(PROJECT_ROOT / "start_ime.sh", root / "start_ime.sh")
    (root / "requirements.txt").write_text("numpy\n")
    (root / "whisper_backend").mkdir()
    (root / "whisper_backend" / "requirements.txt").write_text("numpy\n")
    for venv in (root / "venv", root / "whisper_backend" / "venv"):
        _executable(venv / "bin" / "python", _STUB_PYTHON)
        _executable(venv / "bin" / "pip", _STUB_PIP)
    return root


def _run(root: Path, **env: str) -> tuple[str, list[str]]:
    pip_log = root.parent / "pip.log"
    full_env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(root.parent),
        "LAUNCH_JOB_NAME": "test",
        "PIP_LOG": str(pip_log),
        **env,
    }
    proc = subprocess.run(["/bin/bash", str(root / "start_ime.sh")], cwd=root, env=full_env,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    pip_calls = pip_log.read_text().splitlines() if pip_log.exists() else []
    return proc.stdout, pip_calls


def test_local_hook_injects_keys_after_dotenv(sandbox):
    (sandbox / ".env").write_text("OPENAI_API_KEY=from-dotenv\nGOOGLE_API_KEY=from-dotenv\n")
    (sandbox / "start_ime.local.sh").write_text("export OPENAI_API_KEY=from-hook\n")
    (sandbox / "micwave_core").mkdir()
    (sandbox / "micwave_core" / "__init__.py").write_text("")

    out, _ = _run(sandbox)

    assert "launcher OPENAI=from-hook GOOGLE=from-dotenv" in out


def test_without_hook_keys_come_from_dotenv_or_caller(sandbox):
    (sandbox / ".env").write_text("GOOGLE_API_KEY=from-dotenv\n")
    (sandbox / "micwave_core").mkdir()
    (sandbox / "micwave_core" / "__init__.py").write_text("")

    out, _ = _run(sandbox, OPENAI_API_KEY="from-caller")

    assert "launcher OPENAI=from-caller GOOGLE=from-dotenv" in out


def test_vendored_micwave_core_is_used_in_place(sandbox):
    """Public snapshot layout: `micwave_core/` sits in the app root, so no editable link is installed."""
    (sandbox / "micwave_core").mkdir()
    (sandbox / "micwave_core" / "__init__.py").write_text("")

    _, pip_calls = _run(sandbox)

    assert not [c for c in pip_calls if " -e " in c], pip_calls


def test_sibling_micwave_core_checkout_is_linked_editable_into_both_venvs(sandbox):
    """Development layout: no vendored package, the micwave-core checkout next to the app is linked."""
    core = sandbox.parent / "micwave-core"
    core.mkdir()

    _, pip_calls = _run(sandbox)

    editable = [c for c in pip_calls if c.endswith(f"-e {sandbox}/../micwave-core")]
    assert len(editable) == 2, pip_calls


def test_explicit_core_dir_overrides_the_sibling_default(sandbox, tmp_path):
    core = tmp_path / "elsewhere" / "micwave-core"
    core.mkdir(parents=True)

    _, pip_calls = _run(sandbox, MICWAVE_CORE_DIR=str(core))

    assert len([c for c in pip_calls if c.endswith(f"-e {core}")]) == 2, pip_calls


def test_dependency_manifests_name_no_local_paths():
    for rel in ("requirements.txt", "whisper_backend/requirements.txt"):
        lines = (PROJECT_ROOT / rel).read_text(encoding="utf-8").splitlines()
        assert not [line for line in lines if line.strip().startswith("-e")], rel
