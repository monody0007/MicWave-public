"""dotenv loader tests (task 0436 Phase 2 R2, H4/M2).

The loader must preserve spaces, refuse secrets, honor an allowlist, and never
overwrite an already-set variable (so an environment-injected OPENAI_API_KEY wins over
a local placeholder — the H4 second-run bug).
"""
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_backend import dotenv_loader

CACHE_DIR_WITH_SPACES = "~/Library/Application Support/EchoWave/whisper/recent_audio"


def _write_env(tmp_path, body):
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_preserves_spaces_in_values(tmp_path):
    path = _write_env(tmp_path, f"WHISPERWAVE_SERVER_AUDIO_CACHE_DIR={CACHE_DIR_WITH_SPACES}\n")
    pairs = dict(dotenv_loader.parse_env_file(path))
    # The full path ("Application Support" carries a space) survives — not
    # truncated at the first space the way `export $(... | xargs)` did (M2).
    assert pairs["WHISPERWAVE_SERVER_AUDIO_CACHE_DIR"] == CACHE_DIR_WITH_SPACES
    assert pairs["WHISPERWAVE_SERVER_AUDIO_CACHE_DIR"].count(" ") == 1


def test_secret_keys_are_never_loaded(tmp_path):
    path = _write_env(
        tmp_path,
        "OPENAI_API_KEY=sk-placeholder\n"
        "WHISPERWAVE_SECRET_TOKEN=nope\n"
        "WHISPERWAVE_DELAY=low\n",
    )
    keys = [k for k, _ in dotenv_loader.parse_env_file(path)]
    assert "OPENAI_API_KEY" not in keys
    assert "WHISPERWAVE_SECRET_TOKEN" not in keys
    assert "WHISPERWAVE_DELAY" in keys


def test_allowlist_rejects_foreign_prefixes(tmp_path):
    path = _write_env(tmp_path, "RANDOM_KEY=1\nMICWAVE_DELAY=x\nWHISPERWAVE_DELAY=low\n")
    keys = [k for k, _ in dotenv_loader.parse_env_file(path)]
    assert keys == ["WHISPERWAVE_DELAY"]


def test_existing_value_wins_even_when_legacy_override_is_requested(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPERWAVE_DELAY", "RUNTIME_SENTINEL")
    path = _write_env(tmp_path, "WHISPERWAVE_DELAY=placeholder_from_env\n")
    loaded = dotenv_loader.load_into_environ(path, override=True)
    assert os.environ["WHISPERWAVE_DELAY"] == "RUNTIME_SENTINEL"
    assert "WHISPERWAVE_DELAY" not in loaded


def test_second_run_placeholder_never_shadows_injected_secret(tmp_path, monkeypatch):
    # Second run: the secure shell injected the real key; the loader must not
    # overwrite it — even if a stale placeholder somehow sits in .env (H4).
    monkeypatch.setenv("OPENAI_API_KEY", "secure-runtime-sentinel")
    path = _write_env(
        tmp_path,
        "OPENAI_API_KEY=your_openai_api_key_here\nWHISPERWAVE_DELAY=low\n",
    )
    dotenv_loader.load_into_environ(path, override=False)
    assert os.environ["OPENAI_API_KEY"] == "secure-runtime-sentinel"


def test_finite_parser_normalizes_export_quotes_and_rejects_unknown_keys(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "secure-runtime-sentinel")
    monkeypatch.setenv("WHISPERWAVE_DELAY", "existing-delay")
    for key in (
        "WHISPERWAVE_ACCESS_KEY",
        "BRAINWAVE_AUTH_HEADER",
        "WHISPERWAVE_UNKNOWN_SETTING",
        "WHISPERWAVE_MODEL",
        "WHISPERWAVE_PORT",
        "WHISPERWAVE_LANGUAGE",
    ):
        monkeypatch.delenv(key, raising=False)
    path = _write_env(
        tmp_path,
        "OPENAI_API_KEY=dotenv-placeholder\n"
        "WHISPERWAVE_ACCESS_KEY=must_not_load\n"
        "BRAINWAVE_AUTH_HEADER=must_not_load\n"
        "WHISPERWAVE_UNKNOWN_SETTING=must_not_load\n"
        ' WHISPERWAVE_MODEL = "test-transcribe-model" \n'
        " export WHISPERWAVE_PORT = '45678' \n"
        " WHISPERWAVE_LANGUAGE = zh \n"
        " WHISPERWAVE_DELAY = high \n",
    )

    loaded = dotenv_loader.load_into_environ(path, override=False)

    assert os.environ["OPENAI_API_KEY"] == "secure-runtime-sentinel"
    assert "WHISPERWAVE_ACCESS_KEY" not in os.environ
    assert "BRAINWAVE_AUTH_HEADER" not in os.environ
    assert "WHISPERWAVE_UNKNOWN_SETTING" not in os.environ
    assert os.environ["WHISPERWAVE_MODEL"] == "test-transcribe-model"
    assert os.environ["WHISPERWAVE_PORT"] == "45678"
    assert os.environ["WHISPERWAVE_LANGUAGE"] == "zh"
    assert os.environ["WHISPERWAVE_DELAY"] == "existing-delay"
    assert set(loaded) == {
        "WHISPERWAVE_LANGUAGE",
        "WHISPERWAVE_MODEL",
        "WHISPERWAVE_PORT",
    }


def test_emit_shell_round_trips_spaces(tmp_path, capsys):
    path = _write_env(tmp_path, f"WHISPERWAVE_SERVER_AUDIO_CACHE_DIR={CACHE_DIR_WITH_SPACES}\n")
    dotenv_loader._emit_shell(path)
    out = capsys.readouterr().out.strip()
    # `export "$line"` in start.sh sets the full path from this single line.
    assert out == f"WHISPERWAVE_SERVER_AUDIO_CACHE_DIR={CACHE_DIR_WITH_SPACES}"
    _, _, value = out.partition("=")
    assert value == CACHE_DIR_WITH_SPACES
