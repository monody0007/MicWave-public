"""Space-preserving, allowlisted, secret-safe sidecar config loader.

Used by ``python -m whisper_backend`` before provider modules are imported.
Properties:

  * spaces preserved — values are split on the first '=' only; no shell
    word-splitting / `xargs`, so a path like
    "~/Library/Application Support/EchoWave/whisper/recent_audio" survives intact.
  * allowlisted — only explicitly enumerated settings consumed by this package
    are honored; unknown names are rejected even when they use its namespace.
  * secret-safe — credential-looking or otherwise unknown keys are absent from
    the finite set, so a local placeholder can never shadow an
    environment-injected OPENAI_API_KEY. Environment injection therefore
    has final precedence.
  * non-destructive — load_into_environ never overwrites an already-set
    variable. The legacy override argument is retained only for call compatibility.
"""
import os
import sys

DOTENV_ALLOWED_KEYS = frozenset(
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


def _is_allowed(key: str) -> bool:
    return key in DOTENV_ALLOWED_KEYS


def parse_env_file(path):
    """Return a list of (key, value) pairs, preserving spaces, allowlist-filtered.

    Skips comments / blanks / non-allowlisted keys. Strips one layer of matching
    surrounding quotes but keeps interior spaces. Missing file -> empty list.
    """
    pairs = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].lstrip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if not _is_allowed(key):
                    continue
                pairs.append((key, value))
    except FileNotFoundError:
        pass
    return pairs


def load_into_environ(path, override=False):
    """Load allowlisted keys while always preserving the existing environment.

    ``override`` remains accepted for compatibility, but existing secure and
    non-secret values have unconditional precedence over dotenv input.
    """
    loaded = []
    for key, value in parse_env_file(path):
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def _emit_shell(path):
    # Print KEY=VALUE lines for `while IFS= read -r line; do export "$line"; done`.
    # The consumer quotes the whole line, so embedded spaces are preserved and the
    # RHS is never re-evaluated (no command substitution / injection).
    for key, value in parse_env_file(path):
        sys.stdout.write(f"{key}={value}\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--emit-shell":
        _emit_shell(sys.argv[2])
