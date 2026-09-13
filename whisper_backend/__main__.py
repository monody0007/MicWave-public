"""Run the isolated Whisper transcription sidecar on its dedicated port."""

from __future__ import annotations

import os
from pathlib import Path

from .dotenv_loader import load_into_environ


def main() -> None:
    config_path = Path(
        os.getenv(
            "ECHOWAVE_WHISPER_CONFIG_FILE",
            str(
                Path.home()
                / "Library"
                / "Application Support"
                / "EchoWave"
                / "whisper"
                / "backend.env"
            ),
        )
    ).expanduser()
    # Only WHISPERWAVE_* non-secret keys are accepted. OPENAI_API_KEY must be
    # inherited from the secure launcher environment, never loaded or printed.
    load_into_environ(config_path, override=False)

    import uvicorn

    from .realtime_server import app

    port = int(os.getenv("WHISPERWAVE_PORT", "23459"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()

