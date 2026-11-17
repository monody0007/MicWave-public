"""Desktop-only contract wrapper around the shared micwave-core Echo backend."""

from __future__ import annotations

import os

import uvicorn

from backend_runtime import WIRE_CONTRACT_NAME, WIRE_CONTRACT_VERSION
from micwave_core.realtime_server import create_realtime_app

app = create_realtime_app()


@app.get("/api/v1/desktop-contract", include_in_schema=False)
async def desktop_contract():
    return {
        "backend": "echo",
        "contract": {
            "name": WIRE_CONTRACT_NAME,
            "version": WIRE_CONTRACT_VERSION,
        },
    }


def main() -> None:
    port = int(os.getenv("ECHOWAVE_ECHO_PORT", "23456"))
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
