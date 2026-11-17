"""Shared pytest isolation fixtures for micwave-core."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_local_stores(monkeypatch, tmp_path_factory):
    """Redirect transcript history and audio cache directories to temp dirs so
    no test ever touches ~/Library/Application Support."""
    store_dir = tmp_path_factory.mktemp("transcript_store")
    echo_cache = tmp_path_factory.mktemp("echo_audio_cache")
    whisper_cache = tmp_path_factory.mktemp("whisper_audio_cache")
    monkeypatch.setenv("BRAINWAVE_HISTORY_DIR", str(store_dir))
    monkeypatch.setenv("MICWAVE_SERVER_AUDIO_CACHE_DIR", str(echo_cache))
    monkeypatch.setenv("WHISPERWAVE_SERVER_AUDIO_CACHE_DIR", str(whisper_cache))
    return store_dir
