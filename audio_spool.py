"""Bounded-memory, per-turn PCM16 WAV spool for the desktop capture owner."""

from __future__ import annotations

import errno
import os
import shutil
import tempfile
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional


class AudioSpoolError(RuntimeError):
    """A turn could not be durably spooled or archived."""


class TurnAudioSpool:
    """Incrementally write one turn to disk without retaining full PCM in RAM.

    The in-progress file is already a WAV. ``wave`` repairs its header on
    ``seal()``, after which archives are created with an atomic hard-link+rename
    on the normal same-volume Application Support layout.  A streaming copy is
    used only when a destination lives on another filesystem.
    """

    def __init__(
        self,
        directory: os.PathLike[str] | str,
        *,
        turn_id: Optional[int],
        sample_rate: int,
        channels: int = 1,
        sample_width: int = 2,
    ):
        if sample_rate <= 0 or channels <= 0 or sample_width <= 0:
            raise ValueError("sample_rate, channels, and sample_width must be positive")
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.turn_id = turn_id
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sample_width = int(sample_width)
        self.frame_width = self.channels * self.sample_width
        fd, temp_name = tempfile.mkstemp(
            prefix=f".turn-{turn_id if turn_id is not None else 'unknown'}-",
            suffix=".wav.part",
            dir=str(self.directory),
        )
        os.close(fd)
        self.path = Path(temp_name)
        self.byte_count = 0
        self._sealed = False
        self._cleaned = False
        self._lock = threading.RLock()
        try:
            self._wave = wave.open(str(self.path), "wb")
            self._wave.setnchannels(self.channels)
            self._wave.setsampwidth(self.sample_width)
            self._wave.setframerate(self.sample_rate)
        except Exception:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            raise

    @property
    def sealed(self) -> bool:
        return self._sealed

    @property
    def frame_count(self) -> int:
        return self.byte_count // self.frame_width

    def append(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        if len(pcm_bytes) % self.frame_width:
            raise AudioSpoolError(
                f"unaligned PCM chunk: {len(pcm_bytes)} bytes for frame width {self.frame_width}"
            )
        with self._lock:
            if self._sealed or self._cleaned:
                raise AudioSpoolError("cannot append to a sealed/cleaned turn spool")
            try:
                self._wave.writeframesraw(pcm_bytes)
                self.byte_count += len(pcm_bytes)
            except Exception as exc:
                raise AudioSpoolError(f"failed to append PCM to {self.path}: {exc}") from exc

    def seal(self) -> Path:
        with self._lock:
            if self._cleaned:
                raise AudioSpoolError("cannot seal a cleaned turn spool")
            if self._sealed:
                return self.path
            try:
                self._wave.close()
                with self.path.open("rb") as handle:
                    os.fsync(handle.fileno())
            except Exception as exc:
                raise AudioSpoolError(f"failed to seal turn spool {self.path}: {exc}") from exc
            self._sealed = True
            return self.path

    def archive_wav(
        self,
        directory: os.PathLike[str] | str,
        *,
        outcome: str,
        prefix: str = "",
    ) -> Path:
        source = self.seal()
        target_dir = Path(directory).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_outcome = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(outcome)
        ).strip("_") or "unknown"
        turn_label = f"T{self.turn_id}" if self.turn_id is not None else "Tunknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = target_dir / f"{prefix}{timestamp}_{turn_label}_{safe_outcome}.wav"
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(target_dir),
        )
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            temp_path.unlink()
            try:
                os.link(source, temp_path)
            except OSError as exc:
                if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                    raise
                with source.open("rb") as src, temp_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                    dst.flush()
                    os.fsync(dst.fileno())
            os.replace(temp_path, destination)
            return destination
        except Exception as exc:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            raise AudioSpoolError(f"failed to archive turn spool to {destination}: {exc}") from exc

    def cleanup(self) -> None:
        with self._lock:
            if self._cleaned:
                return
            if not self._sealed:
                try:
                    self._wave.close()
                except Exception:
                    pass
                self._sealed = True
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._cleaned = True

    def __enter__(self) -> "TurnAudioSpool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

