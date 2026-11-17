"""Shared dictation transcript store (transcripts.jsonl).

One JSONL file holds every final dictation text: the Mac IME appends its turns
and the micwave_phone service (another process) appends phone turns. Records
without a ``source`` field are Mac turns; phone turns carry ``source: "phone"``.

Concurrency contract: every record is written by a single ``os.write`` on an
``O_APPEND`` descriptor, so lines from the two writers never interleave.
Readers only consume bytes up to the last newline, so a reader that races a
writer never parses half a line.

Env:
  BRAINWAVE_HISTORY_DIR  store directory; default
                         ~/Library/Application Support/Brainwave IME
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Iterable, Optional

HISTORY_DIR_ENV = "BRAINWAVE_HISTORY_DIR"
HISTORY_FILE_NAME = "transcripts.jsonl"
TS_FORMAT = "%Y-%m-%d %H:%M:%S"

SOURCE_PHONE = "phone"
SOURCE_MAC = "mac"

WINDOW_TODAY = "today"
WINDOW_7D = "7d"
WINDOW_ALL = "all"
WINDOWS = (WINDOW_TODAY, WINDOW_7D, WINDOW_ALL)

_SOURCE_TAGS = {SOURCE_PHONE: "\U0001F4F1", SOURCE_MAC: "\U0001F4BB"}


def history_dir() -> str:
    custom_dir = os.getenv(HISTORY_DIR_ENV)
    if custom_dir:
        return os.path.expanduser(custom_dir)
    return os.path.join(
        os.path.expanduser("~"), "Library", "Application Support", "Brainwave IME"
    )


def history_file_path() -> str:
    return os.path.join(history_dir(), HISTORY_FILE_NAME)


def now_ts() -> str:
    return datetime.now().strftime(TS_FORMAT)


def append_record(path: str, record: dict) -> None:
    """Append one record as one line with a single write syscall.

    If the file does not end in a newline (a writer died mid-line, a short
    write, a hand edit), the same write starts with one, so the new record is
    not glued onto the torn line. After a short write a newline is written on a
    best-effort basis for the same reason. Text that cannot be encoded as UTF-8
    (lone surrogates) is kept with replacement characters rather than lost.

    New files get mode 0644 (subject to umask); an existing file keeps its mode.
    Raises OSError on failure, including a short write.
    """
    serialized = json.dumps(record, ensure_ascii=False) + "\n"
    try:
        line = serialized.encode("utf-8")
    except UnicodeEncodeError:
        line = serialized.encode("utf-8", errors="replace")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        size = os.fstat(fd).st_size
        if size > 0 and os.pread(fd, 1, size - 1) != b"\n":
            line = b"\n" + line
        written = os.write(fd, line)
        if written != len(line):
            try:
                os.write(fd, b"\n")
            except OSError:
                pass  # best effort: the short write is reported below either way
            raise OSError(f"short append to transcript store: {written}/{len(line)} bytes")
    finally:
        os.close(fd)


class TranscriptStoreReader:
    """Incremental reader: the first refresh reads the whole file, later ones
    read only bytes appended since. A replaced or truncated file is reread."""

    def __init__(self, path: str):
        self._path = path
        self._records: list[dict] = []
        self._offset = 0
        self._identity: Optional[tuple[int, int]] = None

    @property
    def path(self) -> str:
        return self._path

    def refresh(self) -> list[dict]:
        """Return all valid records, oldest first (file order)."""
        try:
            st = os.stat(self._path)
        except FileNotFoundError:
            self._reset(None)
            return self._records
        identity = (st.st_dev, st.st_ino)
        if identity != self._identity or st.st_size < self._offset:
            self._reset(identity)
        if st.st_size > self._offset:
            with open(self._path, "rb") as handle:
                if self._offset > 0:
                    # Consumed bytes always end in a newline; anything else
                    # means the file was truncated in place and regrown.
                    handle.seek(self._offset - 1)
                    if handle.read(1) != b"\n":
                        self._reset(identity)
                handle.seek(self._offset)
                chunk = handle.read(st.st_size - self._offset)
            end = chunk.rfind(b"\n")
            if end >= 0:
                self._records.extend(_parse_lines(chunk[: end + 1]))
                self._offset += end + 1
        return self._records

    def _reset(self, identity: Optional[tuple[int, int]]) -> None:
        self._records = []
        self._offset = 0
        self._identity = identity


def _parse_lines(data: bytes) -> Iterable[dict]:
    for raw in data.split(b"\n"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        text = record.get("text")
        if not isinstance(text, str) or not text:
            continue
        yield record


def record_source(record: dict) -> str:
    return SOURCE_PHONE if record.get("source") == SOURCE_PHONE else SOURCE_MAC


def parse_ts(record: dict) -> Optional[datetime]:
    ts = record.get("ts")
    if not isinstance(ts, str):
        return None
    try:
        return datetime.strptime(ts, TS_FORMAT)
    except ValueError:
        return None


def filter_records(
    records: list[dict],
    query: str = "",
    window: str = WINDOW_ALL,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Newest first by ts (file order breaks ties). ``query`` terms are
    whitespace-separated and must all appear, case-insensitively, in the text or
    the ts string (so "2026-11-09" or "09:3" match by time). ``window`` keeps
    today's records or those from the last 7 calendar days including today;
    records without a parseable ts only show under "all"."""
    if window not in WINDOWS:
        raise ValueError(f"unknown window: {window!r}")
    now = now or datetime.now()
    earliest = None
    if window == WINDOW_TODAY:
        earliest = datetime(now.year, now.month, now.day).strftime(TS_FORMAT)
    elif window == WINDOW_7D:
        earliest = (datetime(now.year, now.month, now.day) - timedelta(days=6)).strftime(TS_FORMAT)
    terms = query.casefold().split()

    # TS_FORMAT strings order like the times they encode, so filtering and
    # sorting compare strings; strptime per record would cost ~50 ms per
    # keystroke on an 8k-record store.
    matched = []
    for position, record in enumerate(records):
        ts = record.get("ts")
        sort_ts = ts if _looks_like_ts(ts) else ""
        if earliest is not None and sort_ts < earliest:
            continue
        if terms:
            haystack = f"{record.get('text', '')} {ts or ''}".casefold()
            if not all(term in haystack for term in terms):
                continue
        matched.append((sort_ts, position, record))
    matched.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in matched]


def _looks_like_ts(ts) -> bool:
    return (
        isinstance(ts, str)
        and len(ts) == 19
        and ts[4] == "-"
        and ts[7] == "-"
        and ts[10] == " "
        and ts[13] == ":"
        and ts[16] == ":"
    )


def format_recent_title(record: dict) -> str:
    """One panel row: '<MM-DD HH:MM> | <source tag> [backend] preview'."""
    raw_text = str(record.get("text", "")).strip()
    preview = " ".join(raw_text.splitlines()).strip()
    if len(preview) > 80:
        preview = preview[:77] + "..."
    when = parse_ts(record)
    if when is not None:
        time_label = when.strftime("%m-%d %H:%M")
    else:
        time_label = str(record.get("ts", ""))[:16]
    prefix = f"{time_label} | " if time_label else ""
    tag = _SOURCE_TAGS[record_source(record)]
    backend = record.get("backend")
    backend_prefix = f"[{backend}] " if backend else ""
    return f"{prefix}{tag} {backend_prefix}{preview or '(empty)'}"
