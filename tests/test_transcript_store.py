import json
import os
import pathlib
import stat
import sys
import threading
from datetime import datetime

import pytest


from micwave_core import transcript_store
from micwave_core.transcript_store import (
    TranscriptStoreReader,
    append_record,
    filter_records,
    format_recent_title,
    history_file_path,
    record_source,
)


def _lines(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text("utf-8").splitlines()]


def test_history_file_path_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINWAVE_HISTORY_DIR", str(tmp_path / "h"))
    assert history_file_path() == str(tmp_path / "h" / "transcripts.jsonl")


def test_append_record_writes_one_line_with_one_syscall(monkeypatch, tmp_path):
    path = tmp_path / "store" / "transcripts.jsonl"
    calls = []
    real_write = os.write

    def counting_write(fd, data):
        calls.append(len(data))
        return real_write(fd, data)

    monkeypatch.setattr(transcript_store.os, "write", counting_write)
    record = {"ts": "2026-11-10 09:00:00", "text": "中文 line\nsecond" * 2000}
    append_record(str(path), record)

    assert len(calls) == 1
    assert _lines(path) == [record]


def test_append_record_keeps_existing_file_mode(tmp_path):
    path = tmp_path / "transcripts.jsonl"
    path.write_text("", encoding="utf-8")
    os.chmod(path, 0o600)
    append_record(str(path), {"ts": "2026-11-10 09:00:00", "text": "a"})
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_append_record_short_write_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_store.os, "write", lambda fd, data: len(data) - 1)
    with pytest.raises(OSError):
        append_record(str(tmp_path / "t.jsonl"), {"ts": "x", "text": "abc"})


def test_concurrent_appends_from_threads_never_interleave(tmp_path):
    path = str(tmp_path / "transcripts.jsonl")
    big = "x" * 70000

    def worker(n):
        for i in range(20):
            append_record(path, {"ts": "2026-11-10 09:00:00", "text": f"{n}-{i}-{big}"})

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    records = _lines(path)
    assert len(records) == 80
    assert all(r["text"].endswith(big) for r in records)


def test_reader_incremental_skips_bad_lines_and_waits_for_newline(tmp_path):
    path = tmp_path / "transcripts.jsonl"
    path.write_bytes(
        b'{"ts": "2026-11-10 09:00:00", "text": "one"}\n'
        b"not json\n"
        b'{"ts": "2026-11-10 09:00:01", "text": ""}\n'
        b'["list"]\n'
        b'{"ts": "2026-11-10 09:00:02", "text": "par'
    )
    reader = TranscriptStoreReader(str(path))
    assert [r["text"] for r in reader.refresh()] == ["one"]

    with open(path, "ab") as handle:
        handle.write(b'tial"}\n')
    assert [r["text"] for r in reader.refresh()] == ["one", "partial"]

    # Rewrite already-consumed bytes in place (same inode, same size): an
    # incremental reader does not see this, a full rereader would.
    with open(path, "r+b") as handle:
        handle.write(b'{"ts": "2026-11-10 09:00:00", "text": "ONE"}')
    append_record(str(path), {"ts": "2026-11-10 09:00:03", "text": "three", "source": "phone"})
    assert [r["text"] for r in reader.refresh()] == ["one", "partial", "three"]


def test_reader_rereads_replaced_or_truncated_file(tmp_path):
    path = tmp_path / "transcripts.jsonl"
    append_record(str(path), {"ts": "2026-11-10 09:00:00", "text": "old-1"})
    append_record(str(path), {"ts": "2026-11-10 09:00:01", "text": "old-2"})
    reader = TranscriptStoreReader(str(path))
    assert len(reader.refresh()) == 2

    path.write_text(json.dumps({"ts": "2026-11-10 10:00:00", "text": "new"}) + "\n", encoding="utf-8")
    assert [r["text"] for r in reader.refresh()] == ["new"]

    path.unlink()
    assert reader.refresh() == []


def _rec(ts, text, **extra):
    return {"ts": ts, "text": text, **extra}


NOW = datetime(2026, 11, 10, 12, 0, 0)


def test_filter_sorts_newest_first_by_ts_not_file_order():
    records = [
        _rec("2026-11-10 09:00:00", "mac today"),
        _rec("2026-11-10 11:00:00", "phone today", source="phone"),
        _rec("2026-11-01 08:00:00", "backfilled old", source="phone", backfilled=True),
    ]
    out = filter_records(records, now=NOW)
    assert [r["text"] for r in out] == ["phone today", "mac today", "backfilled old"]


def test_filter_equal_ts_keeps_later_line_first():
    records = [_rec("2026-11-10 09:00:00", "first"), _rec("2026-11-10 09:00:00", "second")]
    assert [r["text"] for r in filter_records(records, now=NOW)] == ["second", "first"]


def test_filter_windows():
    records = [
        _rec("2026-11-10 00:00:00", "today midnight"),
        _rec("2026-11-09 23:59:59", "yesterday"),
        _rec("2026-11-04 00:00:00", "six days ago"),
        _rec("2026-11-03 23:59:59", "seven days ago"),
        _rec("garbage", "no ts"),
    ]
    texts = lambda w: {r["text"] for r in filter_records(records, window=w, now=NOW)}
    assert texts("today") == {"today midnight"}
    assert texts("7d") == {"today midnight", "yesterday", "six days ago"}
    assert texts("all") == {r["text"] for r in records}
    with pytest.raises(ValueError):
        filter_records(records, window="week", now=NOW)


def test_filter_query_matches_text_and_ts_case_insensitive_all_terms():
    records = [
        _rec("2026-11-09 09:30:00", "Deploy the API"),
        _rec("2026-11-10 09:31:00", "deploy later"),
        _rec("2026-11-10 10:00:00", "午饭 吃什么"),
    ]
    q = lambda s: [r["text"] for r in filter_records(records, query=s, now=NOW)]
    assert q("DEPLOY") == ["deploy later", "Deploy the API"]
    assert q("deploy 2026-11-09") == ["Deploy the API"]
    assert q("09:3") == ["deploy later", "Deploy the API"]
    assert q("午饭") == ["午饭 吃什么"]
    assert q("deploy 午饭") == []


def test_record_source_and_title():
    phone = _rec("2026-11-10 09:35:43", "hello\nworld", source="phone", backend="micwave_phone")
    mac = _rec("2026-11-10 09:35:43", "x" * 100)
    assert record_source(phone) == "phone"
    assert record_source(mac) == "mac"
    assert format_recent_title(phone) == "11-10 09:35 | \U0001F4F1 [micwave_phone] hello world"
    title = format_recent_title(mac)
    assert title.startswith("11-10 09:35 | \U0001F4BB ")
    assert title.endswith("x" * 77 + "...")


def test_append_after_torn_tail_keeps_new_record_in_one_write(monkeypatch, tmp_path):
    path = tmp_path / "transcripts.jsonl"
    path.write_bytes(b'{"ts": "2026-11-10 09:00:00", "text": "good"}\n{"text":"par')
    calls = []
    real_write = os.write
    monkeypatch.setattr(
        transcript_store.os, "write", lambda fd, data: calls.append(data) or real_write(fd, data)
    )
    append_record(str(path), {"ts": "2026-11-10 09:00:01", "text": "new"})
    assert len(calls) == 1 and calls[0].startswith(b"\n")
    texts = [r["text"] for r in TranscriptStoreReader(str(path)).refresh()]
    assert texts == ["good", "new"]


def test_append_to_clean_or_empty_file_adds_no_blank_line(tmp_path):
    path = tmp_path / "transcripts.jsonl"
    append_record(str(path), {"ts": "x", "text": "a"})
    append_record(str(path), {"ts": "x", "text": "b"})
    assert path.read_bytes().count(b"\n") == 2
    assert not path.read_bytes().startswith(b"\n")


def test_short_write_terminates_line_so_next_append_survives(monkeypatch, tmp_path):
    path = tmp_path / "transcripts.jsonl"
    real_write = os.write
    state = {"first": True}

    def short_once(fd, data):
        if state["first"]:
            state["first"] = False
            return real_write(fd, data[:5])
        return real_write(fd, data)

    monkeypatch.setattr(transcript_store.os, "write", short_once)
    with pytest.raises(OSError):
        append_record(str(path), {"ts": "x", "text": "lost"})
    assert path.read_bytes().endswith(b"\n")
    append_record(str(path), {"ts": "x", "text": "kept"})
    assert [r["text"] for r in TranscriptStoreReader(str(path)).refresh()] == ["kept"]


def test_lone_surrogate_text_is_kept_with_replacement(tmp_path):
    path = tmp_path / "transcripts.jsonl"
    append_record(str(path), {"ts": "x", "text": "ok \ud800 end"})
    (record,) = TranscriptStoreReader(str(path)).refresh()
    assert record["text"].startswith("ok ") and record["text"].endswith(" end")


def test_reader_detects_in_place_truncate_and_regrow(tmp_path):
    path = tmp_path / "transcripts.jsonl"
    append_record(str(path), {"ts": "2026-11-10 09:00:00", "text": "first-record"})
    reader = TranscriptStoreReader(str(path))
    assert [r["text"] for r in reader.refresh()] == ["first-record"]
    ino = path.stat().st_ino
    # same inode, truncated then regrown past the old offset
    with open(path, "r+b") as handle:
        handle.truncate(0)
        handle.write(
            (json.dumps({"ts": "2026-11-10 10:00:00", "text": "a"}) + "\n").encode()
            + (json.dumps({"ts": "2026-11-10 10:00:01", "text": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}) + "\n").encode()
        )
    assert path.stat().st_ino == ino
    assert [r["text"] for r in reader.refresh()] == ["a", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]
