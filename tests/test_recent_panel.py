"""Recent panel logic: rereads the shared transcript store (phone turns are
appended by another process), filters by text/time, tags the source."""
import json
import queue
from datetime import datetime
from types import SimpleNamespace

import pytest

import ime_menubar
from ime_menubar import BrainwaveIMEApp
from micwave_core.transcript_store import TranscriptStoreReader, append_record


class _Button:
    def __init__(self):
        self.title = None
        self.enabled = None

    def setTitle_(self, title):
        self.title = title

    def setEnabled_(self, enabled):
        self.enabled = enabled


class _Label:
    def __init__(self):
        self.value = None

    def setStringValue_(self, value):
        self.value = value


def _app(tmp_path, rows=3):
    store = tmp_path / "transcripts.jsonl"
    app = BrainwaveIMEApp.__new__(BrainwaveIMEApp)
    app._pending_transcripts = queue.Queue()
    app._history_dir = str(tmp_path)
    app._history_file = str(store)
    app._history_reader = TranscriptStoreReader(str(store))
    app._recent_limit = rows
    app._recent_query = ""
    app._recent_window = "all"
    app._recent_transcripts = []
    app._recent_match_count = 0
    app._recent_panel_buttons = [_Button() for _ in range(rows)]
    app._recent_count_label = _Label()
    return app, store


def _today(hms):
    return datetime.now().strftime("%Y-%m-%d ") + hms


def test_reload_picks_up_phone_records_written_by_another_process(tmp_path):
    app, store = _app(tmp_path)
    append_record(str(store), {"ts": _today("08:00:00"), "text": "mac one", "backend": "echo"})
    app._load_recent_history()
    assert [r["text"] for r in app._recent_transcripts] == ["mac one"]

    # another process (micwave_phone) appends
    append_record(str(store), {"ts": _today("09:00:00"), "text": "phone one", "source": "phone"})
    app._load_recent_history()
    app._update_recent_panel()
    assert [r["text"] for r in app._recent_transcripts] == ["phone one", "mac one"]
    titles = [b.title for b in app._recent_panel_buttons]
    assert "\U0001F4F1" in titles[0] and "phone one" in titles[0]
    assert "\U0001F4BB" in titles[1] and "[echo]" in titles[1]
    assert titles[2] == "(empty)" and app._recent_panel_buttons[2].enabled is False
    assert app._recent_count_label.value == "2 matches"


def test_drain_appends_mac_entry_to_store_and_refreshes_rows(tmp_path):
    app, store = _app(tmp_path)
    append_record(str(store), {"ts": _today("07:00:00"), "text": "phone earlier", "source": "phone"})
    app._load_recent_history()
    app._pending_transcripts.put({"ts": _today("10:00:00"), "text": "mac new", "mode": "optimized"})
    app._drain_pending_transcripts()

    lines = [json.loads(line) for line in store.read_text("utf-8").splitlines()]
    assert [r["text"] for r in lines] == ["phone earlier", "mac new"]
    assert [r["text"] for r in app._recent_transcripts] == ["mac new", "phone earlier"]
    assert "mac new" in app._recent_panel_buttons[0].title


def test_search_and_window_callbacks_filter_rows(tmp_path):
    app, store = _app(tmp_path, rows=5)
    append_record(str(store), {"ts": "2020-01-01 09:00:00", "text": "deploy old"})
    append_record(str(store), {"ts": _today("09:00:00"), "text": "Deploy today", "source": "phone"})
    append_record(str(store), {"ts": _today("10:00:00"), "text": "lunch"})
    app._load_recent_history()
    assert len(app._recent_transcripts) == 3

    app.recentSearchChanged_(SimpleNamespace(stringValue=lambda: "deploy"))
    assert [r["text"] for r in app._recent_transcripts] == ["Deploy today", "deploy old"]

    app.recentWindowChanged_(SimpleNamespace(selectedSegment=lambda: 0))  # Today
    assert app._recent_window == "today"
    assert [r["text"] for r in app._recent_transcripts] == ["Deploy today"]
    assert app._recent_count_label.value == "1 match"

    app.recentSearchChanged_(SimpleNamespace(stringValue=lambda: ""))
    assert [r["text"] for r in app._recent_transcripts] == ["lunch", "Deploy today"]


def test_rows_are_capped_but_count_reports_all_matches(tmp_path):
    app, store = _app(tmp_path, rows=2)
    for i in range(5):
        append_record(str(store), {"ts": _today(f"0{i}:00:00"), "text": f"t{i}"})
    app._load_recent_history()
    app._update_recent_panel()
    assert [r["text"] for r in app._recent_transcripts] == ["t4", "t3"]
    assert app._recent_count_label.value == "showing 2 of 5 matches"


def test_button_click_copies_the_filtered_row(tmp_path, monkeypatch):
    app, store = _app(tmp_path)
    append_record(str(store), {"ts": _today("09:00:00"), "text": "alpha"})
    append_record(str(store), {"ts": _today("10:00:00"), "text": "beta"})
    app._load_recent_history()
    app.recentSearchChanged_(SimpleNamespace(stringValue=lambda: "alpha"))
    copied = []
    app._copy_to_clipboard = copied.append
    app._recent_panel = None
    monkeypatch.setattr(app, "_dismiss_recent_panel", lambda: None, raising=False)
    app.recentPanelButtonClicked_(SimpleNamespace(tag=lambda: 0))
    assert copied == ["alpha"]


@pytest.mark.skipif(not ime_menubar.HAS_APPKIT, reason="AppKit unavailable")
def test_build_recent_panel_constructs_real_appkit_controls(tmp_path):
    app, _store = _app(tmp_path, rows=4)
    app._recent_panel = None
    app._build_recent_panel()
    assert len(app._recent_panel_buttons) == 4
    assert app._recent_search_field is not None
    assert app._recent_window_control.segmentCount() == 3
    assert app._recent_window_control.selectedSegment() == 2
    app._load_recent_history()
    app._update_recent_panel()
    assert app._recent_count_label.stringValue() == "0 matches"


def test_opening_panel_rereads_store(tmp_path, monkeypatch):
    app, store = _app(tmp_path)
    app._load_recent_history()
    assert app._recent_transcripts == []
    append_record(str(store), {"ts": _today("09:00:00"), "text": "from phone", "source": "phone"})
    monkeypatch.setattr(ime_menubar, "HAS_APPKIT", False)
    monkeypatch.setattr(ime_menubar.rumps, "alert", lambda *a, **k: None)
    app._open_recent_panel(None)
    assert [r["text"] for r in app._recent_transcripts] == ["from phone"]


def test_opening_panel_with_pending_entry_refreshes_store_once(tmp_path, monkeypatch):
    app, store = _app(tmp_path)
    app._load_recent_history()
    app._pending_transcripts.put({"ts": _today("09:00:00"), "text": "pending mac"})
    calls = []
    real_refresh = app._history_reader.refresh
    app._history_reader.refresh = lambda: calls.append(1) or real_refresh()
    monkeypatch.setattr(ime_menubar, "HAS_APPKIT", False)
    monkeypatch.setattr(ime_menubar.rumps, "alert", lambda *a, **k: None)
    app._open_recent_panel(None)
    assert len(calls) == 1
    assert [r["text"] for r in app._recent_transcripts] == ["pending mac"]
