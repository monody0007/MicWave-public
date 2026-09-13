import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from whisper_backend.transcript_merge import append_transcript_delta


def test_append_with_empty_values():
    assert append_transcript_delta("", "") == ""
    assert append_transcript_delta("", "hello") == "hello"
    assert append_transcript_delta("hello", "") == "hello"


@pytest.mark.parametrize(
    "deltas",
    [
        ("SSOT", "SSOT"),
        ("你好啊", "你好啊"),
        ("abc", "abc"),
        ("S", "S", "O", "T"),
    ],
)
def test_r3_m4_provider_deltas_append_as_newly_available_chunks(deltas):
    transcript = ""
    for delta in deltas:
        transcript = append_transcript_delta(transcript, delta)
    assert transcript == "".join(deltas)
