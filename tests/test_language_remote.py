import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from language_remote import MAX_MINUTES, decide  # noqa: E402

NOW = datetime(2026, 11, 23, 10, 0, tzinfo=timezone.utc)
REQ = {"id": "a", "minutes": 60, "not_after": "2026-11-23T12:00:00+00:00"}


def test_starts_once():
    assert decide(REQ, "", NOW, 12) == (60, "start")
    assert decide(REQ, "a", NOW, 12) == (0, "already handled")


def test_empty_and_missing():
    assert decide(None, "", NOW, 12)[0] == 0
    assert decide({"id": "", "minutes": 60, "not_after": ""}, "", NOW, 12) == (0, "no request")


def test_expired_and_bad():
    assert decide({**REQ, "not_after": "2026-11-23T09:00:00+00:00"}, "", NOW, 12) == (0, "expired")
    assert decide({**REQ, "not_after": "soon"}, "", NOW, 12) == (0, "bad request")
    assert decide({**REQ, "minutes": 0}, "", NOW, 12) == (0, "bad request")
    assert decide({**REQ, "minutes": "rm -rf"}, "", NOW, 12) == (0, "bad request")


def test_clamp_and_night():
    assert decide({**REQ, "minutes": 9999}, "", NOW, 12) == (MAX_MINUTES, "start")
    assert decide({**REQ, "minutes": 1}, "", NOW, 12) == (10, "start")
    assert decide(REQ, "", NOW, 3) == (0, "night window")
    assert decide({**REQ, "not_after": "2026-11-23T12:00:00"}, "", NOW, 12) == (60, "start")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
