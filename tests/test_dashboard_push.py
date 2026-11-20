"""Dashboard push: URL derivation, payload shape, and that failures never raise."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nics.telemetry.push import build_payload, metrics_url, push  # noqa: E402


def test_url_derived_from_data_url():
    env = {"NICS_DATA_URL": "https://n8n.example.com/webhook/nics-lab-market-export"}
    assert metrics_url(env) == "https://n8n.example.com/webhook/nics-lab-metrics"


def test_explicit_url_wins():
    env = {"NICS_DATA_URL": "https://a.example/webhook/x", "NICS_METRICS_URL": "https://b.example/hook"}
    assert metrics_url(env) == "https://b.example/hook"


def test_no_url_no_push():
    logs = []
    assert push("nics-trading", "self_learn", {}, env={}, log=logs.append) is False
    assert "skipped" in logs[0]


def test_payload_is_small_and_has_no_key():
    big = {"x": "a" * 50000}
    body = json.loads(build_payload("nics-trading", "self_learn", big, text="t" * 9000, cycle="c" * 100))
    assert set(body) == {"model", "kind", "cycle", "summary", "text"}
    assert len(body["summary"]) <= 30000 and len(body["text"]) == 4000 and len(body["cycle"]) == 64
    assert json.loads(body["summary"])["truncated"] is True


def test_network_failure_does_not_raise():
    env = {"NICS_METRICS_URL": "http://127.0.0.1:9/nope", "NICS_DATA_KEY": "k" * 64}
    logs = []
    assert push("nics-trading", "self_learn", {"a": 1}, env=env, log=logs.append, timeout=2) is False
    assert "failed" in logs[0] and "k" * 64 not in logs[0]
