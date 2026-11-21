"""Offline test of scripts/opinion_live.py on SYNTHETIC data (no network, results meaningless).

Checks which signals get an opinion (only SENT, recent, not yet answered) and that the
model returns one opinion per signal with both probabilities.

  python tests/test_opinion.py
"""
import csv
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from nics.common.config import load_config  # noqa: E402
from nics.training import trading as T  # noqa: E402
import opinion_live  # noqa: E402
from test_trading_pipeline import make_snapshot  # noqa: E402


def main() -> None:
    now = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc).timestamp()
    rows = [{"id": 1, "delivery_status": "SENT", "created_at": "2026-11-21T11:00:00+00:00"},
            {"id": 2, "delivery_status": "READY", "created_at": "2026-11-21T11:00:00+00:00"},
            {"id": 3, "delivery_status": "sent", "created_at": "2026-11-21T06:00:00+00:00"},  # too old
            {"id": 4, "delivery_status": "SENT", "created_at": "2026-11-21T11:30:00+00:00"},  # answered
            {"id": 5, "delivery_status": "SENT", "created_at": ""}]
    got = [r["id"] for r in opinion_live.new_sent_signals(rows, {"4"}, 3, now)]
    assert got == [1], got
    assert opinion_live.opinion_url({"NICS_DATA_URL": "https://h.example/webhook/x?y=1"}) == \
        "https://h.example/webhook/nics-lab-opinion"
    assert opinion_live.opinion_url({"NICS_OPINION_URL": " https://o/x ", "NICS_DATA_URL": "https://h/"}) == "https://o/x"
    assert opinion_live.opinion_url({}) is None

    tmp = Path(tempfile.mkdtemp())
    try:
        make_snapshot(tmp)
        text = (ROOT / "configs/trading/nano.yaml").read_text()
        text = text.replace("max_steps: 3000", "max_steps: 30").replace("eval_interval: 100", "eval_interval: 10")
        text = text.replace("experiments_dir: experiments/trading/nics-trading-nano", f"experiments_dir: {tmp / 'exp'}")
        text = text.replace("min_signal_samples: 50", "min_signal_samples: 10")
        (tmp / "c.yaml").write_text(text)
        subprocess.run([sys.executable, str(ROOT / "scripts/train_trading.py"), "--config", str(tmp / "c.yaml"),
                        "--snapshot", str(tmp), "--experiment-id", "d"], check=True, capture_output=True)
        cfg = load_config(str(tmp / "c.yaml"))
        model, _ = T.load_release(tmp / "exp" / "d" / "release", cfg)
        with open(tmp / "signals.csv", newline="") as f:
            ids = [r["id"] for r in csv.DictReader(f)][-5:]
        ops = opinion_live.score(cfg, tmp, model, set(ids) | {"no-such-id"})
        assert {o["signal_id"] for o in ops} <= set(ids) and len(ops) >= 1, ops
        for o in ops:
            assert o["scanner_direction"] in ("LONG", "SHORT") and o["model_direction"] in ("LONG", "SHORT"), o
            assert 0 <= o["p_scanner_side"] <= 1 and 0 <= o["p_other_side"] <= 1, o
            agree = o["model_direction"] == o["scanner_direction"]
            assert agree == (o["p_scanner_side"] >= o["p_other_side"]), o
            # the model's own plan: consistent geometry, best expected R of its grid
            assert o["plan_direction"] in ("LONG", "SHORT") and o["plan_entry_kind"] in ("signal", "market"), o
            up = o["plan_direction"] == "LONG"
            assert (o["plan_sl"] < o["plan_entry"] < o["plan_tp"]) if up else (o["plan_tp"] < o["plan_entry"] < o["plan_sl"]), o
            assert abs(abs(o["plan_tp"] - o["plan_entry"]) / abs(o["plan_entry"] - o["plan_sl"]) - o["plan_rr"]) < 0.02, o
            assert 0 <= o["plan_p"] <= 1 and o["exp_r_signal"] is not None, o
        print(f"OK: {len(ops)} opinions, e.g. {ops[0]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
