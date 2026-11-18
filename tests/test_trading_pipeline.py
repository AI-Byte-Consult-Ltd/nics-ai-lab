"""Smoke test for the NICS Trading pipeline on SYNTHETIC data.

The random-walk candles here only check that the code runs end to end.
Results on them mean nothing and must never be reported as model results.

  python tests/test_trading_pipeline.py
"""
import csv
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_snapshot(d: Path, bars: int = 3000) -> None:
    rnd = random.Random(0)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with open(d / "candles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "interval", "candle_time", "open", "high", "low", "close", "volume", "bid", "ask"])
        for sym in ("XAUUSD", "EURUSD"):
            p = 2000.0 if sym == "XAUUSD" else 1.1
            for i in range(bars):
                o = p
                c = o * (1 + rnd.gauss(0, 0.001))
                h, l = max(o, c) * (1 + abs(rnd.gauss(0, 0.0005))), min(o, c) * (1 - abs(rnd.gauss(0, 0.0005)))
                ts = (t0 + timedelta(minutes=15 * i)).isoformat()
                w.writerow([sym, "M15", ts, o, h, l, c, rnd.randint(100, 1000), c, c * 1.00005])
                p = c
    with open(d / "signals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "symbol", "direction", "entry_low", "entry_high", "stop_loss", "tp1",
                    "confidence_score", "technical_score", "status", "final_result", "hit_targets",
                    "created_at", "closed_at", "origin_mode"])
        rows = list(csv.DictReader(open(d / "candles.csv")))
        for k in range(400):
            r = rows[rnd.randint(200, len(rows) - 300)]
            c = float(r["close"]); long = rnd.random() < 0.5
            sl = c * (0.997 if long else 1.003); tp = c * (1.004 if long else 0.996)
            w.writerow([f"s{k}", r["symbol"], "LONG" if long else "SHORT", c, c, sl, tp, 70, 0.5,
                        "CLOSED", "", "{}", r["candle_time"], "", "AUTO"])


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        make_snapshot(tmp)
        cfg = tmp / "smoke.yaml"
        text = (ROOT / "configs/trading/nano.yaml").read_text()
        text = text.replace("max_steps: 3000", "max_steps: 60").replace("eval_interval: 100", "eval_interval: 20")
        text = text.replace("experiments_dir: experiments/trading/nics-trading-nano",
                            f"experiments_dir: {tmp / 'exp'}")
        text = text.replace("min_signal_samples: 50", "min_signal_samples: 10").replace(
            "min_filtered_trades: 20", "min_filtered_trades: 5")
        cfg.write_text(text)
        subprocess.run([sys.executable, str(ROOT / "scripts/train_trading.py"), "--config", str(cfg),
                        "--snapshot", str(tmp), "--experiment-id", "smoke"], check=True)
        assert (tmp / "exp" / "smoke" / "release" / "model.safetensors").exists()
        assert (tmp / "exp" / "smoke" / "release" / "metrics.json").exists()
        print("SMOKE TEST PASSED (synthetic data, results meaningless)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
