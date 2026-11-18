"""Smoke test for the NICS Trading pipeline on SYNTHETIC data.

The random-walk candles here only check that the code runs end to end with
every data source present (broker candles on two timeframes, public D1,
scanner signals, shadow trades, own MT5 trades). Results on them mean
nothing and must never be reported as model results.

  python tests/test_trading_pipeline.py
"""
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import data as market  # noqa: E402

CANDLE_COLS = ["source", "symbol", "interval", "candle_time", "open", "high", "low", "close", "volume", "bid", "ask"]


def walk(rnd, p, n, sigma):
    out = []
    for _ in range(n):
        o = p
        c = o * (1 + rnd.gauss(0, sigma))
        h, lo = max(o, c) * (1 + abs(rnd.gauss(0, sigma / 2))), min(o, c) * (1 - abs(rnd.gauss(0, sigma / 2)))
        out.append((o, h, lo, c))
        p = c
    return out


def make_snapshot(d: Path, bars: int = 3000) -> None:
    rnd = random.Random(0)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    m15 = {}
    with open(d / "candles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CANDLE_COLS)
        for sym, p in (("XAU/USD", 2000.0), ("EUR/USD", 1.1)):
            rows = []
            for i, (o, h, lo, c) in enumerate(walk(rnd, p, bars, 0.001)):
                ts = (t0 + timedelta(minutes=15 * i)).isoformat()
                w.writerow(["vantage_mt5", sym, "15min", ts, o, h, lo, c, rnd.randint(100, 1000), c, c * 1.00005])
                rows.append((ts, c))
            m15[sym] = rows
            for i, (o, h, lo, c) in enumerate(walk(rnd, p, bars // 4, 0.002)):
                ts = (t0 + timedelta(hours=i)).isoformat()
                w.writerow(["vantage_mt5", sym, "1h", ts, o, h, lo, c, rnd.randint(100, 1000), c, c * 1.00005])
    with open(d / "public_candles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CANDLE_COLS)
        for i, (o, h, lo, c) in enumerate(walk(rnd, 1500.0, 1500, 0.01)):
            ts = (t0 - timedelta(days=1500 - i)).isoformat()
            w.writerow(["public:yahoo:GC=F", "XAU/USD", "1day", ts, o, h, lo, c, 0, "", ""])
    with open(d / "signals.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "symbol", "direction", "entry_low", "entry_high", "stop_loss", "tp1", "status",
                    "created_at", "origin_mode"])
        for k in range(400):
            sym = rnd.choice(list(m15))
            ts, c = m15[sym][rnd.randint(200, bars - 300)]
            long = rnd.random() < 0.5
            sl = c * (0.997 if long else 1.003); tp = c * (1.004 if long else 0.996)
            w.writerow([f"s{k}", sym, "LONG" if long else "SHORT", c, c, sl, tp, "CLOSED", ts,
                        rnd.choice(["SCHEDULED", "TARGETED"])])
    with open(d / "swing_shadow.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "strategy", "symbol", "direction", "entry_time", "entry_price", "stop_loss",
                    "take_profit", "status"])
        for k in range(40):
            ts, c = m15["XAU/USD"][rnd.randint(200, bars - 300)]
            w.writerow([k, "MTF_CASCADE_v1", "XAU/USD", "LONG", ts, c, c * 0.996, c * 1.006, "TP"])
    with open(d / "own_trades_prop.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "position_id", "symbol", "direction", "open_time", "open_price", "stop_loss",
                    "take_profit"])
        for k in range(40):
            ts, c = m15["EUR/USD"][rnd.randint(200, bars - 300)]
            w.writerow(["prop", k, "EURUSD.r", "SHORT", ts, c, c * 1.003, c * 0.995])
    (d / "manifest.json").write_text(json.dumps({"dataset_id": "synthetic"}))


def check_symbols() -> None:
    known = {"US500", "EUR/USD", "UKOUSDft", "XAU/USD", "BTC/USD"}
    for raw, want in (("SP500.r", "US500"), ("EURUSD.r", "EUR/USD"), ("CL_BRENT", "UKOUSDft"),
                      ("XAUUSD+", "XAU/USD"), ("BTCUSD", "BTC/USD"), ("FOO", None)):
        got = market.canonical_symbol(raw, known)
        assert got == want, (raw, got, want)


def main() -> None:
    check_symbols()
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
        rel = tmp / "exp" / "smoke" / "release"
        assert (rel / "model.safetensors").exists()
        metrics = json.loads((rel / "metrics.json").read_text())
        man = json.loads((rel / "training-manifest.json").read_text())
        kinds = man["split"]["setup_samples"]["all_by_kind"]
        assert {"scanner_scheduled", "scanner_targeted", "shadow", "own_trade"} <= set(kinds), kinds
        assert man["split"]["horizon_samples"]["train_public"] > 0
        assert set(man["split"]["horizon_samples"]["train_by_timeframe"]) >= {"15min", "1h", "1day"}
        assert "test_by_timeframe" in metrics["horizon"]
        print("SMOKE TEST PASSED (synthetic data, results meaningless)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
