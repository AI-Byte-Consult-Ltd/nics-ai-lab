"""Offline test of the dip-buying research (src/nics/market/dip_long.py).

SYNTHETIC bars, results meaningless. Checks: the rule looks only at past closes (a longer
history does not change earlier trades), costs lower the result, a random walk shows no edge,
the position cap is respected, the script runs on a snapshot.

  python tests/test_dip_long.py
"""
import csv
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import dip_long as dl  # noqa: E402


def bars(n: int, seed: int, drift: float = 0.0003) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    t0 = datetime(2000, 1, 3, tzinfo=timezone.utc).timestamp()
    return np.column_stack([t0 + 86400 * np.arange(n), o, h, lo, c])


def main() -> None:
    a = bars(3000, 1)
    full = dl.trades(a, down=3)
    part = dl.trades(a[:2000], down=3)
    assert len(full) > 20 and len(part) > 10
    for x, y in zip(part[:-1], full):  # the last trade of the short history may be cut by its end
        assert (x["i"], x["j"]) == (y["i"], y["j"]) and abs(x["ret"] - y["ret"]) < 1e-12
    for tr in full:
        i = tr["i"]
        assert a[i, 4] > a[i - 200:i, 4].mean() and all(a[i - k, 4] < a[i - k - 1, 4] for k in range(3))
        assert 1 <= tr["days"] <= 10
    free = dl.trades(a, down=3, cost=0.0, financing=0.0)
    assert np.mean([x["ret"] for x in free]) > np.mean([x["ret"] for x in full])
    # a random walk without drift: no edge beyond noise
    rw = sum((dl.trades(bars(6000, s, drift=0.0), down=3, cost=0.0, financing=0.0) for s in range(2, 10)), [])
    st = dl.stat([x["ret"] for x in rw])
    assert abs(st["t"]) < 3.5, st
    assert dl.is_dip(np.array([5.0, 4, 3, 2]), 3, down=3) and not dl.is_dip(np.array([5.0, 4, 5, 4]), 3, down=3)
    assert dl.is_dip(np.array([5.0, 6, 7, 8, 9, 4]), 5, down=0, low=5) and dl.strength2(np.array([3.0, 2, 1]), 2) == 0.0
    series = {f"S{k}": bars(3000, 20 + k) for k in range(5)}
    p1 = dl.portfolio(series, max_positions=1, year_from=2000, down=3)
    p5 = dl.portfolio(series, max_positions=5, year_from=2000, down=3)
    assert p1["trades"] < p5["trades"] and p1["skipped"] > p5["skipped"] and p1["max_drawdown_pct"] > 0
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp)
        with open(snap / "public_candles.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source", "symbol", "interval", "candle_time", "open", "high", "low", "close", "volume", "x", "y"])
            for sym, seed in (("US500", 31), ("US100", 32), ("XAU/USD", 33)):
                for row in bars(1500, seed):
                    ts = datetime.fromtimestamp(row[0], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    w.writerow(["public:test", sym, "1day", ts, *[f"{v:.4f}" for v in row[1:]], 0, "", ""])
                w.writerow(["public:test", sym, "1week", "2000-01-03T00:00:00Z", 1, 1, 1, 1, 0, "", ""])
        got = dl.load_daily(snap)
        assert set(got) == {"US500", "US100", "XAU/USD"} and len(got["US500"]) == 1500
        out = snap / "out"
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "dip_long_research.py"), "--snapshot", str(snap), "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert (out / "dip_long_last.json").exists() and "indices" in (out / "dip_long_last.txt").read_text(encoding="utf-8")
    print("OK dip_long")


if __name__ == "__main__":
    main()
