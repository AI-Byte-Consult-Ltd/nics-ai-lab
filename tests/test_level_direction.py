"""Offline test of the level-method direction research (src/nics/market/level_direction.py).

SYNTHETIC random-walk bars, results meaningless. Checks that nothing looks into the future
(a run on a shorter history gives the same early trades), that each direction rule does what
its name says, and that the script runs on a snapshot.

  python tests/test_level_direction.py
"""
import csv
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import level_direction as ld  # noqa: E402


def bars(n: int, seed: int, drift: float = 0.0):
    rnd = random.Random(seed)
    t0 = datetime(2015, 1, 5, tzinfo=timezone.utc).timestamp()
    t, h, lo, c, p = [], [], [], [], 100.0
    for i in range(n):
        o = p
        p = o * (1 + rnd.gauss(drift, 0.002))
        t.append(t0 + 3600 * i); h.append(max(o, p) * (1 + abs(rnd.gauss(0, 0.0007)))); lo.append(min(o, p) * (1 - abs(rnd.gauss(0, 0.0007)))); c.append(p)
    return np.array(t), np.array(h), np.array(lo), np.array(c)


def main() -> None:
    # swing levels are known only after the two following bars closed
    h = np.array([1, 2, 5, 3, 2, 1, 1.0]); lo = h - 0.5
    price, when, need = ld.fractals(h, lo, np.arange(7.0), 2, 1)
    assert 5.0 in price and need[list(price).index(5.0)] == 5, (price, need)
    assert ld.cluster_levels(np.array([10.0, 10.1, 12.0]), np.array([1.0, 2.0, 3.0]), 0.3) == [10.1, 12.0]

    t, h, lo, c = bars(12000, 1)
    full = ld.prepare(t, h, lo, c)
    assert full is not None
    tr_full, states = ld.run_symbol(full, 0.01)
    assert len(tr_full["base"]) > 30 and states, len(tr_full["base"])
    # no look-ahead: the same history cut short must reproduce the early trades exactly
    cut = 9000
    part = ld.prepare(t[:cut], h[:cut], lo[:cut], c[:cut])
    tr_part, _ = ld.run_symbol(part, 0.01)
    limit = t[cut] - 40 * 3600
    for v in ld.VARIANTS:
        a = [(x["t"], x["dir"], x["why"], round(x["r"], 6)) for x in tr_full[v] if x["t"] < limit]
        b = [(x["t"], x["dir"], x["why"], round(x["r"], 6)) for x in tr_part[v] if x["t"] < limit]
        assert a == b and a, (v, len(a), len(b))
    # each rule does what it says
    assert all(x["dir"] == x["base"] or x["base"] == 0 for x in tr_full["base"])
    assert all(x["dir"] > 0 for x in tr_full["pullback"] if x["base"] < 0 and x["lt"] > 0)
    assert all(x["dir"] > 0 for x in tr_full["rule"] if x["base"] < 0 and x["lt"] > 0)
    assert not any(x["kind"] == "breakout" for x in tr_full["no_low"])
    assert all(x["dir"] == x["lt"] for x in tr_full["lt_only"])
    for v in ld.VARIANTS:
        for x in tr_full[v]:
            assert x["why"] in ("SL", "BE", "TP", "TIME") and x["r"] > -1.6 and np.isfinite(x["fwd"]), x
            if x["why"] == "SL":
                assert x["r"] <= -0.99, x

    tmp = Path(tempfile.mkdtemp())
    try:
        cols = ["source", "symbol", "interval", "candle_time", "open", "high", "low", "close", "volume", "bid", "ask"]
        with open(tmp / "public_candles.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for i in range(len(t)):
                ts = datetime.fromtimestamp(t[i], timezone.utc).isoformat()
                w.writerow(["public:dukascopy:TEST", "XAU/USD", "1h", ts, c[i - 1] if i else c[0], h[i], lo[i], c[i], 1, "", ""])
        with open(tmp / "candles.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for i in range(50):
                ts = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat()
                w.writerow(["vantage_mt5", "XAU/USD", "1h", ts, 100, 101, 99, 100, 5, 100.0, 100.02])
        r = subprocess.run([sys.executable, str(ROOT / "scripts/level_direction_research.py"), "--snapshot", str(tmp),
                            "--from-year", "2015", "--no-push", "--out", str(tmp / "out")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "как сейчас" in r.stdout and "spread 0.02" in r.stdout, r.stdout[-1500:]
        assert (tmp / "out" / "level_direction_last.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("level direction research: ok")


if __name__ == "__main__":
    main()
