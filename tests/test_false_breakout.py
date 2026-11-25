"""Offline test of the false-breakout research (src/nics/market/false_breakout.py).

SYNTHETIC random-walk bars, results meaningless. Checks: no lookahead in features (a shorter
history gives the same early events and features), labels and trades are consistent, a random
walk gives about 50 % AUC, the script runs on a snapshot.

  python tests/test_false_breakout.py
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

from nics.market import false_breakout as fb  # noqa: E402


def bars(n: int, seed: int):
    rnd = random.Random(seed)
    t0 = datetime(2012, 1, 2, tzinfo=timezone.utc).timestamp()
    t, h, lo, c, v, p = [], [], [], [], [], 100.0
    for i in range(n):
        o = p
        p = o * (1 + rnd.gauss(0, 0.002))
        t.append(t0 + 3600 * i); h.append(max(o, p) * (1 + abs(rnd.gauss(0, 0.0007))))
        lo.append(min(o, p) * (1 - abs(rnd.gauss(0, 0.0007)))); c.append(p); v.append(rnd.randint(50, 150))
    return np.array(t), np.array(h), np.array(lo), np.array(c), np.array(v, float)


def main() -> None:
    assert fb.round_step(3600.0) == 10.0 and abs(fb.round_step(1.08) - 0.01) < 1e-12
    t, h, lo, c, v = bars(9000, 1)
    ev = fb.events(t, h, lo, c, v, spread=0.0)
    assert len(ev) > 100, len(ev)
    # no lookahead: cutting the future off keeps early events and their features identical
    cut = 6000
    ev2 = fb.events(t[:cut], h[:cut], lo[:cut], c[:cut], v[:cut], spread=0.0)
    early = [e for e in ev if e["i"] < cut - 40]
    early2 = [e for e in ev2 if e["i"] < cut - 40]
    assert len(early) == len(early2)
    for a, b in zip(early, early2):
        assert a["i"] == b["i"] and all(a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in fb.FEATURES), (a, b)
        assert a["false"] == b["false"] and a["base"] == b["base"]
    for e in ev:
        i, d, L = e["i"], e["dir"], e["level"]
        assert d * (c[i] - L) > 0 and d * (c[i - 1] - L) <= 0        # a fresh close beyond the level
        assert e["base"] is None or -1.01 <= e["base"] <= 1.01
        assert e["touches"] >= 1 and e["age_h"] >= fb.PARAMS["gap_h"]
    # random walk: the model must not find anything out of sample
    many = []
    for seed in range(2, 7):
        tt, hh, ll, cc, vv = bars(60000, seed)
        many += fb.events(tt, hh, ll, cc, vv, 0.0)
    split = datetime(2016, 1, 1, tzinfo=timezone.utc).timestamp()
    c3 = fb.constructions(many)
    assert c3["base"]["n"] == len(many) and c3["confirm2"]["filled"] <= len(many)
    for k in ("base", "confirm2"):  # market entries must not make money on a random walk
        assert abs(c3[k]["avg"]) < 0.04, (k, c3[k])
    # retest is a limit order: its result depends on how the synthetic wicks are drawn, so only
    # check that it fills less often than the market entry
    assert c3["retest"]["filled"] < c3["base"]["filled"]
    m = fb.model_check(many, split)
    assert "out" in m, m
    # the false SHARE is mechanically predictable even on a random walk (shallow breaks return more
    # often), but R must not be: out-of-sample terciles of the model have about the same R
    ter = m["out"]["tercile_by_false_probability"]
    assert m["out"]["auc"] > 0.5
    assert max(abs(ter[k]["avg"]) for k in ter) < 0.08, ter
    # script on a tiny snapshot
    tmp = Path(tempfile.mkdtemp())
    try:
        snap = tmp / "data" / "market" / "2026-11-25"
        snap.mkdir(parents=True)
        tt, hh, ll, cc, vv = bars(40000, 9)
        with open(snap / "public_candles.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "symbol", "interval", "candle_time", "open", "high", "low", "close", "volume", "bid", "ask"])
            for i in range(len(tt)):
                ts = datetime.fromtimestamp(tt[i], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                w.writerow(["public:dukascopy:EURUSD", "EUR/USD", "1h", ts, cc[i - 1] if i else cc[0], hh[i], ll[i], cc[i], vv[i], "", ""])
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "false_breakout_research.py"), "--snapshot", str(snap),
                            "--no-push", "--no-yahoo", "--split-year", "2014", "--out", str(tmp / "out")],
                           capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        assert (tmp / "out" / "false_breakout_last.txt").exists()
        print(r.stdout[-1500:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _ = timedelta
    print("false breakout: OK")


if __name__ == "__main__":
    main()
