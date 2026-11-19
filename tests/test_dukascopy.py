"""Offline test of scripts/fetch_dukascopy.py on SYNTHETIC cached files (no network).

Builds a fake monthly hourly file in the Dukascopy layout (24-byte big-endian records:
offset seconds, open, close, low, high as int32 points, volume float32, LZMA-compressed),
then checks the decoder finds high/low, the price scale and H4 aggregation.

  python tests/test_dukascopy.py
"""
import csv
import lzma
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        rnd = random.Random(1)
        recs, p = [], 1.10
        for hour in range(31 * 24):
            o = p; c = o * (1 + rnd.gauss(0, 0.001))
            h = max(o, c) * 1.0005; lo = min(o, c) * 0.9995
            vol = 0.0 if (hour // 24) % 7 in (5, 6) else 1000.0  # flat weekend hours carry no volume
            recs.append(struct.pack(">5if", hour * 3600, *(round(x * 1e5) for x in (o, c, lo, h)), vol))
            p = c
        cache = tmp / "cache" / "EURUSD"
        cache.mkdir(parents=True)
        (cache / "2024-01.bi5").write_bytes(lzma.compress(b"".join(recs), format=lzma.FORMAT_ALONE))
        snap = tmp / "snap"
        snap.mkdir()
        with open(snap / "candles.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "symbol", "interval", "candle_time", "open", "high", "low", "close",
                        "volume", "bid", "ask"])
            w.writerow(["vantage_mt5", "EUR/USD", "15min", "2026-11-19T00:00:00+00:00", 1.1, 1.1, 1.1, 1.12, 1, "", ""])
        subprocess.run([sys.executable, str(ROOT / "scripts/fetch_dukascopy.py"), "--snapshot", str(snap),
                        "--from-year", "2024", "--symbols", "EUR/USD", "--offline", "--cache",
                        str(tmp / "cache")], check=True)
        rows = list(csv.DictReader(open(snap / "public_candles.csv", newline="")))
        h1 = [r for r in rows if r["interval"] == "1h"]
        h4 = [r for r in rows if r["interval"] == "4h"]
        assert h1 and h4, (len(h1), len(h4))
        assert len(h1) < 31 * 24  # zero-volume hours dropped
        assert h1[0]["candle_time"].startswith("2024-01-01T00:00")
        for r in h1 + h4:
            assert float(r["high"]) >= max(float(r["open"]), float(r["close"])) >= float(r["low"]) > 0, r
            assert 0.9 < float(r["close"]) < 1.4, r  # scale 1e5 detected
        assert all(r["source"] == "public:dukascopy:EURUSD" for r in rows)
        print("DUKASCOPY OFFLINE TEST PASSED (synthetic data)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
