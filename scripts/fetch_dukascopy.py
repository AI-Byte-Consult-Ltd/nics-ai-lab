"""Download long PUBLIC H1 history from Dukascopy and add H1 + H4 bars to a snapshot.

Our broker store only goes back months. Dukascopy publishes free historical
candles for FX, metals, indices and oil back to the mid-2000s. This adds
~15-20 years of hourly bars (and 4-hour bars built from them), so the model and
the trading methodologies see many market regimes, not only 2026.

Politeness and caching:
  - one monthly file per request, at most ~1 request per second, long pause on
    429/503 and stop for the night after repeated refusals (what is cached stays);
  - files are cached in data/cache/dukascopy/ (outside the snapshots), so only the
    current and previous month are downloaded again on later nights.

Format checks (no blind trust in a binary layout):
  - each 24-byte record holds a time offset, four prices and a volume; which of the
    four prices is high / low is detected from the data (high >= all, low <= all);
  - the price scale (power of ten) is chosen so that the latest close matches our
    broker price for that symbol (or Yahoo's); a symbol that matches neither is
    skipped and reported.

License note: Dukascopy historical data, internal research/training only, not
redistributed. If a model is shipped commercially, train it with --no-public or
confirm the data terms first. See docs/dataset-register.md.

  python scripts/fetch_dukascopy.py --snapshot data/market/<date>
  python scripts/fetch_dukascopy.py --snapshot ... --from-year 2010 --symbols EUR/USD,XAU/USD

Appends rows to <snapshot>/public_candles.csv (source "public:dukascopy:<instrument>").
"""
import argparse
import csv
import json
import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.training.checkpoint import sha256_of  # noqa: E402

BASE = "https://datafeed.dukascopy.com/datafeed"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) nics-ai-lab"}
COLS = ["source", "symbol", "interval", "candle_time", "open", "high", "low", "close", "volume", "bid", "ask"]
# our symbol -> Dukascopy instrument
INSTRUMENTS = {
    "EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY", "USD/CHF": "USDCHF",
    "USD/CAD": "USDCAD", "AUD/USD": "AUDUSD", "USD/CNH": "USDCNH", "EUR/JPY": "EURJPY",
    "GBP/JPY": "GBPJPY", "XAU/USD": "XAUUSD", "XAG/USD": "XAGUSD", "UKOUSDft": "BRENTCMDUSD",
    "US500": "USA500IDXUSD", "US100": "USATECHIDXUSD", "US30": "USA30IDXUSD",
    "GER40": "DEUIDXEUR", "JP225": "JPNIDXJPY", "HK50": "HKGIDXHKD", "BTC/USD": "BTCUSD",
}


class Refused(Exception):
    pass


def month_start(y: int, m: int) -> int:
    return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())


def download(url: str, dest: Path, pause: float) -> bytes | None:
    """File bytes, b"" for a month without data (404 / empty), None if refused repeatedly."""
    for attempt in range(3):
        time.sleep(pause)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"")
                return b""
            if e.code in (429, 503):
                time.sleep(60 * (attempt + 1))  # they asked us to slow down
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10 * (attempt + 1))
    return None


def decode(raw: bytes) -> np.ndarray:
    """(n, 6) float64: offset_s, p1..p4 (raw ints), volume."""
    if not raw:
        return np.zeros((0, 6))
    data = lzma.decompress(raw)
    n = len(data) // 24
    return np.array([struct.unpack(">5if", data[i * 24:(i + 1) * 24]) for i in range(n)], dtype=np.float64)


def ohlc_positions(recs: np.ndarray) -> tuple[int, int, int, int]:
    """Columns (1..4) of open, high, low, close, detected from the data."""
    p = recs[:, 1:5]
    hi = int(np.argmax([(p[:, j] >= p.max(axis=1)).mean() for j in range(4)]))
    lo = int(np.argmax([(p[:, j] <= p.min(axis=1)).mean() for j in range(4)]))
    rest = [j for j in range(4) if j not in (hi, lo)]
    if hi == lo or len(rest) != 2:
        raise ValueError("cannot tell high/low columns apart")
    return rest[0] + 1, hi + 1, lo + 1, rest[1] + 1


def reference_prices(snap: Path) -> dict[str, float]:
    """Latest close per symbol from broker candles (preferred) or the public file."""
    found = {}
    for name in ("public_candles.csv", "candles.csv"):
        path = snap / name
        latest: dict[str, tuple[str, float]] = {}
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("source", "").startswith("public:dukascopy") or not r.get("close"):
                        continue
                    if r["symbol"] not in latest or r["candle_time"] > latest[r["symbol"]][0]:
                        latest[r["symbol"]] = (r["candle_time"], float(r["close"]))
        found.update({k: v[1] for k, v in latest.items()})  # broker file read last, so it wins
    return found


def to_h4(t, o, h, lo, c, v):
    key = (t // 14400).astype(np.int64)
    out = []
    starts = np.flatnonzero(np.concatenate([[True], np.diff(key) != 0]))
    ends = np.concatenate([starts[1:], [len(t)]])
    for a, b in zip(starts, ends):
        out.append((int(key[a] * 14400), o[a], h[a:b].max(), lo[a:b].min(), c[b - 1], v[a:b].sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--from-year", type=int, default=2008)
    ap.add_argument("--symbols", default=None, help="comma-separated subset of our symbols")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--offline", action="store_true", help="use cached files only (tests)")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "dukascopy"))
    args = ap.parse_args()
    snap, cache = Path(args.snapshot), Path(args.cache)
    wanted = set(args.symbols.split(",")) if args.symbols else set(INSTRUMENTS)
    ref = reference_prices(snap)
    now = datetime.now(timezone.utc)
    report, refused = {}, 0
    out_path = snap / "public_candles.csv"
    new_file = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLS)
        for ours, inst in INSTRUMENTS.items():
            if ours not in wanted:
                continue
            if refused >= 3:
                report[ours] = "skipped: Dukascopy refused requests, retry next night"
                continue
            parts = []
            for y in range(args.from_year, now.year + 1):
                for m in range(1, 13):
                    if (y, m) > (now.year, now.month):
                        break
                    dest = cache / inst / f"{y}-{m:02d}.bi5"
                    recent = (now.year * 12 + now.month) - (y * 12 + m) <= 1
                    if dest.exists() and (not recent or args.offline):
                        raw = dest.read_bytes()
                    elif args.offline:
                        continue
                    else:
                        raw = download(f"{BASE}/{inst}/{y}/{m - 1:02d}/BID_candles_hour_1.bi5", dest, args.pause)
                        if raw is None:
                            refused += 1
                            break
                    try:
                        recs = decode(raw)
                    except lzma.LZMAError:
                        dest.unlink(missing_ok=True)  # broken or an error page: fetch again next time
                        continue
                    if len(recs):
                        recs[:, 0] += month_start(y, m)
                        parts.append(recs)
                if refused >= 3:
                    break
            if not parts:
                report[ours] = "no data"
                continue
            a = np.concatenate(parts)
            a = a[np.argsort(a[:, 0], kind="stable")]
            a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0])]
            a = a[a[:, 5] > 0]  # Dukascopy writes flat zero-volume bars for closed hours
            if len(a) < 500:
                report[ours] = f"too few bars ({len(a)})"
                continue
            try:
                io, ih, il, ic = ohlc_positions(a)
            except ValueError as e:
                report[ours] = f"skipped: {e}"
                continue
            last = a[-1, ic]
            if ours not in ref or last <= 0:
                report[ours] = "skipped: no reference price to check the scale"
                continue
            scale = 10.0 ** round(np.log10(last / ref[ours]))
            if abs(last / scale / ref[ours] - 1) > 0.15:  # same instrument, recent close within 15 %
                report[ours] = f"skipped: price {last / scale:.5g} does not match ours {ref[ours]:.5g}"
                continue
            t = a[:, 0].astype(np.int64)
            o, h, lo, c, v = (a[:, io] / scale, a[:, ih] / scale, a[:, il] / scale, a[:, ic] / scale, a[:, 5])
            keep = t + 3600 <= now.timestamp()  # drop the forming hour
            t, o, h, lo, c, v = t[keep], o[keep], h[keep], lo[keep], c[keep], v[keep]
            src = f"public:dukascopy:{inst}"
            for i in range(len(t)):
                ts = datetime.fromtimestamp(int(t[i]), timezone.utc).isoformat()
                w.writerow([src, ours, "1h", ts, o[i], h[i], lo[i], c[i], v[i], "", ""])
            h4 = [r for r in to_h4(t, o, h, lo, c, v) if r[0] + 14400 <= now.timestamp()]
            for r in h4:
                ts = datetime.fromtimestamp(r[0], timezone.utc).isoformat()
                w.writerow([src, ours, "4h", ts, *r[1:], "", ""])
            first = datetime.fromtimestamp(int(t[0]), timezone.utc).date().isoformat()
            report[ours] = {"1h": int(len(t)), "4h": len(h4), "from": first, "scale": scale,
                            "last_close": round(float(c[-1]), 6), "our_close": ref[ours]}
            print(f"{ours:10s} 1h {len(t):7d}  4h {len(h4):6d}  from {first}  ({inst})", flush=True)

    mpath = snap / "manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    m["dukascopy_history"] = {
        "dataset_id": "nics-dukascopy-history-v1",
        "downloaded_at": now.isoformat(),
        "from_year": args.from_year,
        "symbols": report,
        "license": "Dukascopy historical data, internal research/training only, not redistributed",
    }
    m.setdefault("sha256", {})["public_candles.csv"] = sha256_of(out_path)
    mpath.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    for k, v in report.items():
        if not isinstance(v, dict):
            print(f"{k:10s} {v}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
