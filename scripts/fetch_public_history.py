"""Download PUBLIC long history (D1 and W1) for our instruments into a snapshot.

Our broker store only goes back months (bridge started mid-2026), so years
of daily and weekly bars come from public market data, the same sources
we already use for manual level analysis:
  - Kraken public API (no key): BTC/USD, ETH/USD
  - Yahoo Finance chart API: indices, gold/silver/Brent futures, stocks, FX

These are NOT our broker's prices (futures vs spot, index vs CFD, small
offsets). Features are scale-free (ATR units), so the shape of the market
is what the model learns from them. Every row keeps its source.

License note: public data for internal research/training. Yahoo's terms
do not allow redistribution; nothing here is republished. If a model is
shipped commercially, train it with --no-public (scripts/train_trading.py)
or confirm the data terms first. See docs/dataset-register.md.

  python scripts/fetch_public_history.py --snapshot data/market/<date>

Writes <snapshot>/public_candles.csv and adds it to <snapshot>/manifest.json.
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.training.checkpoint import sha256_of  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) nics-ai-lab"}
# our symbol -> (provider, provider symbol)
SOURCES = {
    "BTC/USD": ("kraken", "XBTUSD"),
    "ETH/USD": ("kraken", "ETHUSD"),
    "US500": ("yahoo", "^GSPC"),
    "US100": ("yahoo", "^NDX"),
    "US30": ("yahoo", "^DJI"),
    "GER40": ("yahoo", "^GDAXI"),
    "JP225": ("yahoo", "^N225"),
    "HK50": ("yahoo", "^HSI"),
    "XAU/USD": ("yahoo", "GC=F"),
    "XAG/USD": ("yahoo", "SI=F"),
    "UKOUSDft": ("yahoo", "BZ=F"),
    "NVIDIA": ("yahoo", "NVDA"),
    "TSLA": ("yahoo", "TSLA"),
    "META": ("yahoo", "META"),
    "AMAZON": ("yahoo", "AMZN"),
    "NFLX": ("yahoo", "NFLX"),
    "EUR/USD": ("yahoo", "EURUSD=X"),
    "GBP/USD": ("yahoo", "GBPUSD=X"),
    "USD/JPY": ("yahoo", "JPY=X"),
    "USD/CHF": ("yahoo", "CHF=X"),
    "USD/CAD": ("yahoo", "CAD=X"),
    "AUD/USD": ("yahoo", "AUDUSD=X"),
    "USD/CNH": ("yahoo", "CNY=X"),  # offshore CNH has no public history there; onshore CNY as proxy
    "EUR/JPY": ("yahoo", "EURJPY=X"),
    "GBP/JPY": ("yahoo", "GBPJPY=X"),
}
COLS = ["source", "symbol", "interval", "candle_time", "open", "high", "low", "close", "volume", "bid", "ask"]
PERIOD = {"1day": 86400, "1week": 7 * 86400}


def get_json(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def kraken(pair: str, interval: str):
    minutes = 1440 if interval == "1day" else 10080
    res = get_json(f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={minutes}")["result"]
    rows = next(v for k, v in res.items() if k != "last")
    return [(int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[6])) for r in rows]


def yahoo(sym: str, interval: str):
    # D1: last 10 years. W1: everything (range=max thins weekly bars out, so ask by dates).
    span = "range=10y&interval=1d" if interval == "1day" else f"period1=0&period2={int(time.time())}&interval=1wk"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(sym)}?{span}"
    res = get_json(url)["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    out = []
    for i, t in enumerate(res.get("timestamp") or []):
        o, h, lo, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, lo, c) or not (h >= lo > 0):
            continue
        out.append((int(t), float(o), float(h), float(lo), float(c), float(q["volume"][i] or 0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args()
    snap = Path(args.snapshot)
    now = time.time()
    counts = {}
    with open(snap / "public_candles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for ours, (prov, theirs) in SOURCES.items():
            for interval in ("1day", "1week"):
                try:
                    rows = kraken(theirs, interval) if prov == "kraken" else yahoo(theirs, interval)
                except Exception as e:  # one dead symbol must not stop the rest
                    print(f"{ours:10s} {interval:5s} FAILED ({prov} {theirs}): {e}")
                    continue
                rows = [r for r in rows if r[0] + PERIOD[interval] <= now]  # drop the still-forming bar
                for t, o, h, lo, c, v in rows:
                    ts = datetime.fromtimestamp(t, timezone.utc).isoformat()
                    w.writerow([f"public:{prov}:{theirs}", ours, interval, ts, o, h, lo, c, v, "", ""])
                counts[f"{ours}|{interval}"] = len(rows)
                print(f"{ours:10s} {interval:5s} {len(rows):6d} bars  ({prov} {theirs})")
                time.sleep(0.5)

    mpath = snap / "manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    m["public_history"] = {
        "dataset_id": "nics-public-history-v1",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "sources": {k: f"{p}:{s}" for k, (p, s) in SOURCES.items()},
        "rows": counts,
        "license": "public market data, internal research/training only, not redistributed",
    }
    m.setdefault("sha256", {})["public_candles.csv"] = sha256_of(snap / "public_candles.csv")
    mpath.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"Wrote {snap / 'public_candles.csv'}")


if __name__ == "__main__":
    main()
