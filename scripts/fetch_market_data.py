"""Download a snapshot of OUR market data (Vantage MT5 candles + scanner signals).

The n8n workflow "[41 LAB] Market Data Export" serves the data; it needs the
lab export key. Nothing is downloaded from third parties.

  set NICS_DATA_URL=https://<n8n host>/webhook/nics-lab-market-export
  set NICS_DATA_KEY=<lab export key>          (never commit it)
  python scripts/fetch_market_data.py --interval M15 --days 120

Writes data/market/<UTC date>/{candles.csv, signals.csv, manifest.json}.
data/ is git-ignored: the data stays on this machine.
"""
import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.training.checkpoint import sha256_of  # noqa: E402

CANDLE_COLS = ["symbol", "interval", "candle_time", "open", "high", "low", "close", "volume", "bid", "ask"]
SIGNAL_COLS = ["id", "symbol", "direction", "entry_low", "entry_high", "stop_loss", "tp1",
               "confidence_score", "technical_score", "status", "final_result", "hit_targets",
               "created_at", "closed_at", "origin_mode"]


def fetch(url: str, key: str, params: dict) -> dict:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"X-NICS-Lab-Key": key})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="M15")
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()
    url, key = os.environ.get("NICS_DATA_URL"), os.environ.get("NICS_DATA_KEY")
    if not url or not key:
        sys.exit("Set NICS_DATA_URL and NICS_DATA_KEY first (see the docstring).")

    data = fetch(url, key, {"interval": args.interval, "days": args.days})
    out = ROOT / "data" / "market" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "candles.csv", CANDLE_COLS, data.get("candles", []))
    write_csv(out / "signals.csv", SIGNAL_COLS, data.get("signals", []))
    manifest = {
        "dataset_id": "nics-market-v1",
        "source": "own systems: Vantage MT5 bridge candles + NICS scanner signals",
        "exported_at": data.get("exported_at"),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "interval": args.interval,
        "days": args.days,
        "rows": {"candles": len(data.get("candles", [])), "signals": len(data.get("signals", []))},
        "sha256": {"candles.csv": sha256_of(out / "candles.csv"), "signals.csv": sha256_of(out / "signals.csv")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Snapshot: {out}")


if __name__ == "__main__":
    main()
