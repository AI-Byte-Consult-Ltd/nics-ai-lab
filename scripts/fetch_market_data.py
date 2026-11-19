"""Download a snapshot of ALL our trading data for NICS Trading.

The n8n workflow "[41 LAB] Market Data Export" serves it; it needs the lab
export key (n8n: open the workflow, run "Create or Show Lab Key").

What is downloaded (all from our own systems):
  candles.csv          Vantage MT5 bridge candles, every instrument, every
                       timeframe the bridge sends (1min, 5min, 15min, 1h, 4h)
  signals.csv          every NICS scanner signal (scheduled, on-request, scalper,
                       strong and weak) with levels, status and recorded result
  signal_events.csv    signal lifecycle: TP1..TP4 hits, SL, breakeven, time exits
  swing_shadow.csv     our shadow swing strategies, trade by trade with R
  manual_algo.csv      manual signals executed by our EA, with P/L
  economic_events.csv  economic calendar we use for news protection
  journal.csv          our trade journal notes
  instrument_specs.csv contract sizes, tick values, broker symbol names

Public long history (D1/W1) is fetched separately: scripts/fetch_public_history.py.
Our own account trades from the MT5 terminal: scripts/fetch_mt5_history.py.

  set NICS_DATA_URL=https://n8n.aibyteconsult.com/webhook/nics-lab-market-export
  set NICS_DATA_KEY=<lab export key>          (never commit it)
  python scripts/fetch_market_data.py                     (everything)
  python scripts/fetch_market_data.py --intervals 15min,1h,4h --days 365

Writes data/market/<UTC date>/ + manifest.json with sha256 of every file.
data/ is git-ignored: the data stays on this machine.
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.training.checkpoint import sha256_of  # noqa: E402

PAGE = 50000
TABLES = {
    # mode -> (file name, columns)
    "signals": ("signals.csv", [
        "id", "symbol", "label", "asset_class", "plan_group", "direction", "order_type", "entry_low",
        "entry_high", "stop_loss", "original_stop_loss", "tp1", "tp2", "tp3", "tp4", "confidence_score",
        "technical_score", "qwen_approved", "qwen_confidence_adjustment", "risk_profile", "status",
        "final_result", "hit_targets", "breakeven_activated", "origin_mode", "strategy_version",
        "market_session", "created_at", "closed_at", "valid_until", "trade_expires_at", "market_data_time",
        "initial_risk", "spread_at_signal", "slippage_allowance", "exit_price", "realized_r", "mfe_r",
        "mae_r", "delivery_status", "opportunity_score", "strategy_mode", "execution_timeframe",
        "production_gate_passed", "approved_for_publication"]),
    "signal_events": ("signal_events.csv", ["id", "signal_id", "event_type", "price", "event_time"]),
    "swing_shadow": ("swing_shadow.csv", [
        "id", "strategy", "symbol", "direction", "signal_bar_time", "signal_close", "day_atr", "stop_loss",
        "entry_time", "entry_price", "risk", "take_profit", "spread", "cost_r", "status", "bars_held",
        "exit_time", "exit_price", "r_gross", "r_net", "created_at"]),
    "manual_algo": ("manual_algo.csv", [
        "id", "symbol", "direction", "order_type", "entry_price", "stop_loss", "take_profit", "requested_lot",
        "status", "ea_lot", "ea_fill_price", "close_reason", "close_price", "pnl_usd", "created_at",
        "closed_at"]),
    "economic_events": ("economic_events.csv", [
        "event_key", "currency", "title", "impact", "event_time", "forecast", "previous"]),
    "journal": ("journal.csv", ["id", "entry_type", "signal_id", "note", "post_text_en", "created_at"]),
    "instrument_specs": ("instrument_specs.csv", [
        "symbol", "broker_symbol", "market_group", "contract_size", "tick_size", "tick_value", "volume_min",
        "volume_step", "profit_currency"]),
}
CANDLE_COLS = ["source", "symbol", "interval", "candle_time", "open", "high", "low", "close", "volume", "bid", "ask"]


def fetch(url: str, key: str, params: dict) -> dict:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"X-NICS-Lab-Key": key})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 403:
                sys.exit("403: wrong NICS_DATA_KEY (n8n: [41 LAB] Market Data Export -> Create or Show Lab Key).")
            if attempt == 3:
                raise
        except urllib.error.URLError:
            if attempt == 3:
                raise
        time.sleep(2 ** attempt * 5)
    raise RuntimeError("unreachable")


def cell(v):
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intervals", default="all", help="comma list, e.g. 15min,1h,4h (default: all)")
    ap.add_argument("--days", type=int, default=3650, help="history depth (default: everything, up to 10 years)")
    ap.add_argument("--out", default=None, help="output directory (default data/market/<UTC date>)")
    ap.add_argument("--tables", default="all", help="comma list of extra tables (default all), or 'none'")
    args = ap.parse_args()
    url, key = os.environ.get("NICS_DATA_URL"), os.environ.get("NICS_DATA_KEY")
    if not url or not key:
        sys.exit("Set NICS_DATA_URL and NICS_DATA_KEY first (see the docstring).")

    out = Path(args.out) if args.out else ROOT / "data" / "market" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    listing = fetch(url, key, {"mode": "symbols", "days": args.days})
    pairs = [(r["symbol"], r["interval"]) for r in listing.get("rows") or []]
    if args.intervals != "all":
        wanted = {s.strip() for s in args.intervals.split(",")}
        pairs = [p for p in pairs if p[1] in wanted]
    if not pairs:
        sys.exit(f"No candles found. Available: {listing.get('rows')}")

    counts = {"candles": {}}
    with open(out / "candles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CANDLE_COLS, extrasaction="ignore")
        w.writeheader()
        for sym, iv in pairs:
            after, n = "", 0
            while True:
                rows = fetch(url, key, {"mode": "candles", "symbol": sym, "interval": iv, "days": args.days,
                                        "after": after, "limit": PAGE}).get("rows") or []
                for r in rows:
                    r["source"] = "vantage_mt5"
                    w.writerow({c: cell(r.get(c)) for c in CANDLE_COLS})
                n += len(rows)
                if len(rows) < PAGE:
                    break
                after = rows[-1]["candle_time"]
            counts["candles"][f"{sym}|{iv}"] = n
            print(f"{sym:10s} {iv:6s} {n:8d} candles")

    tables = TABLES if args.tables == "all" else {
        m: v for m, v in TABLES.items() if m in {t.strip() for t in args.tables.split(",")}}
    for mode, (name, cols) in tables.items():
        rows = fetch(url, key, {"mode": mode, "days": args.days}).get("rows") or []
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: cell(r.get(c)) for c in cols})
        counts[mode] = len(rows)
        print(f"{mode:16s} {len(rows):8d} rows")

    files = ["candles.csv"] + [n for n, _ in tables.values()]
    manifest = {
        "dataset_id": "nics-market-v2",
        "source": "own systems: Vantage MT5 bridge candles, NICS scanner signals + lifecycle events, "
                  "shadow strategies, EA manual signals, economic calendar, trade journal",
        "exported_at": listing.get("exported_at"),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "days": args.days,
        "intervals": args.intervals,
        "rows": counts,
        "sha256": {n: sha256_of(out / n) for n in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Snapshot: {out}")
    print("Next (optional): python scripts/fetch_public_history.py --snapshot", out)
    print("          and: python scripts/fetch_mt5_history.py --snapshot", out, "--label prop")


if __name__ == "__main__":
    main()
