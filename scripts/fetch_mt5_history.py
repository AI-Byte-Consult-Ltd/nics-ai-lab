"""Export OUR closed trades from a running MetaTrader 5 terminal into a snapshot.

Run on the Windows laptop where the MT5 terminal is open and logged in to
the account (prop or Vantage). Uses the official MetaTrader5 Python package
(pip install MetaTrader5), which reads the terminal's own trade history.
Nothing is sent anywhere; the account number is not written to the file.

  python scripts/fetch_mt5_history.py --snapshot data/market/<date> --label prop
  python scripts/fetch_mt5_history.py --snapshot data/market/<date> --label vantage --days 730

Run it once per account (log the terminal in to each account first, or pass
--terminal "C:\\Path\\to\\terminal64.exe" for a second installed terminal).

Writes <snapshot>/own_trades_<label>.csv, one row per closed position:
  open/close time and price, direction, volume, SL/TP at open (from the
  opening order), profit + commission + swap, EA magic number (0 = manual).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.training.checkpoint import sha256_of  # noqa: E402

COLS = ["label", "position_id", "symbol", "direction", "volume", "open_time", "open_price", "stop_loss",
        "take_profit", "close_time", "close_price", "profit", "commission", "swap", "net", "magic", "source"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--label", required=True, help="short account label, e.g. prop, vantage, algo-elena")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--terminal", default=None, help="path to terminal64.exe (optional)")
    args = ap.parse_args()
    try:
        import MetaTrader5 as mt5  # Windows only
    except ImportError:
        sys.exit("pip install MetaTrader5   (Windows, with the MT5 terminal installed)")
    ok = mt5.initialize(args.terminal) if args.terminal else mt5.initialize()
    if not ok:
        sys.exit(f"MT5 initialize failed: {mt5.last_error()}. Is the terminal open and logged in?")
    try:
        end = datetime.now(timezone.utc) + timedelta(days=1)
        start = end - timedelta(days=args.days + 1)
        deals = mt5.history_deals_get(start, end) or []
        orders = mt5.history_orders_get(start, end) or []
    finally:
        info = mt5.account_info()
        server = info.server if info else "unknown"
        mt5.shutdown()

    sltp = {}
    for o in sorted(orders, key=lambda o: o.time_setup):
        if o.position_id and o.position_id not in sltp and (o.sl or o.tp):
            sltp[o.position_id] = (o.sl or None, o.tp or None)
    by_pos = defaultdict(list)
    for d in deals:
        if d.position_id and d.type in (0, 1):  # buy / sell deals only (no balance, credit...)
            by_pos[d.position_id].append(d)

    rows = []
    for pid, ds in by_pos.items():
        ds.sort(key=lambda d: d.time_msc)
        ins = [d for d in ds if d.entry == 0]
        outs = [d for d in ds if d.entry in (1, 3)]
        if not ins or not outs:
            continue  # still open, or partial history
        vol_in = sum(d.volume for d in ins)
        vol_out = sum(d.volume for d in outs)
        if vol_out + 1e-9 < vol_in:
            continue  # partly closed
        op = sum(d.price * d.volume for d in ins) / vol_in
        cp = sum(d.price * d.volume for d in outs) / vol_out
        profit = sum(d.profit for d in ds)
        comm = sum(d.commission + getattr(d, "fee", 0.0) for d in ds)
        swap = sum(d.swap for d in ds)
        sl, tp = sltp.get(pid, (None, None))
        rows.append({
            "label": args.label, "position_id": pid, "symbol": ins[0].symbol,
            "direction": "LONG" if ins[0].type == 0 else "SHORT", "volume": round(vol_in, 4),
            "open_time": datetime.fromtimestamp(ins[0].time_msc / 1000, timezone.utc).isoformat(),
            "open_price": op, "stop_loss": sl or "", "take_profit": tp or "",
            "close_time": datetime.fromtimestamp(outs[-1].time_msc / 1000, timezone.utc).isoformat(),
            "close_price": cp, "profit": round(profit, 2), "commission": round(comm, 2), "swap": round(swap, 2),
            "net": round(profit + comm + swap, 2), "magic": ins[0].magic, "source": f"mt5:{server}",
        })
    rows.sort(key=lambda r: r["open_time"])
    snap = Path(args.snapshot)
    path = snap / f"own_trades_{args.label}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    mpath = snap / "manifest.json"
    m = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    m.setdefault("own_trades", {})[args.label] = {
        "positions": len(rows), "with_sl_tp": sum(1 for r in rows if r["stop_loss"] and r["take_profit"]),
        "server": server, "days": args.days, "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    m.setdefault("sha256", {})[path.name] = sha256_of(path)
    mpath.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"{len(rows)} closed positions -> {path}")


if __name__ == "__main__":
    main()
