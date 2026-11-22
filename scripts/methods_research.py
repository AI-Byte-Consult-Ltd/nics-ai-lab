"""Backtest the ten classic methods and our own methods on long daily history, honestly.

Reproduces docs/methods_research_2026-11-22.md from a lab snapshot (public D1 history in
public_candles.csv, broker candles in candles.csv for the spreads):

  python scripts/methods_research.py --snapshot data/market/<date>

In-sample = entries before 2018-01-01, out-of-sample = from 2018. History starts in 1990
(USD/CNH in 2011: before that the public series is the pegged onshore rate). Costs = 1.2 x
median broker spread (ask - bid) + swap per calendar day held (yearly rate by asset group).
Research only: nothing is sent anywhere and nothing changes the scanner.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import classic_methods as cm  # noqa: E402
from nics.market import data as market  # noqa: E402

OOS = datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp()
START = datetime(1989, 1, 1, tzinfo=timezone.utc).timestamp()
LATE_START = {"USD/CNH": datetime(2011, 1, 1, tzinfo=timezone.utc).timestamp()}
SWAP = {"forex": 0.02, "gold": 0.03, "metals": 0.03, "oil": 0.04, "indices": 0.04, "stocks": 0.05, "crypto": 0.20}
GROUP_OF = {"XAU/USD": "gold", "XAUUSD247": "gold", "XAG/USD": "metals", "COPPER": "metals", "UKOUSDft": "oil",
            "BTC/USD": "crypto", "ETH/USD": "crypto", "US500": "indices", "US100": "indices", "US30": "indices",
            "GER40": "indices", "JP225": "indices", "HK50": "indices"}


def group(sym: str) -> str:
    return GROUP_OF.get(sym) or ("forex" if "/" in sym else "stocks")


def stats(tr: list[dict]) -> dict:
    if not tr:
        return {"n": 0}
    r = np.array([x["r"] for x in tr])
    order = np.argsort([x["t_out"] for x in tr])
    eq = np.cumsum(r[order])
    dd = float(np.max(np.maximum.accumulate(np.r_[0, eq])[1:] - eq))
    loss = -r[r < 0].sum()
    t = float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 and r.std() > 0 else 0.0
    return {"n": int(len(r)), "R": round(float(r.sum()), 1), "avg": round(float(r.mean()), 3), "t": round(t, 2),
            "win": round(float((r > 0).mean()), 3), "pf": round(float(r[r > 0].sum() / loss), 2) if loss > 0 else None,
            "dd": round(dd, 1), "cost_r": round(float(np.mean([x["cost_r"] for x in tr])), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", default=str(ROOT / "shadow" / "methods_research.json"))
    args = ap.parse_args()
    series = market.load_series(Path(args.snapshot), use_public=True, intervals=["1day", "1h"])
    spread = {}
    for s in series:
        if not s.public and s.interval == "1h":
            ok = s.spread > 0
            if ok.any():
                spread[s.symbol] = float(np.median(s.spread[ok] / s.c[ok]))
    trades = defaultdict(list)
    allm = {**cm.NEW, **cm.OURS}
    for s in series:
        if not (s.public and s.interval == "1day"):
            continue
        m = s.t >= LATE_START.get(s.symbol, START)
        d = {"t": s.t[m], "o": s.o[m], "h": s.h[m], "l": s.l[m], "c": s.c[m]}
        if len(d["c"]) < 400 or s.symbol not in spread:
            print(f"{s.symbol}: skipped (history {len(d['c'])} days, spread known: {s.symbol in spread})")
            continue
        for name, fn in allm.items():
            for x in cm.simulate(d, fn(d), 1.2 * spread[s.symbol], SWAP[group(s.symbol)]):
                x["sym"] = s.symbol
                trades[name].append(x)
    res = {}
    for name in allm:
        tr = trades[name]
        oos = [x for x in tr if x["t_in"] >= OOS]
        per = defaultdict(float)
        for x in oos:
            per[x["sym"]] += x["r"]
        res[name] = {"in_sample": stats([x for x in tr if x["t_in"] < OOS]), "out_of_sample": stats(oos),
                     "oos_long": stats([x for x in oos if x["dir"] > 0]), "oos_short": stats([x for x in oos if x["dir"] < 0]),
                     "oos_instruments_positive": round(sum(v > 0 for v in per.values()) / max(1, len(per)), 2)}
        print(name, json.dumps(res[name]["in_sample"]), "->", json.dumps(res[name]["out_of_sample"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
