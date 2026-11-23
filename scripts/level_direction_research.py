"""Level method with different direction rules on public hourly history since 2008.

Research only: nothing is sent to users, the scanner, the level method or any EA.
See src/nics/market/level_direction.py for the rules and the honesty notes.

  python scripts/level_direction_research.py                       (latest snapshot in data/market)
  python scripts/level_direction_research.py --snapshot data/market/<date> --symbols XAU/USD,US500

Writes experiments/research/level_direction_last.json and .txt, and sends the short summary to
the private dashboard (same channel as the nightly report; --no-push to skip).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import data as market  # noqa: E402
from nics.market import level_direction as ld  # noqa: E402
from nics.telemetry.push import push  # noqa: E402

YAHOO = {"XAU/USD": "GC=F", "XAG/USD": "SI=F", "COPPER": "HG=F", "UKOUSDft": "BZ=F", "BTC/USD": "BTC-USD",
         "ETH/USD": "ETH-USD", "US500": "^GSPC", "US100": "^NDX", "US30": "^DJI", "GER40": "^GDAXI", "JP225": "^N225",
         "HK50": "^HSI", "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "USD/CHF": "CHF=X",
         "USD/CAD": "CAD=X", "AUD/USD": "AUDUSD=X", "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X", "NVIDIA": "NVDA",
         "TSLA": "TSLA", "META": "META", "AMAZON": "AMZN", "NFLX": "NFLX", "AMD": "AMD", "AVGO": "AVGO", "ASML": "ASML",
         "AMAT": "AMAT", "LRCX": "LRCX", "MU": "MU", "TSM": "TSM", "DELL": "DELL", "SMCI": "SMCI"}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) nics-ai-lab"}


def yahoo_h1(sym: str, cache: Path):
    """About two years of public hourly bars (the provider keeps no more). Cached for the day.

    Public data, internal research only, not redistributed (docs/dataset-register.md)."""
    dest = cache / (YAHOO[sym].replace("^", "").replace("=", "_") + "_" + datetime.now(timezone.utc).strftime("%Y%m%d") + ".json")
    if dest.exists():
        res = json.loads(dest.read_text(encoding="utf-8"))
    else:
        now = int(time.time())
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(YAHOO[sym])}"
               f"?period1={now - 728 * 86400}&period2={now}&interval=1h")
        time.sleep(1.0)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            res = json.loads(r.read().decode("utf-8"))["chart"]["result"][0]
        cache.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"timestamp": res.get("timestamp"), "indicators": res["indicators"]}), encoding="utf-8")
    q = res["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(res.get("timestamp") or []):
        h, lo, c = q["high"][i], q["low"][i], q["close"][i]
        if None in (h, lo, c) or not (h >= lo > 0):
            continue
        rows.append((int(t) // 3600 * 3600, float(h), float(lo), float(c)))
    rows.sort()
    a = np.array(rows, dtype=np.float64)
    if len(a) < 2:
        return None
    a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0])]
    return a[:-1]  # drop the still-forming hour


GROUP = {"COPPER": "metals", "ETH/USD": "crypto", "XAU/USD": "metals", "XAG/USD": "metals", "UKOUSDft": "oil", "BTC/USD": "crypto", "US500": "indices",
         "US100": "indices", "US30": "indices", "GER40": "indices", "JP225": "indices", "HK50": "indices"}
ERAS = (("2008-2013", 2008, 2014), ("2014-2019", 2014, 2020), ("2020-2026", 2020, 2100))
STATE_RU = {(-1, 1): "короткий вниз, большой вверх (откат)", (-1, -1): "оба вниз", (1, 1): "оба вверх",
            (1, -1): "короткий вверх, большой вниз", (0, 1): "без тренда, большой вверх", (0, -1): "без тренда, большой вниз"}


def era_of(ts: float) -> str:
    y = datetime.fromtimestamp(ts, timezone.utc).year
    return next((name for name, a, b in ERAS if a <= y < b), "other")


def stats(tr: list[dict]) -> dict:
    if not tr:
        return {"n": 0}
    r = np.array([x["r"] for x in tr])
    f = np.array([x["fwd"] for x in tr])
    sl = np.array([x["why"] == "SL" for x in tr])
    fb = np.array([x["why"] == "SL" and x["fill_bar"] for x in tr])
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
    return {"n": len(tr), "R": round(float(r.sum()), 1), "avg": round(float(r.mean()), 3),
            "t": round(float(r.mean() / se), 1) if se > 0 else 0.0, "stops_pct": round(100 * float(sl.mean())),
            "fill_bar_stops_pct": round(100 * float(fb.mean())),
            "R_without_fill_bar_stops": round(float(r[~fb].sum()), 1),
            "fwd": round(float(f.mean()), 3),
            "fwd_t": round(float(f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))), 1) if len(f) > 1 and f.std() > 0 else 0.0}


def state_stats(rows: list[tuple]) -> dict:
    if not rows:
        return {"n": 0}
    m = np.array([x[5] for x in rows])
    se = m.std(ddof=1) / np.sqrt(len(m) / 6) if len(m) > 12 else 0.0  # windows overlap about six times
    return {"n": len(m), "move": round(float(m.mean()), 3), "t": round(float(m.mean() / se), 1) if se > 0 else 0.0}


def summarise(trades: dict[str, list], states: list[tuple]) -> dict:
    out: dict = {"variants": {}, "base_by_state": {}, "states": {}}
    for v, tr in trades.items():
        o = {"all": stats(tr), "long": stats([x for x in tr if x["dir"] > 0]),
             "short": stats([x for x in tr if x["dir"] < 0]), "era": {}, "group": {}, "symbol": {}}
        for name, _, _ in ERAS:
            sub = [x for x in tr if era_of(x["t"]) == name]
            o["era"][name] = {"all": stats(sub), "long": stats([x for x in sub if x["dir"] > 0]),
                              "short": stats([x for x in sub if x["dir"] < 0])}
        by_g, by_s = defaultdict(list), defaultdict(list)
        for x in tr:
            by_g[x["group"]].append(x); by_s[x["symbol"]].append(x)
        for g, sub in sorted(by_g.items()):
            o["group"][g] = {"long": stats([x for x in sub if x["dir"] > 0]), "short": stats([x for x in sub if x["dir"] < 0])}
        for s, sub in sorted(by_s.items()):
            st = stats(sub)
            o["symbol"][s] = [st["n"], st.get("R", 0), st.get("fwd", 0)]
        out["variants"][v] = o
    for (b, lt), name in STATE_RU.items():
        sub = [x for x in trades.get("base", []) if x["base"] == b and x["lt"] == lt]
        out["base_by_state"][f"{b}|{lt}"] = {"name": name, "long": stats([x for x in sub if x["dir"] > 0]),
                                             "short": stats([x for x in sub if x["dir"] < 0])}
        rows = [x for x in states if x[1] == b and x[2] == lt]
        o = {"name": name, "all": state_stats(rows), "era": {n: state_stats([x for x in rows if era_of(x[0]) == n]) for n, _, _ in ERAS}}
        if b < 0:
            o["by_depth_below_ema20"] = {k: state_stats([x for x in rows if a <= -x[3] < c])
                                         for k, a, c in (("0-1", -99, 1), ("1-2", 1, 2), ("2-3", 2, 3), ("3+", 3, 99))}
            o["by_rebound_from_3d_low"] = {k: state_stats([x for x in rows if a <= x[4] < c])
                                           for k, a, c in (("0-0.5", -99, 0.5), ("0.5-1", 0.5, 1), ("1-1.5", 1, 1.5), ("1.5+", 1.5, 99))}
        out["states"][f"{b}|{lt}"] = o
    return out


def russian(s: dict, meta: dict) -> str:
    L = [f"{'Длинная история' if meta.get('set') == 'long' else 'Два года, все инструменты'}: "
         f"метод уровней и правило направления на часовой истории {meta['from']}–{meta['to']}, "
         f"инструментов {meta['symbols']}. Проверка на прошлом, не прогноз. Стоп на свече входа считается убытком, "
         "поэтому числа занижены одинаково для всех вариантов: сравнивать варианты между собой."]
    names = {"base": "как сейчас", "pullback": "откат в большом росте = покупка", "no_low": "без шорта у минимума",
             "rule": "новое правило целиком", "lt_only": "только большой тренд"}
    for v, o in s["variants"].items():
        a, lg, sh = o["all"], o["long"], o["short"]
        if not a["n"]:
            continue
        L.append(f"{names.get(v, v)}: сделок {a['n']}, итог {a['R']:+}R, стопов {a['stops_pct']} %; "
                 f"покупки {lg.get('n', 0)} / {lg.get('R', 0):+}R, продажи {sh.get('n', 0)} / {sh.get('R', 0):+}R; "
                 f"направление без стопа {a['fwd']:+} ATR (t {a['fwd_t']})")
    groups = sorted({g for o in s["variants"].values() for g in o.get("group", {})})
    if len(groups) > 1:
        gname = {"metals": "металлы", "indices": "индексы", "stocks": "акции", "forex": "валюты", "crypto": "крипто", "oil": "нефть"}
        L.append("По группам, направление без стопа в ATR (t, сделок): как сейчас / новое правило")
        for g in groups:
            for side, ru in (("long", "покупки"), ("short", "продажи")):
                cell = []
                for v in ("base", "rule"):
                    x = s["variants"].get(v, {}).get("group", {}).get(g, {}).get(side)
                    cell.append(f"{x['fwd']:+} ({x['fwd_t']}, {x['n']})" if x and x.get("n") else "нет")
                L.append(f"  {gname.get(g, g)}, {ru}: {cell[0]} / {cell[1]}")
    L.append("Куда шёл рынок за следующие сутки (в дневных ATR, плюс = рост):")
    for o in s["states"].values():
        if o["all"]["n"]:
            L.append(f"  {o['name']}: {o['all']['move']:+} (t {o['all']['t']}, часов {o['all']['n']})")
    return "\n".join(L)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Russian summary on a Windows console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--from-year", type=int, default=2008)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--variants", default=",".join(ld.VARIANTS))
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "research"))
    ap.add_argument("--no-yahoo", action="store_true", help="skip the two-year hourly download")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "public_h1"))
    args = ap.parse_args()
    if args.snapshot:
        snap = Path(args.snapshot)
    else:
        snaps = sorted(p for p in (ROOT / "data" / "market").glob("*") if (p / "public_candles.csv").exists())
        if not snaps:
            sys.exit("no snapshot with public_candles.csv in data/market; run scripts/self_learn.bat once first")
        snap = snaps[-1]
    only = {x.strip() for x in args.symbols.split(",") if x.strip()}
    variants = tuple(v for v in args.variants.split(",") if v in ld.VARIANTS)
    start = datetime(args.from_year, 1, 1, tzinfo=timezone.utc).timestamp()
    print(f"snapshot {snap.name}: loading hourly bars ...", flush=True)
    series = market.load_series(snap, use_public=True, intervals=["1h"])
    spreads: dict[str, float] = {}
    for s in series:
        if not s.public and (s.spread > 0).any():
            spreads[s.symbol] = float(np.median(s.spread[s.spread > 0]))
    sets = {"long": {"trades": {v: [] for v in variants}, "states": [], "used": [], "t": [None, None]},
            "two_years": {"trades": {v: [] for v in variants}, "states": [], "used": [], "t": [None, None]}}

    def run_one(sym: str, t, h, lo, c, key: str, begin: float) -> None:
        t0 = time.time()
        p = ld.prepare(t, h, lo, c)
        if p is None:
            print(f"  {sym}: history too short, skipped", flush=True)
            return
        spread = spreads.get(sym, 0.0)
        tr, st = ld.run_symbol(p, spread, variants, begin)
        g = GROUP.get(sym) or ("forex" if "/" in sym else "stocks")
        S = sets[key]
        for v in variants:
            for x in tr[v]:
                x["symbol"], x["group"] = sym, g
            S["trades"][v].extend(tr[v])
        S["states"].extend(st)
        S["used"].append(sym)
        a, b = float(p.t[min(int(np.searchsorted(p.t, begin)), len(p.t) - 1)]), float(p.t[-1])
        S["t"] = [a if S["t"][0] is None else min(S["t"][0], a), b if S["t"][1] is None else max(S["t"][1], b)]
        print(f"  {sym} [{key}]: {len(p.t)} hours, base trades {len(tr[variants[0]])}, spread {spread:g}, {time.time() - t0:.0f}s", flush=True)

    for s in series:
        if not s.public or (only and s.symbol not in only):
            continue
        keep = ~((s.h == s.l) & (s.v == 0)) & (s.t >= start - 400 * 86400)  # no flat weekend hours; 400 days of warm-up
        if keep.sum() < 24 * 400:
            print(f"  {s.symbol}: only {int(keep.sum())} hourly bars, skipped", flush=True)
            continue
        run_one(s.symbol, s.t[keep], s.h[keep], s.l[keep], s.c[keep], "long", start)
    if not args.no_yahoo:
        print("two years of public hourly bars for every instrument ...", flush=True)
        for sym in YAHOO:
            if only and sym not in only:
                continue
            try:
                a = yahoo_h1(sym, Path(args.cache))
            except Exception as e:  # one instrument without data must not stop the rest
                print(f"  {sym}: public hourly bars not available ({type(e).__name__})", flush=True)
                continue
            if a is None or len(a) < 2500:
                print(f"  {sym}: too few hourly bars, skipped", flush=True)
                continue
            run_one(sym, a[:, 0], a[:, 1], a[:, 2], a[:, 3], "two_years", 0.0)
    if not sets["long"]["used"] and not sets["two_years"]["used"]:
        sys.exit("no public hourly history: neither the long cache nor the two-year download gave data")
    fmt = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")  # noqa: E731
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full, slim, texts = {}, {}, []

    def c(x):  # compact rows keep the dashboard record small: n, R, stops %, fwd, fwd t
        if isinstance(x, dict) and "n" in x:
            return [x["n"], x.get("R", x.get("move", 0)), x.get("stops_pct", x.get("t", 0)), x.get("fwd", 0), x.get("fwd_t", 0)]
        return {k: c(v) for k, v in x.items()} if isinstance(x, dict) else x

    for key, S in sets.items():
        if not S["used"]:
            continue
        meta = {"snapshot": snap.name, "set": key, "symbols": len(S["used"]), "symbol_list": S["used"],
                "from": fmt(S["t"][0]), "to": fmt(S["t"][1]), "params": ld.PARAMS,
                "note": "Research on past public candles, not a forecast. A stop inside the fill bar counts as a "
                        "loss (pessimistic for every variant). fwd = 24 h move after the fill bar in daily ATR."}
        summary = {"meta": meta, **summarise(S["trades"], S["states"])}
        full[key] = summary
        texts.append(russian(summary, meta))
        slim[key] = {"meta": {k: meta[k] for k in ("symbols", "from", "to")},
                     "variants": {v: c({k: o[k] for k in (("all", "long", "short", "era", "group") if key == "long" else ("all", "long", "short", "group"))})
                                  for v, o in summary["variants"].items()},
                     "base_by_state": c(summary["base_by_state"]),
                     "states": c({k: {kk: vv for kk, vv in o.items() if kk != "era"} for k, o in summary["states"].items()})}
    text = "\n\n".join(texts)
    (out / "level_direction_last.json").write_text(json.dumps(full, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "level_direction_last.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.no_push:
        slim["cols"] = "n, R (or move), stops % (or t), fwd, fwd t"
        push("nics-trading", "eval", slim, text, cycle=f"level_direction {snap.name}")


if __name__ == "__main__":
    main()
