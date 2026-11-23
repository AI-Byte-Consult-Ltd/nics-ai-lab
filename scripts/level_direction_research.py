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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import data as market  # noqa: E402
from nics.market import level_direction as ld  # noqa: E402
from nics.telemetry.push import push  # noqa: E402

GROUP = {"XAU/USD": "metals", "XAG/USD": "metals", "UKOUSDft": "oil", "BTC/USD": "crypto", "US500": "indices",
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
    L = [f"Метод уровней и правило направления на часовой истории {meta['from']}–{meta['to']}, "
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
    trades = {v: [] for v in variants}
    states: list[tuple] = []
    used, t_min, t_max = [], None, None
    for s in series:
        if not s.public or (only and s.symbol not in only):
            continue
        keep = ~((s.h == s.l) & (s.v == 0)) & (s.t >= start - 400 * 86400)  # no flat weekend hours; 400 days of warm-up
        if keep.sum() < 24 * 400:
            print(f"  {s.symbol}: only {int(keep.sum())} hourly bars, skipped", flush=True)
            continue
        t0 = time.time()
        p = ld.prepare(s.t[keep], s.h[keep], s.l[keep], s.c[keep])
        if p is None:
            continue
        spread = spreads.get(s.symbol, 0.0)
        tr, st = ld.run_symbol(p, spread, variants, start)
        g = GROUP.get(s.symbol) or ("forex" if "/" in s.symbol else "stocks")
        for v in variants:
            for x in tr[v]:
                x["symbol"], x["group"] = s.symbol, g
            trades[v].extend(tr[v])
        states.extend(st)
        used.append(s.symbol)
        a, b = float(p.t[max(int(np.searchsorted(p.t, start)), 0)]), float(p.t[-1])
        t_min, t_max = (a if t_min is None else min(t_min, a)), (b if t_max is None else max(t_max, b))
        print(f"  {s.symbol}: {len(p.t)} hours, base trades {len(tr[variants[0]])}, spread {spread:g}, {time.time() - t0:.0f}s", flush=True)
    if not used:
        sys.exit("no public hourly series in the snapshot (Dukascopy history is fetched by the nightly cycle)")
    fmt = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")  # noqa: E731
    meta = {"snapshot": snap.name, "symbols": len(used), "symbol_list": used, "from": fmt(t_min), "to": fmt(t_max),
            "params": ld.PARAMS, "note": "Research on past public candles, not a forecast. A stop inside the fill "
            "bar counts as a loss (pessimistic for every variant). fwd = 24 h move after the fill in daily ATR."}
    summary = {"meta": meta, **summarise(trades, states)}
    text = russian(summary, meta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "level_direction_last.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "level_direction_last.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.no_push:
        def c(x):  # compact rows keep the dashboard record small: n, R, stops %, fwd, fwd t
            if isinstance(x, dict) and "n" in x:
                return [x["n"], x.get("R", x.get("move", 0)), x.get("stops_pct", x.get("t", 0)), x.get("fwd", 0), x.get("fwd_t", 0)]
            return {k: c(v) for k, v in x.items()} if isinstance(x, dict) else x
        slim = {"meta": {k: meta[k] for k in ("snapshot", "symbols", "from", "to")},
                "cols": "n, R (or move), stops % (or t), fwd, fwd t",
                "variants": {v: c({k: o[k] for k in ("all", "long", "short", "era", "group")}) for v, o in summary["variants"].items()},
                "base_by_state": c(summary["base_by_state"]), "states": c(summary["states"])}
        push("NICS Trading", "level_direction", slim, text, cycle=snap.name)


if __name__ == "__main__":
    main()
