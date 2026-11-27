"""Risk discipline and trade management from two trading-psychology books, tested on our data.

Research only: nothing is sent to users, the scanner or any EA. See src/nics/market/discipline.py
for the rules, the honesty notes and why no book text or author names appear anywhere.

  python scripts/discipline_research.py                       (latest snapshot with public_candles.csv)
  python scripts/discipline_research.py --snapshot data/market/<date> --trades data/research/scanner_replay/trades

Part 1 (needs --trades, the 4.5-year scanner replay, one .jsonl per instrument): streaks,
sizing rules and the 20-trade edge window on 31,927 strong scanner trades.
Part 2 (hourly bars in <snapshot>/public_candles.csv, interval 1h): trade management variants
on the same trend entries and on random entries (control).
Writes experiments/research/discipline_last.json and .txt and sends the summary to the private
dashboard (--no-push to skip).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nics.market import data as market  # noqa: E402
from nics.market import discipline as dsc  # noqa: E402
from nics.telemetry.push import push  # noqa: E402

# Typical broker spread as a fraction of price (median of our Vantage quotes; replay research 26.11).
SPREAD = {"XAU/USD": 0.000095, "XAG/USD": 0.0008, "US500": 0.000083, "US100": 0.000065, "GER40": 0.0001,
          "UKOUSDft": 0.00055, "JP225": 0.00023, "HK50": 0.00023, "FRA40": 0.00018, "BTC/USD": 0.00028,
          "ETH/USD": 0.0015, "SOLUSD": 0.002, "US30": 0.0001, "COPPER": 0.0006}
SPLIT = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp()


def part_trades(folder: Path) -> dict:
    by: dict[str, list] = {}
    for f in sorted(folder.glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if "P0" not in d.get("b", []):  # strong signals as the scanner sends them today
                continue
            by.setdefault(d["sym"], []).append((d["t"], d["v"]["base"][0]))
    seqs = {s: [r for _, r in sorted(v)] for s, v in by.items()}
    lists = list(seqs.values())
    res = {"trades": sum(len(v) for v in lists), "streaks": dsc.streak_table(lists),
           "runs_z": {s: dsc.runs_z(v) for s, v in seqs.items()},
           "lag1": {s: round(float(np.corrcoef(v[:-1], v[1:])[0, 1]), 3) for s, v in seqs.items() if len(v) > 30},
           "edge_window": dsc.edge_window(lists), "sizing": {}}
    for rule in dsc.SIZING:
        parts = [dsc.sizing_run(v, rule) for v in lists]
        tot, risk = sum(p["sum"] for p in parts), sum(p["risk"] for p in parts)
        res["sizing"][rule] = {"sum": round(tot, 1), "per_risk": round(tot / risk, 4), "worst_dd": max(p["dd"] for p in parts),
                               "max_size": max(p["max_size"] for p in parts)}
    return res


def part_bars(series, symbols: set[str]) -> dict:
    trades: dict[str, dict[str, list]] = {"trend": {v: [] for v in dsc.VARIANTS}, "random": {v: [] for v in dsc.VARIANTS}}
    seqs_fixed: list[list[float]] = []
    used, first, last = [], None, None
    for s in series:
        if s.interval != "1h" or (symbols and s.symbol not in symbols) or len(s.t) < 24 * 300:
            continue
        sp = SPREAD.get(s.symbol)
        if sp is None:
            continue
        t0 = time.time()
        b = dsc.Bars(s.t, s.o, s.h, s.l, s.c, s.symbol, sp)
        ent = dsc.trend_entries(b)
        rnd = dsc.random_entries(b, ent)
        for name, el in (("trend", ent), ("random", rnd)):
            for v in dsc.VARIANTS:
                for (i, dr, D) in el:
                    r = dsc.simulate(b, i, dr, D, v)
                    trades[name][v].append((float(b.t[i]), dr, r["r"], s.symbol, r["units"]))
        seqs_fixed.append([x[2] for x in trades["trend"]["fixed"] if x[3] == s.symbol])
        used.append(s.symbol)
        first = min(first or s.t[0], s.t[0]); last = max(last or s.t[-1], s.t[-1])
        print(f"  {s.symbol}: {len(s.t)} hours, entries {len(ent)}, {time.time() - t0:.0f}s", flush=True)
    out = {"symbols": used, "first": _day(first), "last": _day(last), "sets": {}}
    for name, per in trades.items():
        out["sets"][name] = {}
        for v, rows in per.items():
            rows.sort()
            r = [x[2] for x in rows]
            out["sets"][name][v] = {
                "all": dsc.stats(r),
                "long": dsc.stats([x[2] for x in rows if x[1] > 0]),
                "short": dsc.stats([x[2] for x in rows if x[1] < 0]),
                "before_2023": dsc.stats([x[2] for x in rows if x[0] < SPLIT]),
                "from_2023": dsc.stats([x[2] for x in rows if x[0] >= SPLIT]),
                "units_avg": round(float(np.mean([x[4] for x in rows])), 2) if rows else 0,
                "by_symbol": {sym: dsc.stats([x[2] for x in rows if x[3] == sym])["mean"] for sym in used},
            }
    out["sizing_on_trend"] = {}
    for rule in dsc.SIZING:
        parts = [dsc.sizing_run(v, rule) for v in seqs_fixed if v]
        tot, risk = sum(p["sum"] for p in parts), sum(p["risk"] for p in parts)
        out["sizing_on_trend"][rule] = {"sum": round(tot, 1), "per_risk": round(tot / risk, 4) if risk else 0,
                                        "worst_dd": max((p["dd"] for p in parts), default=0)}
    out["streaks_on_trend"] = dsc.streak_table(seqs_fixed, 4)
    return out


def _day(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d") if t else ""


def _s(x: dict) -> str:
    return f"{x['mean']:+.3f}R (t {x['t']:+.1f}, n {x['n']}, просадка {x['dd']:.0f}R)"


def text_report(res: dict) -> str:
    L = ["Дисциплина риска и сопровождение сделки: проверка на наших данных", ""]
    p1 = res.get("trades_part")
    if p1:
        st = p1["streaks"]
        L.append(f"1. Серии на сделках сканера (повтор 4,5 года, {p1['trades']} сильных сделок, база {st['all']['mean']:+.3f}R, прибыльных {st['all']['win']:.1%}):")
        L.append("   после k убытков подряд: " + ", ".join(f"{k}: {v['mean']:+.3f} ({v['win']:.0%}, n {v['n']})" for k, v in st["after_losses"].items()))
        L.append("   после k прибылей подряд: " + ", ".join(f"{k}: {v['mean']:+.3f} ({v['win']:.0%})" for k, v in st["after_wins"].items()))
        zs = list(p1["runs_z"].values())
        L.append(f"   тест серий по инструментам: z от {min(zs):+.2f} до {max(zs):+.2f} (|z| < 2 = серий нет)")
        ew = p1["edge_window"]
        L.append(f"   сумма последних 20 сделок -> следующая: корреляция {ew.get('corr', 0):+.3f}, по квинтилям {ew.get('quintiles')}")
        L.append("   правила размера позиции (R на единицу риска; худшая просадка на инструменте):")
        for k, v in p1["sizing"].items():
            L.append(f"     {dsc.SIZING[k]}: {v['per_risk']:+.4f} R/риск, сумма {v['sum']:+.0f}R, просадка {v['worst_dd']:.0f}R, макс. размер ×{v['max_size']:g}")
        L.append("")
    p2 = res.get("bars_part")
    if p2:
        L.append(f"2. Сопровождение сделки на часовых свечах {p2['first']} … {p2['last']}, {len(p2['symbols'])} инструментов, спред учтён.")
        L.append("   Вход одинаковый для всех вариантов: дневной тренд + новый 5-дневный экстремум, риск 1,5 дневного ATR.")
        for name, title in (("trend", "Трендовый вход"), ("random", "Контроль: случайный вход")):
            L.append(f"   {title}:")
            for v, x in p2["sets"][name].items():
                L.append(f"     {dsc.VARIANT_RU[v]}: {_s(x['all'])}; лонги {x['long']['mean']:+.3f}, шорты {x['short']['mean']:+.3f}; "
                         f"до 2023 {x['before_2023']['mean']:+.3f}, с 2023 {x['from_2023']['mean']:+.3f}; единиц в среднем {x['units_avg']}")
        L.append("   Размер позиции на трендовых сделках (вариант «стоп и тейк»):")
        for k, v in p2["sizing_on_trend"].items():
            L.append(f"     {dsc.SIZING[k]}: {v['per_risk']:+.4f} R/риск, сумма {v['sum']:+.0f}R, просадка {v['worst_dd']:.0f}R")
        st = p2["streaks_on_trend"]
        L.append("   После k убытков подряд (трендовые сделки): " + ", ".join(f"{k}: {v['mean']:+.3f} (n {v['n']})" for k, v in st["after_losses"].items())
                 + f"; все {st['all']['mean']:+.3f}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="")
    ap.add_argument("--trades", default=str(ROOT / "data" / "research" / "scanner_replay" / "trades"))
    ap.add_argument("--symbols", default="")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "research"))
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    res: dict = {"made": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tr = Path(a.trades)
    if tr.exists() and any(tr.glob("*.jsonl")):
        print("part 1: scanner replay trades ...", flush=True)
        res["trades_part"] = part_trades(tr)
    if a.snapshot:
        snap = Path(a.snapshot)
    else:
        snaps = sorted(p for p in (ROOT / "data" / "market").glob("*") if (p / "public_candles.csv").exists())
        snap = snaps[-1] if snaps else None
    if snap:
        print(f"part 2: hourly bars from {snap} ...", flush=True)
        series = market.load_series(snap, use_public=True, intervals=["1h"])
        only = {x.strip() for x in a.symbols.split(",") if x.strip()}
        res["bars_part"] = part_bars([s for s in series if s.public], only)
    if len(res) == 1:
        sys.exit("nothing to test: no replay trades and no snapshot with hourly public bars")
    txt = text_report(res)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "discipline_last.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "discipline_last.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    if not a.no_push:
        slim = {"discipline": {k: v for k, v in res.items() if k != "bars_part"}}
        push("nics-trading", "eval", slim, txt, cycle="discipline " + res["made"][:10])


if __name__ == "__main__":
    main()
