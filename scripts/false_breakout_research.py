"""False breakouts on public hourly history: can they be told apart at entry? (research only)

See src/nics/market/false_breakout.py for the definitions and the honesty notes. Nothing is sent
to users, the scanner, the level method or any EA.

  python scripts/false_breakout_research.py                    (latest snapshot in data/market)
  python scripts/false_breakout_research.py --symbols XAU/USD,US500 --no-yahoo

Long set: Dukascopy hourly bars in <snapshot>/public_candles.csv (fill it with
scripts/fetch_dukascopy.py). Fitted on years before --split-year, judged only on the years after.
Two-year set: public hourly bars for all instruments (same download as level_direction_research).
Writes experiments/research/false_breakout_last.json and .txt and sends the summary to the private
dashboard (--no-push to skip).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nics.market import data as market  # noqa: E402
from nics.market import false_breakout as fb  # noqa: E402
from nics.telemetry.push import push  # noqa: E402

FEATURE_RU = {"touches": "касаний уровня", "age_h": "возраст уровня, ч", "round": "круглый уровень (1 = да)",
              "hour": "час UTC", "d1_with": "тренд D1 (+1 по пробою, −1 против)", "h4_with": "тренд H4 (+1 по пробою)",
              "ext_d1": "растянутость от EMA20 D1 по пробою, ATR D1", "depth": "глубина закрытия за уровнем, ATR H1",
              "bar_range": "размер свечи пробоя, ATR H1", "close_pos": "закрытие у края свечи (1 = у самого края)",
              "approach": "скорость подхода за 6 ч, ATR H1", "vol_ratio": "тиковый объём к среднему за сутки"}


def group_of(sym: str) -> str:
    g = {"XAU/USD": "metals", "XAG/USD": "metals", "COPPER": "metals", "UKOUSDft": "oil", "BTC/USD": "crypto", "ETH/USD": "crypto",
         "US500": "indices", "US100": "indices", "US30": "indices", "GER40": "indices", "JP225": "indices", "HK50": "indices"}
    return g.get(sym) or ("forex" if "/" in sym else "stocks")


def summarise(ev: list[dict], split_t: float | None) -> dict:
    s = {"all": fb.describe(ev), "constructions": fb.constructions(ev),
         "side": {k: {**fb.describe([e for e in ev if e["dir"] == d]), "constructions": fb.constructions([e for e in ev if e["dir"] == d])}
                  for k, d in (("down", -1), ("up", 1))},
         "group": {g: fb.describe([e for e in ev if e["group"] == g]) for g in sorted({e["group"] for e in ev})}}
    if split_t is not None:
        ins, oos = [e for e in ev if e["t"] < split_t], [e for e in ev if e["t"] >= split_t]
        s["features_in"], s["features_out"] = fb.by_feature(ins), fb.by_feature(oos)
        s["constructions_in"], s["constructions_out"] = fb.constructions(ins), fb.constructions(oos)
        s["all_in"], s["all_out"] = fb.describe(ins), fb.describe(oos)
        s["model"] = fb.model_check(ev, split_t)
    else:
        s["features"] = fb.by_feature(ev)
    return s


def stable_features(s: dict, min_n: int = 150, min_gap: int = 5) -> list[str]:
    """Buckets whose false share differs from the average by >= min_gap points in BOTH periods, same sign."""
    out = []
    fi, fo = s.get("features_in", {}), s.get("features_out", {})
    base_i, base_o = s["all_in"].get("false_pct", 0), s["all_out"].get("false_pct", 0)
    for name, buckets in fi.items():
        for lab, a in buckets.items():
            b = fo.get(name, {}).get(lab)
            if not b or a["n"] < min_n or b["n"] < min_n:
                continue
            da, db = a["false_pct"] - base_i, b["false_pct"] - base_o
            if abs(da) >= min_gap and abs(db) >= min_gap and da * db > 0:
                out.append(f"{FEATURE_RU[name]} {lab}: ложных {a['false_pct']} % / {b['false_pct']} % "
                           f"(в среднем {base_i:.0f} / {base_o:.0f}), R на сделку {a.get('avg', 0):+} / {b.get('avg', 0):+} "
                           f"(n {a['n']} / {b['n']})")
    return out


def russian(name: str, s: dict, meta: dict) -> str:
    a = s["all"]
    L = [f"{name}: {meta['symbols']} инструментов, {meta['from']}–{meta['to']}, пробоев {a['n']}. Проверка на прошлом, не прогноз; "
         "часовые свечи; если стоп и тейк оба внутри одной свечи, порядок неизвестен и сделка считается за 0R."]
    if meta.get("ambiguous_pct") is not None:
        L.append(f"Таких неоднозначных сделок {meta['ambiguous_pct']} % от всех входов.")
    if meta.get("skipped"):
        L.append("Без длинной истории (пропущены): " + ", ".join(meta["skipped"]))
    L.append("Важно: долю ложных частично можно угадать даже на случайных данных (мелкий пробой чаще возвращается), "
             "поэтому признак полезен, только если меняет R сделок, а не только долю ложных.")
    L.append(f"Ложных (вернулись за уровень за {fb.PARAMS['fb_h']} ч раньше хода на 1 ATR): {a['false_pct']} %; "
             f"вход сразу 1:1 даёт {a.get('R', 0):+}R ({a.get('avg', 0):+} на сделку, t {a.get('t', 0)}).")
    for k, ru in (("down", "пробой вниз (шорт)"), ("up", "пробой вверх (лонг)")):
        x = s["side"][k]
        if x["n"]:
            c = x["constructions"]
            L.append(f"  {ru}: {x['n']}, ложных {x['false_pct']} %; сразу {c['base'].get('R', 0):+}R "
                     f"({c['base']['avg_filled']:+} на сделку), ждать 2 закрытия {c['confirm2'].get('R', 0):+}R "
                     f"(вход в {c['confirm2']['filled']}, {c['confirm2']['avg_filled']:+} на сделку), "
                     f"ретест {c['retest'].get('R', 0):+}R (вход в {c['retest']['filled']}, {c['retest']['avg_filled']:+} на сделку)")
    if "constructions_in" in s:
        for per, key in (("до раздела", "constructions_in"), ("после раздела", "constructions_out")):
            c = s[key]
            L.append(f"  {per}, R на сделку: сразу {c['base']['avg_filled']:+}, 2 закрытия {c['confirm2']['avg_filled']:+}, "
                     f"ретест {c['retest']['avg_filled']:+}")
        m = s["model"]
        if "out" in m:
            t = m["out"]["tercile_by_false_probability"]
            L.append(f"Модель «вероятность ложного» по 12 признакам, подобрана до {meta['split']}, проверена после: "
                     f"AUC {m['in']['auc']} / {m['out']['auc']} (0.5 = угадывание). После раздела: "
                     f"низкая треть ложных {t['low']['false_pct']} % ({t['low'].get('avg', 0):+} R), "
                     f"высокая {t['high']['false_pct']} % ({t['high'].get('avg', 0):+} R).")
        else:
            L.append(f"Модель: мало пробоев для проверки до и после раздела ({m.get('n_in', 0)} / {m.get('n_out', 0)}).")
        st = stable_features(s)
        L.append("Признаки, которые сдвигают долю ложных одинаково до и после раздела:" if st else
                 "Ни один признак не сдвигает долю ложных одинаково до и после раздела (порог 5 п.п., n ≥ 150).")
        L += ["  " + x for x in st[:12]]
    return "\n".join(L)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--from-year", type=int, default=2008)
    ap.add_argument("--split-year", type=int, default=2018)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-yahoo", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="first add missing Dukascopy hourly months to the snapshot (cached, polite)")
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache" / "public_h1"))
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
    if args.fetch:  # long history for every instrument; months already cached are not downloaded again
        cmd = [sys.executable, str(ROOT / "scripts" / "fetch_dukascopy.py"), "--snapshot", str(snap), "--from-year", str(args.from_year)]
        if only:
            cmd += ["--symbols", ",".join(sorted(only))]
        print("Dukascopy hourly history (first time about an hour, later minutes) ...", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"Dukascopy download stopped (code {r.returncode}); using what is cached", flush=True)
    start = datetime(args.from_year, 1, 1, tzinfo=timezone.utc).timestamp()
    split_t = datetime(args.split_year, 1, 1, tzinfo=timezone.utc).timestamp()
    print(f"snapshot {snap.name}: loading hourly bars ...", flush=True)
    series = market.load_series(snap, use_public=True, intervals=["1h"])
    spreads = {s.symbol: float(np.median(s.spread[s.spread > 0])) for s in series if not s.public and (s.spread > 0).any()}
    sets: dict[str, dict] = {"long": {"ev": [], "used": [], "t": [None, None]}, "two_years": {"ev": [], "used": [], "t": [None, None]}}

    def run_one(sym, t, h, lo, c, v, key, begin):
        t0 = time.time()
        a0 = fb.AMBIGUOUS[0]
        ev = [e for e in fb.events(t, h, lo, c, v, spreads.get(sym, 0.0)) if e["t"] >= begin]
        for e in ev:
            e["symbol"], e["group"] = sym, group_of(sym)
        S = sets[key]
        S["ev"].extend(ev); S["used"].append(sym)
        S["amb"] = S.get("amb", 0) + fb.AMBIGUOUS[0] - a0
        if ev:
            S["t"] = [min(x for x in (S["t"][0], ev[0]["t"]) if x is not None), max(x for x in (S["t"][1], ev[-1]["t"]) if x is not None)]
        print(f"  {sym} [{key}]: {len(t)} hours, breakouts {len(ev)}, {time.time() - t0:.0f}s", flush=True)

    skipped: list[str] = []
    fb.AMBIGUOUS[0] = 0
    for s in series:
        if not s.public or (only and s.symbol not in only):
            continue
        keep = ~((s.h == s.l) & (s.v == 0)) & (s.t >= start - 400 * 86400)
        if keep.sum() < 24 * 400:
            print(f"  {s.symbol}: only {int(keep.sum())} hourly bars, skipped", flush=True)
            skipped.append(f"{s.symbol} ({int(keep.sum())} ч)")
            continue
        run_one(s.symbol, s.t[keep], s.h[keep], s.l[keep], s.c[keep], s.v[keep], "long", start)
    if not args.no_yahoo:
        from level_direction_research import YAHOO, yahoo_h1
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
                continue
            run_one(sym, a[:, 0], a[:, 1], a[:, 2], a[:, 3], None, "two_years", 0.0)
    if not any(S["ev"] for S in sets.values()):
        sys.exit("no breakouts found: no public hourly history in the snapshot and no two-year download")
    fmt = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m")  # noqa: E731
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    full, texts = {}, []
    for key, S in sets.items():
        if not S["ev"]:
            continue
        meta = {"snapshot": snap.name, "symbols": len(S["used"]), "symbol_list": S["used"], "from": fmt(S["t"][0]),
                "to": fmt(S["t"][1]),
                "ambiguous_pct": round(100 * S.get("amb", 0) / max(1, sum(e[k] is not None for e in S["ev"] for k in ("base", "confirm2", "retest")))),
                "skipped": skipped if key == "long" else [], "split": str(args.split_year) if key == "long" else None, "params": fb.PARAMS}
        sp = split_t if key == "long" and S["t"][0] < split_t < S["t"][1] else None
        if sp is None:
            meta["split"] = None
        summ = summarise(S["ev"], sp)
        full[key] = {"meta": meta, **summ}
        texts.append(russian("Длинная история" if key == "long" else "Два года, все инструменты", summ, meta))
    text = "\n\n".join(texts)
    (out / "false_breakout_last.json").write_text(json.dumps(full, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "false_breakout_last.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.no_push:
        slim = {k: {"meta": {m: v["meta"][m] for m in ("symbols", "from", "to", "split")}, "all": v["all"],
                    "constructions": v["constructions"], "model_auc": {p: v.get("model", {}).get(p, {}).get("auc") for p in ("in", "out")}}
                for k, v in full.items()}
        push("nics-trading", "eval", slim, text, cycle=f"false_breakout {snap.name}")


if __name__ == "__main__":
    main()
