"""Daily "buy a short dip inside an uptrend" rule on long public history (research only).

Nothing here sends a signal or changes the scanner or any EA.

Why it exists (2026-11-26): the multi-year replay of the live scanner showed no directional
edge on intraday scale (a trade loses about one spread in both directions). A scan of other
rule families on the same minute history left one family that stays positive after costs on
data it was not chosen on: on stock indices and stocks, after a short fall inside a long
uptrend the next days are on average up. Shorts by the mirrored rule do not work.

Rule (long only, daily closes, decision at the close, entry at that close):
- uptrend: close above the mean of the previous `trend` closes (200); 0 switches it off;
- dip, one of: `down` lower closes in a row (3); 2-bar strength index below `rsi2` (10);
  close at the lowest close of `low` days (5);
- exit: first close above the mean of the previous `exit_mean` closes (5), or after `max_days`;
- no stop inside the rule (every stop tried made the result worse); risk is limited by size
  and by the number of positions held at once.

Costs: `cost` per trade (spread, both sides) and `financing` per night held.

Honesty notes:
- a list of today's large stocks is survivor-biased: they are the ones that kept rising, so
  the stock numbers are an upper bound; the index numbers do not have this problem;
- positions in several indices open on the same days and fall together, so the trade count
  overstates the independent evidence: judge by the yearly table and the drawdown;
- the worst days are market crashes: a position sized for a normal day loses several times
  the usual amount on such a day.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

GROUPS = {
    "indices": ["US500", "US100", "US30", "GER40", "JP225", "HK50"],
    "stocks": ["NVIDIA", "TSLA", "META", "AMAZON", "NFLX", "AMD", "AVGO", "MU", "TSM", "ASML"],
    "metals_oil": ["XAU/USD", "XAG/USD", "UKOUSDft"],
    "crypto": ["BTC/USD", "ETH/USD"],
    "forex": ["EUR/USD", "GBP/USD", "USD/JPY"],
}


def load_daily(snapshot: Path) -> dict[str, np.ndarray]:
    """symbol -> array [t, open, high, low, close] of public daily bars, oldest first."""
    rows = defaultdict(list)
    with open(Path(snapshot) / "public_candles.csv", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        head = next(r)
        ix = {k: head.index(k) for k in ("symbol", "interval", "candle_time", "open", "high", "low", "close")}
        for row in r:
            if row[ix["interval"]] != "1day":
                continue
            try:
                t = datetime.fromisoformat(row[ix["candle_time"]].replace("Z", "+00:00"))
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                v = [t.timestamp()] + [float(row[ix[k]]) for k in ("open", "high", "low", "close")]
            except ValueError:
                continue
            if v[4] > 0 and v[2] >= v[3]:
                rows[row[ix["symbol"]]].append(v)
    return {s: np.array(sorted(v)) for s, v in rows.items() if len(v) > 260}


def strength2(c: np.ndarray, i: int) -> float:
    d = np.diff(c[i - 2:i + 1])
    up, dn = d[d > 0].sum(), -d[d < 0].sum()
    return 100.0 * up / (up + dn) if up + dn > 0 else 50.0


def is_dip(c: np.ndarray, i: int, down: int = 3, rsi2: float = 0.0, low: int = 0) -> bool:
    """Uses closes up to and including bar i only."""
    if down and all(c[i - k] < c[i - k - 1] for k in range(down)):
        return True
    if rsi2 and strength2(c, i) < rsi2:
        return True
    return bool(low and c[i] <= c[i - low:i].min())


def trades(a: np.ndarray, trend: int = 200, exit_mean: int = 5, max_days: int = 10, cost: float = 0.0002,
           financing: float = 0.0002, **dip) -> list[dict]:
    """One position at a time per instrument. Returns entry/exit bar, net return, range share."""
    c, h, lo = a[:, 4], a[:, 2], a[:, 3]
    out, i = [], max(trend, 25)
    while i < len(c) - 1:
        if (not trend or c[i] > c[i - trend:i].mean()) and is_dip(c, i, **dip):
            j = i + 1
            while j < len(c) - 1 and not (c[j] > c[j - exit_mean:j].mean() or j - i >= max_days):
                j += 1
            rng = float(np.mean(np.maximum(h[i - 19:i + 1] - lo[i - 19:i + 1], np.abs(c[i - 19:i + 1] - c[i - 20:i]))) / c[i])
            rng = max(rng, 0.002)  # old public bars sometimes have high = low
            out.append({"i": i, "j": j, "t": float(a[i, 0]), "ret": float(c[j] / c[i] - 1 - cost - financing * (j - i)),
                        "days": j - i, "range": rng})
            i = j
        else:
            i += 1
    return out


def stat(r) -> dict:
    r = np.asarray(r, float)
    if len(r) < 8:
        return {"n": int(len(r))}
    return {"n": int(len(r)), "mean_pct": round(float(r.mean() * 100), 3),
            "t": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 1), "win": round(float((r > 0).mean()), 3)}


def year_of(t: float) -> int:
    return datetime.fromtimestamp(t, timezone.utc).year


def portfolio(series: dict[str, np.ndarray], risk_per_range: float = 0.005, max_positions: int = 3,
              year_from: int = 1990, cost: float = 0.0002, financing: float = 0.0002, **rule) -> dict:
    """Daily mark-to-market. Size: a move of one average daily range = `risk_per_range` of the
    account. A new dip is skipped while `max_positions` are open. Returns are sums, not compounded."""
    entries = []
    for name, a in series.items():
        d = (a[:, 0] // 86400).astype(int)
        for tr in trades(a, cost=cost, financing=financing, **rule):
            entries.append((int(d[tr["i"]]), name, tr, d))
    entries.sort(key=lambda e: (e[0], e[1]))
    pnl, busy, n, skipped = defaultdict(float), [], 0, 0
    for d0, name, tr, d in entries:
        if year_of(d0 * 86400) < year_from:
            continue
        busy = [b for b in busy if b > d0]
        if len(busy) >= max_positions:
            skipped += 1
            continue
        busy.append(int(d[tr["j"]]))
        n += 1
        w, c = risk_per_range / tr["range"], series[name][:, 4]
        for k in range(tr["i"] + 1, tr["j"] + 1):
            pnl[int(d[k])] += w * (c[k] / c[k - 1] - 1 - financing)
        pnl[int(d[tr["j"]])] -= w * cost
    if len(pnl) < 30:
        return {"trades": n}
    days = sorted(pnl)
    r = np.array([pnl[x] for x in days])
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    span = max((days[-1] - days[0]) / 365.25, 0.5)
    years = defaultdict(float)
    for x, v in zip(days, r):
        years[year_of(x * 86400)] += v
    return {"trades": n, "skipped": skipped, "per_year_pct": round(float(eq[-1] / span * 100), 1),
            "max_drawdown_pct": round(dd * 100, 1), "worst_day_pct": round(float(r.min() * 100), 1),
            "sharpe": round(float(r.sum() / span / (r.std() * np.sqrt(len(r) / span) + 1e-12)), 2),
            "years": {str(y): round(v * 100, 1) for y, v in sorted(years.items())},
            "losing_years": sum(1 for v in years.values() if v < 0), "n_years": len(years)}


PERIODS = ((1990, 1999), (2000, 2009), (2010, 2019), (2020, 2022), (2023, 2100))
VARIANTS = {
    "three_down_closes": {"down": 3},
    "two_down_closes": {"down": 2},
    "four_down_closes": {"down": 4},
    "strength2_below_10": {"down": 0, "rsi2": 10},
    "five_day_low": {"down": 0, "low": 5},
    "three_down_no_trend_rule": None,  # filled in run(): same dip without the uptrend condition
}


def run(series: dict[str, np.ndarray]) -> dict:
    res = {"groups": {}}
    for g, names in GROUPS.items():
        have = {n: series[n] for n in names if n in series}
        if not have:
            continue
        block = {"instruments": sorted(have), "variants": {}}
        for vname, dip in VARIANTS.items():
            kw = {"down": 3, "trend": 0} if dip is None else dict(dip)
            allt, per = [], {}
            for n, a in have.items():
                tr = trades(a, **kw)
                allt += tr
                per[n] = stat([x["ret"] for x in tr])
            block["variants"][vname] = {
                "all": stat([x["ret"] for x in allt]),
                "hold_days": round(float(np.mean([x["days"] for x in allt])), 1) if allt else None,
                "periods": {f"{a}-{min(b, 2026)}": stat([x["ret"] for x in allt if a <= year_of(x["t"]) <= b]) for a, b in PERIODS},
                "instruments": per,
            }
        block["portfolio_three_down"] = {f"max_{k}_positions": portfolio(have, max_positions=k, down=3) for k in (2, 3, 6)}
        res["groups"][g] = block
    base = res["groups"].get("indices", {}).get("variants", {}).get("three_down_closes")
    if base:
        per = [p for p in base["periods"].values() if p.get("n", 0) >= 30]
        res["verdict_indices"] = ("держится" if base["all"].get("t", 0) >= 3 and per and all(p["mean_pct"] > 0 for p in per)
                                  else "не подтверждено")
    return res
