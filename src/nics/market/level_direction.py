"""Level method on long hourly history, with several ways to choose the DIRECTION of the plan.

Research only. Nothing here sends a signal or changes the scanner, the level method or any EA.

Why: all our broker intraday history (since September 2026) is one falling market, so the live
level method was tuned and its instruments chosen on shorts only. This module replays the same
level logic (daily and 4-hour swing levels, bounce / breakout / range plans, stop beyond the
level, take at the next level with RR >= 2, breakeven at +1R, 24 h hold) on public hourly bars
since 2008, where markets both rose and fell, and compares direction rules:

  base      the live rule: D1 close vs EMA20 with a 3-day slope AND H4 EMA20 vs EMA50; plans
            only with that trend, both sides when the two disagree
  pullback  base, but a short-term "down" inside a long-term uptrend (D1 EMA50 > EMA200) is
            treated as a pullback: long plans from support, no shorts
  no_low    base, but no breakdown shorts (short only from resistance after a rebound) and no
            breakout longs at the high
  rule      pullback + no_low together (the rule proposed on 2026-11-23)
  lt_only   direction from the long-term trend only (EMA50 vs EMA200)

Honesty notes:
  - a decision at hour T uses only bars closed before T; a swing level counts only after the
    two following bars closed;
  - hourly bars cannot show what happened first inside one hour. If the stop is inside the
    range of the fill bar the trade counts as a LOSS. That is pessimistic for every variant in
    the same way, so compare variants with each other, do not read R as a forecast;
  - `fwd` is a stop-free check of the direction alone: the move from the close of the fill bar
    over the next 24 hourly bars, in daily ATR, signed by the trade direction (measured from
    the close, not from the order price, so a random walk gives zero).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DAY = 86400
DAY_SHIFT = 7200  # trading day boundary close to the broker's server midnight
VARIANTS = ("base", "pullback", "no_low", "rule", "lt_only")
PARAMS = dict(frac=2, h4_days=6, cluster=0.2, front=0.02, brk_off=0.03, buf=0.12, min_risk=0.15,
              tp_front=0.03, rr_min=2.0, max_dist=0.6, valid_h=10, hold_h=24, be=1.0, cost_spreads=1.2)


def ema(x: np.ndarray, k: int) -> np.ndarray:
    a = 2.0 / (k + 1)
    out = np.empty(len(x))
    e = x[0]
    for i, v in enumerate(x):
        e = v if i == 0 else v * a + e * (1 - a)
        out[i] = e
    return out


def aggregate(t: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, keys: np.ndarray):
    """Bars grouped by an integer key (ascending): key, first time, high, low, close."""
    starts = np.r_[0, np.nonzero(np.diff(keys))[0] + 1]
    return (keys[starts], t[starts], np.maximum.reduceat(h, starts), np.minimum.reduceat(lo, starts),
            c[np.r_[starts[1:] - 1, len(c) - 1]])


def fractals(h: np.ndarray, lo: np.ndarray, t: np.ndarray, k: int, first: int):
    """Swing highs / lows. Returns price, bar time, and how many CLOSED bars make it known."""
    price, when, need = [], [], []
    for i in range(first, len(h) - k):
        a = max(0, i - k)
        if a == i:
            continue
        hi = h[i] > h[a:i].max() and h[i] >= h[i + 1:i + k + 1].max()
        low = lo[i] < lo[a:i].min() and lo[i] <= lo[i + 1:i + k + 1].min()
        if hi:
            price.append(h[i]); when.append(t[i]); need.append(i + k + 1)
        if low:
            price.append(lo[i]); when.append(t[i]); need.append(i + k + 1)
    return np.array(price), np.array(when), np.array(need)


def cluster_levels(price: np.ndarray, when: np.ndarray, tol: float) -> list[float]:
    """Touches closer than `tol` form one level; its price is the most recent touch."""
    if not len(price):
        return []
    order = np.argsort(price)
    p, w = price[order], when[order]
    groups, cur = [], [0]
    for i in range(1, len(p)):
        if p[i] - p[cur[0]] <= tol:
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    merged: list[list[int]] = []
    rep = lambda g: p[max(g, key=lambda j: w[j])]  # noqa: E731
    for g in groups:
        if merged and rep(g) - rep(merged[-1]) < tol / 2:
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    return [float(rep(g)) for g in merged]


@dataclass
class Prepared:
    """Everything that does not depend on the direction rule, for one symbol."""
    t: np.ndarray; h: np.ndarray; lo: np.ndarray; c: np.ndarray
    nd: np.ndarray; n4: np.ndarray            # closed days / closed 4h bars known at each hour
    atr_d: np.ndarray; atr_h: np.ndarray      # per day / per hour
    td: np.ndarray; lt: np.ndarray; e20d: np.ndarray  # per day
    t4: np.ndarray                             # per 4h bar
    fd: tuple; f4: tuple                       # swing touches (price, time, bars needed)
    h4_t: np.ndarray
    lo72: np.ndarray; hi72: np.ndarray


def prepare(t: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> Prepared | None:
    dk = ((t + DAY_SHIFT) // DAY).astype(np.int64)
    k4 = (t // 14400).astype(np.int64)
    dkey, dt, dh, dl, dc = aggregate(t, h, lo, c, dk)
    hkey, ht, hh, hl, hc = aggregate(t, h, lo, c, k4)
    if len(dkey) < 260 or len(hkey) < 100:
        return None
    pc = np.r_[dc[0], dc[:-1]]
    tr_d = np.maximum(dh - dl, np.maximum(np.abs(dh - pc), np.abs(dl - pc)))
    atr_d = np.convolve(tr_d, np.ones(14) / 14, mode="full")[:len(tr_d)]
    pch = np.r_[c[0], c[:-1]]
    tr_h = np.maximum(h - lo, np.maximum(np.abs(h - pch), np.abs(lo - pch)))
    atr_h = np.convolve(tr_h, np.ones(14) / 14, mode="full")[:len(tr_h)]
    e20, e50, e200 = ema(dc, 20), ema(dc, 50), ema(dc, 200)
    slope = np.r_[np.zeros(3), e20[3:] - e20[:-3]]
    td = np.where((dc > e20) & (slope > 0), 1, np.where((dc < e20) & (slope < 0), -1, 0))
    lt = np.where(e50 > e200, 1, -1)
    e20h, e50h = ema(hc, 20), ema(hc, 50)
    t4 = np.where(e20h > e50h, 1, -1)
    lo72 = np.array([lo[max(0, i - 72):i].min() if i else lo[0] for i in range(len(lo))])
    hi72 = np.array([h[max(0, i - 72):i].max() if i else h[0] for i in range(len(h))])
    return Prepared(t, h, lo, c, np.searchsorted(dkey, dk, side="left"), np.searchsorted(hkey, k4, side="left"),
                    atr_d, atr_h, td, lt, e20, t4, fractals(dh, dl, dt, PARAMS["frac"], 1),
                    fractals(hh, hl, ht, PARAMS["frac"], 2), ht, lo72, hi72)


def state_at(p: Prepared, i: int) -> dict | None:
    """Market state known at the open of hourly bar i (bars before i only)."""
    nd, n4 = int(p.nd[i]), int(p.n4[i])
    if nd < 210 or n4 < 60 or i < 80:
        return None
    d, q = nd - 1, n4 - 1
    atr_d, atr_h = float(p.atr_d[d]), float(p.atr_h[i - 1])
    if not (atr_d > 0 and atr_h > 0):
        return None
    px = float(p.c[i - 1])
    td, t4 = int(p.td[d]), int(p.t4[q])
    base = td if td != 0 and td == t4 else 0
    return dict(px=px, atr_d=atr_d, atr_h=atr_h, base=base, lt=int(p.lt[d]), ext=(px - float(p.e20d[d])) / atr_d,
                reb=(px - float(p.lo72[i])) / atr_d if base < 0 else (float(p.hi72[i]) - px) / atr_d if base > 0 else 0.0,
                nd=nd, n4=n4)


def levels_at(p: Prepared, s: dict) -> tuple[list[float], list[float]]:
    """Two nearest resistances above and two nearest supports below the price."""
    px, atr_d = s["px"], s["atr_d"]
    win = 6 * atr_d
    fp, ft, fn = p.fd
    k = int(np.searchsorted(fn, s["nd"], side="right"))
    m = np.abs(fp[:k] - px) <= win
    price, when = fp[:k][m], ft[:k][m]
    hp, htime, hn = p.f4
    k4 = int(np.searchsorted(hn, s["n4"], side="right"))
    last_t = p.h4_t[s["n4"] - 1]
    a = int(np.searchsorted(htime, last_t - PARAMS["h4_days"] * DAY, side="left"))
    if k4 > a:
        m4 = np.abs(hp[a:k4] - px) <= win
        price, when = np.r_[price, hp[a:k4][m4]], np.r_[when, htime[a:k4][m4]]
    lv = cluster_levels(price, when, PARAMS["cluster"] * atr_d)
    gap = 0.1 * s["atr_h"]
    above = sorted(x for x in lv if x > px + gap)[:2]
    below = sorted((x for x in lv if x < px - gap), reverse=True)[:2]
    return above, below


def direction(variant: str, s: dict) -> tuple[int, bool, bool]:
    """(trend used for the plan, breakout plans allowed, bounce plans allowed)."""
    base, lt = s["base"], s["lt"]
    if variant == "base":
        return base, True, True
    if variant == "lt_only":
        return lt, True, True
    trend, brk = base, True
    if variant in ("pullback", "rule") and base < 0 and lt > 0:
        trend = 1
    if variant in ("no_low", "rule") and trend != 0:
        brk = False
    return trend, brk, True


def plans(variant: str, s: dict, above: list[float], below: list[float], spread: float) -> list[dict]:
    P, a = PARAMS, s["atr_d"]
    px = s["px"]
    min_risk = max(P["min_risk"] * a, 5 * spread)
    out: list[dict] = []

    def mk(d: int, kind: str, entry: float, stop: float, targets: list[float]) -> None:
        risk = abs(entry - stop)
        if risk < min_risk:
            stop, risk = entry - d * min_risk, min_risk
        if abs(entry - px) > P["max_dist"] * a:
            return
        for tg in targets:
            tp = tg - d * P["tp_front"] * a
            if d * (tp - entry) / risk >= P["rr_min"]:
                out.append(dict(dir=d, kind=kind, entry=entry, stop=stop, tp=tp, risk=risk))
                return

    trend, brk, _ = direction(variant, s)
    buf = P["buf"] * a
    r0, s0 = (above[0] if above else None), (below[0] if below else None)
    if trend > 0:
        if s0 is not None:
            mk(1, "bounce", s0 + P["front"] * a, s0 - buf, above)
        if r0 is not None and brk:
            mk(1, "breakout", r0 + P["brk_off"] * a, r0 - buf, above[1:])
    elif trend < 0:
        if r0 is not None:
            mk(-1, "bounce", r0 - P["front"] * a, r0 + buf, below)
        if s0 is not None and brk:
            mk(-1, "breakout", s0 - P["brk_off"] * a, s0 + buf, below[1:])
    else:
        if s0 is not None:
            mk(1, "range", s0 + P["front"] * a, s0 - buf, above)
        if r0 is not None:
            mk(-1, "range", r0 - P["front"] * a, r0 + buf, below)
    return out


def simulate(p: Prepared, i0: int, ps: list[dict], spread: float) -> tuple[dict | None, int]:
    """One-cancels-other plan set from hourly bar i0. Returns (closed trade or None, next free bar)."""
    P = PARAMS
    n = len(p.t)
    live = [True] * len(ps)
    fill, fi = None, -1
    i = i0
    end = p.t[i0] + P["valid_h"] * 3600
    while i < n and p.t[i] < end and fill is None:
        h, lo, c = p.h[i], p.lo[i], p.c[i]
        for k, x in enumerate(ps):
            if not live[k]:
                continue
            d = x["dir"]
            if (d > 0 and h >= x["tp"]) or (d < 0 and lo <= x["tp"]):
                live[k] = False
                continue
            if x["kind"] == "breakout":
                hit = h + spread >= x["entry"] if d > 0 else lo <= x["entry"]
            else:
                hit = lo + spread <= x["entry"] if d > 0 else h >= x["entry"]
            if hit:
                fill, fi = x, i
                break
        if fill is None:
            for k, x in enumerate(ps):
                if live[k] and x["kind"] != "breakout" and x["dir"] * (c - x["stop"]) < 0:
                    live[k] = False
            if not any(live):
                return None, i + 1
            i += 1
    if fill is None:
        return None, i
    d, e, risk = fill["dir"], fill["entry"], fill["risk"]
    cur, be = fill["stop"], False
    stop_t = p.t[fi] + P["hold_h"] * 3600
    cost = P["cost_spreads"] * spread
    j24 = min(fi + 24, n - 1)

    def done(why: str, price: float, q: int):
        r = (d * (price - e) - (cost - spread)) / risk
        return dict(dir=d, kind=fill["kind"], why=why, r=float(r), t=float(p.t[fi]), fill_bar=q == fi, entry=float(p.c[fi]),
                    rr=d * (fill["tp"] - e) / risk, fwd_px=float(p.c[j24])), q + 1

    for q in range(fi, n):
        hi = p.h[q] + spread if d < 0 else p.h[q]
        low = p.lo[q] + spread if d < 0 else p.lo[q]
        if (low <= cur) if d > 0 else (hi >= cur):
            return done("BE" if be else "SL", cur, q)
        if q > fi and ((hi >= fill["tp"]) if d > 0 else (low <= fill["tp"])):
            return done("TP", fill["tp"], q)
        fr = d * ((hi if d > 0 else low) - e) / risk
        if q > fi and not be and fr >= P["be"]:
            cur, be = e + d * 0.05 * risk, True
        if p.t[q] >= stop_t:
            return done("TIME", p.c[q] + spread if d < 0 else p.c[q], q)
    return None, n


def run_symbol(p: Prepared, spread: float, variants=VARIANTS, start: float = 0.0) -> tuple[dict, list]:
    """Trades per variant, plus the hourly state log for the stop-free direction check."""
    cache: dict[int, tuple] = {}

    def ctx(i: int):
        if i not in cache:
            s = state_at(p, i)
            cache[i] = (s, levels_at(p, s)) if s else (None, None)
        return cache[i]

    trades = {v: [] for v in variants}
    first = int(np.searchsorted(p.t, start))
    for v in variants:
        i = max(first, 80)
        while i < len(p.t) - 1:
            s, lv = ctx(i)
            if s is None:
                i += 1
                continue
            ps = plans(v, s, lv[0], lv[1], spread)
            if not ps:
                i += 1
                continue
            tr, nxt = simulate(p, i, ps, spread)
            if tr:
                tr.update(base=s["base"], lt=s["lt"], ext=s["ext"], reb=s["reb"],
                          fwd=tr["dir"] * (tr.pop("fwd_px") - tr.pop("entry")) / s["atr_d"])
                trades[v].append(tr)
            i = max(nxt, i + 1)
    states = []
    for i in range(max(first, 80), len(p.t) - 25, 4):  # every 4th hour is enough for the state check
        s = state_at(p, i)
        if s:
            states.append((float(p.t[i]), s["base"], s["lt"], s["ext"], s["reb"], (float(p.c[i + 23]) - s["px"]) / s["atr_d"]))
    return trades, states
