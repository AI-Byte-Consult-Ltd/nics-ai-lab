"""False breakouts on long hourly history: can they be told apart at the moment of entry?

Research only. Nothing here sends a signal or changes the scanner, the level method or any EA.

Question (2026-11-24): on 10 weeks of our own M5 data about 55-70 % of breakdown shorts turned out
false, and no simple feature separated them at entry. This module repeats the question on public
hourly bars over many years, both sides (breakdowns and breakouts up), with features known at
the close of the breakout bar, and checks every finding on years that were not used to find it.

Event: at the close of hour i the close crosses the level L (the lowest low / highest high of
hours i-W .. i-G, i.e. W hours back without the last G hours) while the previous close was still
on the other side. One event per symbol and side per COOLDOWN hours.

Label (looks ahead, used only to describe and to fit on the in-sample years): FALSE when an
hourly close returns to the other side of L within FB hours, before price moved 1 ATR(H1) in the
breakout direction.

Trade (for R): entry at the close of the breakout bar, stop and take 1 ATR(H1) each side, 24 h
hold, costs 1.2 spreads. Hourly bars cannot show what came first inside an hour: when stop and
take are both inside one bar the trade counts as 0 R (unknown order) and the share of such
trades is reported. A construction that enters less often is compared both on all events
(missed = 0 R) and per filled trade, because with a negative average fewer trades alone look
"better".

Entry constructions compared on the same events (no event is dropped by a rule; an entry that
never comes is a missed trade, counted as 0 R):
  base      market at the close of the breakout bar
  confirm2  market at the close of the next bar only if it also closes beyond L (2 closes)
  retest    limit at L within RETEST_H hours after the break, stop/take 1 ATR from L (in the fill
            bar a loss only if the bar closed beyond the stop: slightly optimistic for retest)

Warning learned from the synthetic test: the share of false breakouts can be predicted even on a
random walk (a shallow close beyond the level returns more often, purely mechanically). So a
feature or model is judged by R of the trades, never by the share of false breakouts alone.

Features at the close of the breakout bar (no lookahead): level touches, level age, round-number
level, hour of day (UTC), D1 trend and H4 trend relative to the break, extension from D1 EMA20,
break depth, bar range, close position in the bar, approach speed over 6 hours, tick-volume ratio.
"""
from __future__ import annotations

import math

import numpy as np

DAY = 86400
DAY_SHIFT = 7200
PARAMS = dict(window_h=120, gap_h=6, cooldown_h=6, fb_h=6, hold_h=24, retest_h=6, cost_spreads=1.2,
              touch_atr=0.15, round_atr=0.25)
FEATURES = ("touches", "age_h", "round", "hour", "d1_with", "h4_with", "ext_d1", "depth", "bar_range",
            "close_pos", "approach", "vol_ratio")
BUCKETS = {
    "touches": (2, 4), "age_h": (24, 72), "round": (0.5,), "hour": (7, 12, 17), "d1_with": (-0.5, 0.5),
    "h4_with": (0,), "ext_d1": (0, 1, 2), "depth": (0.1, 0.3, 0.6), "bar_range": (0.8, 1.3, 2.0),
    "close_pos": (0.5, 0.8), "approach": (0.5, 1.5, 3.0), "vol_ratio": (0.8, 1.3, 2.0),
}


def ema(x: np.ndarray, k: int) -> np.ndarray:
    a = 2.0 / (k + 1)
    out = np.empty(len(x))
    e = x[0]
    for i, v in enumerate(x):
        e = v if i == 0 else v * a + e * (1 - a)
        out[i] = e
    return out


def _agg(t, h, lo, c, keys):
    starts = np.r_[0, np.nonzero(np.diff(keys))[0] + 1]
    return keys[starts], np.maximum.reduceat(h, starts), np.minimum.reduceat(lo, starts), c[np.r_[starts[1:] - 1, len(c) - 1]]


def _atr(h, lo, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    cs = np.cumsum(np.r_[0.0, tr])
    out = np.full(len(tr), np.nan)
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def round_step(price: float) -> float:
    return 10.0 ** (math.floor(math.log10(price)) - 2) if price > 0 else 0.0


def events(t, h, lo, c, v=None, spread: float = 0.0, P: dict | None = None) -> list[dict]:
    """All breakout events of one symbol with features, label and the three entry constructions."""
    P = {**PARAMS, **(P or {})}
    n = len(c)
    W, G = P["window_h"], P["gap_h"]
    if n < W + 400:
        return []
    v = np.zeros(n) if v is None else np.asarray(v, dtype=float)
    atr_h = _atr(h, lo, c)
    # daily and 4-hour context: only CLOSED bars before hour i
    dk = ((t + DAY_SHIFT) // DAY).astype(np.int64)
    dkey, dh, dl, dc = _agg(t, h, lo, c, dk)
    atr_d, e20d = _atr(dh, dl, dc), ema(dc, 20)
    td = np.zeros(len(dc))
    td[3:] = np.where((dc[3:] > e20d[3:]) & (e20d[3:] > e20d[:-3]), 1, np.where((dc[3:] < e20d[3:]) & (e20d[3:] < e20d[:-3]), -1, 0))
    nd = np.searchsorted(dkey, dk) - 1          # last closed day at each hour
    k4 = (t // 14400).astype(np.int64)
    hkey, _, _, hc = _agg(t, h, lo, c, k4)
    t4 = np.where(ema(hc, 20) > ema(hc, 50), 1, -1)
    n4 = np.searchsorted(hkey, k4) - 1
    from numpy.lib.stride_tricks import sliding_window_view as swv
    span = W - G + 1
    lmin = np.full(n, np.nan); hmax = np.full(n, np.nan); amin = np.zeros(n, int); amax = np.zeros(n, int)
    wl, wh = swv(lo, span), swv(h, span)            # row j covers bars j .. j+span-1
    rows = np.arange(n - W, dtype=int)              # hour i = row + W, window rows i-W .. i-G
    lmin[W:] = wl[rows].min(axis=1); hmax[W:] = wh[rows].max(axis=1)
    amin[W:] = rows + wl[rows].argmin(axis=1); amax[W:] = rows + wh[rows].argmax(axis=1)
    out: list[dict] = []
    last = {1: -10**9, -1: -10**9}
    cost = P["cost_spreads"] * spread
    for i in range(W + 1, n - 1):
        a = atr_h[i]
        if not (a > 0) or nd[i] < 25 or n4[i] < 60:
            continue
        for d in (-1, 1):
            L = lmin[i] if d < 0 else hmax[i]
            if not (d * (c[i] - L) > 0 and d * (c[i - 1] - L) <= 0) or i - last[d] < P["cooldown_h"]:
                continue
            if d * (c[i - 1] - (lmin[i - 1] if d < 0 else hmax[i - 1])) > 0:
                continue  # the previous hour already closed beyond its own level: not a fresh break
            last[d] = i
            e, ad, j0 = c[i], atr_d[nd[i]], int(amin[i] if d < 0 else amax[i])
            win = slice(i - W, i - G + 1)
            ref = lo[win] if d < 0 else h[win]
            near = np.abs(ref - L) <= P["touch_atr"] * a
            touches = int(np.sum(near & ~np.r_[False, near[:-1]]))  # separate visits to the level
            step = round_step(abs(L))
            rd = abs(L - round(L / step) * step) / a if step > 0 else 99.0
            rng = h[i] - lo[i]
            vm = v[max(0, i - 24):i].mean() if i > 0 else 0.0
            f = {
                "touches": touches, "age_h": i - j0, "round": 1.0 if rd <= P["round_atr"] else 0.0,
                "hour": int((t[i] // 3600) % 24), "d1_with": float(d * td[nd[i]]), "h4_with": float(d * t4[n4[i]]),
                "ext_d1": float(d * (c[i] - e20d[nd[i]]) / ad) if ad > 0 else 0.0,
                "depth": float(d * (c[i] - L) / a), "bar_range": float(rng / a),
                "close_pos": float((h[i] - c[i]) / rng if d < 0 else (c[i] - lo[i]) / rng) if rng > 0 else 0.5,
                "approach": float(d * (c[i] - c[max(0, i - 6)]) / a),
                "vol_ratio": float(v[i] / vm) if vm > 0 and v[i] > 0 else float("nan"),
            }
            # label: close back on the other side of L within fb_h hours, before 1 ATR in the break direction
            fb = False
            for j in range(i + 1, min(n, i + 1 + P["fb_h"])):
                if (lo[j] <= e - a) if d < 0 else (h[j] >= e + a):
                    break  # moved 1 ATR in the break direction first: a true breakout
                if d * (c[j] - L) <= 0:
                    fb = True
                    break
            out.append({"i": i, "t": float(t[i]), "dir": d, "level": float(L), "atr": float(a), "false": fb, **f,
                        "base": _trade(h, lo, c, i + 1, d, e, a, spread, cost, P["hold_h"]),
                        "confirm2": (_trade(h, lo, c, i + 2, d, c[i + 1], a, spread, cost, P["hold_h"])
                                     if i + 2 < n and d * (c[i + 1] - L) > 0 and not _hit(h, lo, i + 1, d, e, a, spread) else None),
                        "retest": _retest(h, lo, c, i, d, L, a, spread, cost, P)})
    return out


AMBIGUOUS = [0]   # trades whose stop and take were both inside one bar (reset by the caller)
AMBIGUOUS_R = 0.0


def _hit(h, lo, j, d, e, a, spread) -> bool:
    """Stop of a trade opened at e was touched in bar j (shorts exit at the ask)."""
    return (h[j] + spread >= e + a) if d < 0 else (lo[j] <= e - a)


def _trade(h, lo, c, j0, d, e, a, spread, cost, hold) -> float | None:
    """R of a 1:1 trade (stop and take 1 ATR) from bar j0.

    When stop and take are both inside one hourly bar the order is unknown. v0.20.0 counted that
    as a loss; on real data after breakouts this happens often and pushed every variant to about
    -0.3 R. Since v0.20.1 such a trade counts as AMBIGUOUS_R (0 R, the average of +1 and -1 minus
    costs) and is counted in AMBIGUOUS so the share is visible."""
    n = len(c)
    if j0 >= n:
        return None
    for j in range(j0, min(n, j0 + hold)):
        stop = _hit(h, lo, j, d, e, a, spread)
        take = (lo[j] <= e - a) if d < 0 else (h[j] >= e + a)
        if stop and take:
            AMBIGUOUS[0] += 1
            return AMBIGUOUS_R - cost / a
        if stop:
            return -1.0 - cost / a
        if take:
            return 1.0 - cost / a
    j = min(n, j0 + hold) - 1
    return float(d * (c[j] - e) - cost) / a


def _retest(h, lo, c, i, d, L, a, spread, cost, P) -> float | None:
    n = len(c)
    for j in range(i + 1, min(n, i + 1 + P["retest_h"])):
        if (lo[j] <= c[i] - a) if d < 0 else (h[j] >= c[i] + a):
            return None  # ran away to the take of the market entry: no retest
        if (h[j] >= L) if d < 0 else (lo[j] <= L):
            # inside the fill bar the order of touches is unknown: a loss only if the bar CLOSED beyond the stop
            if (c[j] + spread >= L + a) if d < 0 else (c[j] <= L - a):
                return -1.0 - cost / a
            return _trade(h, lo, c, j + 1, d, L, a, spread, cost, P["hold_h"])
    return None


# ----------------------------------------------------------------------------- analysis


def _r(x: list) -> dict:
    if not x:
        return {"n": 0}
    r = np.array(x, dtype=float)
    se = r.std(ddof=1) / math.sqrt(len(r)) if len(r) > 1 else 0.0
    return {"n": len(r), "R": round(float(r.sum()), 1), "avg": round(float(r.mean()), 3),
            "t": round(float(r.mean() / se), 1) if se > 0 else 0.0}


def describe(ev: list[dict]) -> dict:
    if not ev:
        return {"n": 0}
    base = [e["base"] for e in ev if e["base"] is not None]
    return {"n": len(ev), "false_pct": round(100 * float(np.mean([e["false"] for e in ev]))), **{k: v for k, v in _r(base).items() if k != "n"}}


def constructions(ev: list[dict]) -> dict:
    """Each construction on ALL events: a missed entry counts as 0 R (nothing is filtered out)."""
    out = {}
    for k in ("base", "confirm2", "retest"):
        r = [e[k] if e[k] is not None else 0.0 for e in ev]
        filled = sum(e[k] is not None for e in ev)
        per = [e[k] for e in ev if e[k] is not None]
        out[k] = {**_r(r), "filled": filled, "avg_filled": round(float(np.mean(per)), 3) if per else 0.0}
    return out


def bucket_of(name: str, x: float) -> int:
    if x != x:  # nan
        return -1
    return int(np.searchsorted(np.array(BUCKETS[name]), x, side="right"))


def by_feature(ev: list[dict]) -> dict:
    out = {}
    for name in FEATURES:
        cuts = BUCKETS[name]
        o = {}
        for b in range(-1, len(cuts) + 1):
            sub = [e for e in ev if bucket_of(name, e[name]) == b]
            if not sub:
                continue
            lab = "нет данных" if b < 0 else (f"<{cuts[0]:g}" if b == 0 else f">={cuts[-1]:g}" if b == len(cuts) else f"{cuts[b - 1]:g}-{cuts[b]:g}")
            o[lab] = describe(sub)
        out[name] = o
    return out


def design(ev: list[dict]) -> np.ndarray:
    """One-hot buckets of every feature (no scaling problems, robust to outliers)."""
    cols = []
    for name in FEATURES:
        k = len(BUCKETS[name]) + 1
        m = np.zeros((len(ev), k + 1))
        for r, e in enumerate(ev):
            m[r, bucket_of(name, e[name]) + 1] = 1.0
        cols.append(m[:, 1:])  # bucket -1 (no data) is the reference
    return np.hstack(cols + [np.ones((len(ev), 1))])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 300) -> np.ndarray:
    """Plain L2 logistic regression by Newton steps (numpy only)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return 0.5
    order = np.argsort(np.r_[pos, neg])
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def model_check(ev: list[dict], split_t: float) -> dict:
    """Fit 'probability of a false breakout' on years before split_t, judge it only after."""
    ins = [e for e in ev if e["t"] < split_t]
    oos = [e for e in ev if e["t"] >= split_t]
    if len(ins) < 500 or len(oos) < 200:
        return {"note": "too few events for an in-sample / out-of-sample check", "n_in": len(ins), "n_out": len(oos)}
    w = fit_logistic(design(ins), np.array([e["false"] for e in ins], float))
    res = {"n_in": len(ins), "n_out": len(oos)}
    for name, sub in (("in", ins), ("out", oos)):
        s = design(sub) @ w
        y = np.array([e["false"] for e in sub], float)
        q = np.quantile(s, [1 / 3, 2 / 3])
        terc = {}
        for k, (lo_, hi_) in enumerate(((-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf))):
            part = [e for e, x in zip(sub, s) if lo_ <= x < hi_ or (k == 2 and x == hi_)]
            terc[("low", "mid", "high")[k]] = {**describe(part), "constructions": constructions(part)}
        res[name] = {"auc": round(auc(s, y), 3), "tercile_by_false_probability": terc}
    res["weights"] = [round(float(x), 3) for x in w]
    return res
