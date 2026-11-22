"""Daily trend context: the trend methods that beat ours on long history, as model features.

Research 22.11.2026 (docs/methods_research_2026-11-22.md): ten well-known trading methods were
coded as rules with our own parameters and backtested on 36 instruments, 1990-2026, costs and
swaps included, out-of-sample from 2018. Three trend methods passed the pre-registered bar
(out-of-sample average R > 0 with t >= 2, profit factor >= 1.2, positive on more than half of
the instruments, better than random entries in the same direction): 12-month momentum,
50/200 moving-average trend and the 55/20 channel system. Two more were positive but not
significant (cloud trend, MACD trend); they are added as features too and the model decides
how much they are worth. None of them is a filter: they only describe the daily trend.

Every bar of every timeframe gets the state of each method on the instrument's DAILY candles,
as of the last daily candle that had CLOSED by the end of that bar (strictly causal):
+1 = the method is long, -1 = short, 0 = flat / not ready. Plus the average vote.

Daily candles come from broker D1 (bridge 1.36+) where they exist, public D1 history before
that (scaled to broker prices; the states do not depend on the price scale). Live scoring
fetches only intraday candles, so the nightly cycle stores the context in
data/context/daily_trend.csv and live scoring reads it from there (states are at most one
daily close old there, and they change slowly).

Neutral names only (no authors, schools or companies).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from nics.market.indicators import ema, rolling

DAY = 86400
FEATURE_METHODS = {
    # id: neutral description
    "trend_momentum_12m": "12-month momentum: sign of the 252-day return",
    "trend_ma_50_200": "moving-average trend: EMA50 above / below EMA200",
    "trend_channel_55_20": "channel system: long after a close above the 55-day high until a close "
                           "below the 20-day low (mirror for shorts)",
    "trend_cloud": "cloud trend: close above the cloud and the 9-line above the 26-line (mirror)",
    "trend_macd": "MACD trend: MACD and its signal line agree with the sign of MACD",
}
TREND_FEATURES = [f"d1_{k}" for k in FEATURE_METHODS] + ["d1_trend_vote"]
CACHE = Path(__file__).resolve().parents[3] / "data" / "context" / "daily_trend.csv"


def _prev(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) > n:
        out[n:] = x[:-n]
    return out


def daily_states(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """(n, 5) states on daily bars, each in {-1, 0, +1}; uses bars <= i only."""
    n = len(c)
    out = np.zeros((n, len(FEATURE_METHODS)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = c / _prev(c, 252) - 1
    out[:, 0] = np.nan_to_num(np.sign(ret))
    e50, e200 = ema(c, 50), ema(c, 200)
    out[:, 1] = np.nan_to_num(np.sign(e50 - e200))
    hi55, lo55 = _prev(rolling(h, 55, np.max), 1), _prev(rolling(l, 55, np.min), 1)
    hi20, lo20 = _prev(rolling(h, 20, np.max), 1), _prev(rolling(l, 20, np.min), 1)
    st = 0
    for i in range(n):
        if st > 0 and c[i] < lo20[i]:
            st = 0
        elif st < 0 and c[i] > hi20[i]:
            st = 0
        if c[i] > hi55[i]:
            st = 1
        elif c[i] < lo55[i]:
            st = -1
        out[i, 2] = st
    ten = (rolling(h, 9, np.max) + rolling(l, 9, np.min)) / 2
    kij = (rolling(h, 26, np.max) + rolling(l, 26, np.min)) / 2
    sa, sb = _prev((ten + kij) / 2, 26), _prev((rolling(h, 52, np.max) + rolling(l, 52, np.min)) / 2, 26)
    top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
    out[:, 3] = np.where((c > top) & (ten > kij), 1, np.where((c < bot) & (ten < kij), -1, 0))
    macd = ema(c, 12) - ema(c, 26)
    sig = np.full(n, np.nan)
    ok = ~np.isnan(macd)
    if ok.sum() >= 9:
        sig[ok] = ema(macd[ok], 9)
    out[:, 4] = np.where((macd > 0) & (macd > sig), 1, np.where((macd < 0) & (macd < sig), -1, 0))
    return out


def _spliced_daily(series) -> dict:
    """symbol -> (open times, high, low, close): broker D1, with public D1 before it (scaled)."""
    pub = {s.symbol: s for s in series if s.public and s.interval == "1day"}
    brk = {s.symbol: s for s in series if not s.public and s.interval == "1day"}
    out = {}
    for sym in set(pub) | set(brk):
        p, b = pub.get(sym), brk.get(sym)
        if b is None or len(b.c) == 0:
            out[sym] = (p.t, p.h, p.l, p.c)
            continue
        if p is None:
            out[sym] = (b.t, b.h, b.l, b.c)
            continue
        bday = {int(t // DAY): x for t, x in zip(b.t, b.c)}
        ratio = [bday[int(t // DAY)] / x for t, x in zip(p.t, p.c) if int(t // DAY) in bday and x > 0]
        k = float(np.median(ratio)) if len(ratio) >= 10 else 1.0
        m = p.t < b.t[0]
        out[sym] = (np.r_[p.t[m], b.t], np.r_[p.h[m] * k, b.h], np.r_[p.l[m] * k, b.l], np.r_[p.c[m] * k, b.c])
    return out


def build(series) -> dict:
    """symbol -> (close times of daily bars, states (n, 5))."""
    ctx = {}
    for sym, (t, h, l, c) in _spliced_daily(series).items():
        if len(c) >= 30:
            ctx[sym] = (t + DAY, daily_states(h, l, c))
    return ctx


def save(ctx: dict, path: Path = CACHE, keep_days: int = 30) -> None:
    """Only the last `keep_days` daily states per symbol: enough for live scoring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "closed_at"] + list(FEATURE_METHODS))
        for sym, (tc, st) in sorted(ctx.items()):
            for i in range(max(0, len(tc) - keep_days), len(tc)):
                w.writerow([sym, int(tc[i])] + [int(v) for v in st[i]])


def load(path: Path = CACHE) -> dict:
    if not path.exists():
        return {}
    rows: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["symbol"], []).append([float(r["closed_at"])] + [float(r[k]) for k in FEATURE_METHODS])
    return {s: (np.array(v)[:, 0], np.array(v)[:, 1:]) for s, v in rows.items()}


def attach(series, ctx: dict) -> None:
    """Sets s.htf (n, len(TREND_FEATURES)) on every series: the daily states as of each bar's close."""
    for s in series:
        out = np.zeros((len(s.c), len(TREND_FEATURES)))
        if s.symbol in ctx:
            tc, st = ctx[s.symbol]
            j = np.searchsorted(tc, s.t + s.bar_seconds, side="right") - 1
            ok = j >= 0
            out[ok, :-1] = st[j[ok]]
            out[:, -1] = out[:, :-1].mean(axis=1)
        s.htf = out


def prepare(series, save_to: Path | None = None, load_from: Path = CACHE) -> dict:
    """Context for build_samples: from the snapshot's daily candles when it has them (and then
    refresh the cache for live scoring, only for real snapshots), else from the cache."""
    ctx = build(series)
    if ctx and save_to is not None:
        try:
            save(ctx, save_to)
        except OSError:
            pass
    elif not ctx:
        ctx = load(load_from)
    attach(series, ctx)
    return ctx
