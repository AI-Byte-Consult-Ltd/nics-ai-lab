"""Trading methodologies: rules that turn candles into trade setups on history.

This is how NICS Trading learns from trading methods. Each methodology is a
plain function that walks a broker candle series bar by bar (using only bars
already closed) and proposes setups with entry, stop and target. The
training script labels every setup from later candles (TP1 before SL) and
the signal head learns which setups of which method, in which market
context, actually work. The methodology name is an input feature.

To teach the model a new method:
  1. write a function `my_method(s, ctx) -> list[dict]` below (same shape as
     the others: id, kind, symbol, direction, t, entry, sl, tp1);
  2. add it to METHODS and its kind to METHOD_KINDS in data.py;
  3. retrain (the self-learning cycle picks it up automatically).

Stops sit beyond the level with a buffer that is not the obvious spot
(stop hunting), and both sides of a range are traded, per our manual
trading rules. These are training material, not live signals.
"""
from __future__ import annotations

import numpy as np

from nics.market.indicators import adx, ema, rolling


def _ctx(s):
    a = s.atr
    return {
        "a": a,
        "e20": ema(s.c, 20), "e50": ema(s.c, 50),
        "adx": adx(s.h, s.l, s.c),
        # channel of the PREVIOUS bars (excludes the current bar)
        "hi20": np.concatenate([[np.nan], rolling(s.h, 20, np.max)[:-1]]),
        "lo20": np.concatenate([[np.nan], rolling(s.l, 20, np.min)[:-1]]),
        "hi48": np.concatenate([[np.nan], rolling(s.h, 48, np.max)[:-1]]),
        "lo48": np.concatenate([[np.nan], rolling(s.l, 48, np.min)[:-1]]),
    }


def _setup(s, kind, i, direction, sl, tp):
    return {"id": f"{kind}-{s.symbol}-{s.interval}-{int(s.t[i])}-{'L' if direction > 0 else 'S'}",
            "kind": kind, "symbol": s.symbol, "direction": direction,
            "t": float(s.t[i] + s.bar_seconds),  # decided when bar i has closed
            "entry": float(s.c[i]), "sl": float(sl), "tp1": float(tp), "bar": i}


def range_both_sides(s, x, start):
    """Range (ADX < 20, channel >= 3 ATR): long from the bottom, short from the top."""
    out = []
    for i in range(start, len(s.c)):
        a, lo, hi = x["a"][i], x["lo48"][i], x["hi48"][i]
        if not (a > 0) or np.isnan(lo) or x["adx"][i] >= 20 or hi - lo < 3 * a:
            continue
        mid = (hi + lo) / 2
        if s.l[i] <= lo + 0.1 * a and s.c[i] > lo:
            out.append(_setup(s, "method_range_both_sides", i, 1, lo - 0.6 * a, mid))
        elif s.h[i] >= hi - 0.1 * a and s.c[i] < hi:
            out.append(_setup(s, "method_range_both_sides", i, -1, hi + 0.6 * a, mid))
    return out


def breakout20(s, x, start):
    """Close beyond the 20-bar channel with a trend (ADX > 20)."""
    out = []
    for i in range(start, len(s.c)):
        a = x["a"][i]
        if not (a > 0) or np.isnan(x["hi20"][i]) or x["adx"][i] <= 20:
            continue
        if s.c[i] > x["hi20"][i]:
            out.append(_setup(s, "method_breakout20", i, 1, s.c[i] - 1.5 * a, s.c[i] + 2.0 * a))
        elif s.c[i] < x["lo20"][i]:
            out.append(_setup(s, "method_breakout20", i, -1, s.c[i] + 1.5 * a, s.c[i] - 2.0 * a))
    return out


def ema_pullback(s, x, start):
    """Trend EMA20 > EMA50 (both rising): buy the touch of EMA20 that closes back above; mirror."""
    out = []
    e20, e50 = x["e20"], x["e50"]
    for i in range(max(start, 5), len(s.c)):
        a = x["a"][i]
        if not (a > 0) or np.isnan(e50[i]) or np.isnan(e50[i - 5]):
            continue
        if e20[i] > e50[i] and e50[i] > e50[i - 5] and s.l[i] <= e20[i] < s.c[i]:
            sl = s.l[i - 4:i + 1].min() - 0.5 * a
            out.append(_setup(s, "method_ema_pullback", i, 1, sl, s.c[i] + 1.5 * (s.c[i] - sl)))
        elif e20[i] < e50[i] and e50[i] < e50[i - 5] and s.h[i] >= e20[i] > s.c[i]:
            sl = s.h[i - 4:i + 1].max() + 0.5 * a
            out.append(_setup(s, "method_ema_pullback", i, -1, sl, s.c[i] - 1.5 * (sl - s.c[i])))
    return out


def level_bounce(s, x, start, lookback=200):
    """Swing level touched at least twice before; price comes back and closes off it."""
    out = []
    n = len(s.c)
    # fractal swing points, known only 2 bars after they form
    swing_lo = np.zeros(n, bool)
    swing_hi = np.zeros(n, bool)
    for j in range(2, n - 2):
        swing_lo[j] = s.l[j] < min(s.l[j - 2], s.l[j - 1], s.l[j + 1], s.l[j + 2])
        swing_hi[j] = s.h[j] > max(s.h[j - 2], s.h[j - 1], s.h[j + 1], s.h[j + 2])
    for i in range(start, n):
        a = x["a"][i]
        if not (a > 0):
            continue
        j0, j1 = max(0, i - lookback), i - 2  # swings confirmed by bar i
        lows, highs = s.l[j0:j1][swing_lo[j0:j1]], s.h[j0:j1][swing_hi[j0:j1]]
        for lv in lows[-12:]:
            if (abs(lows - lv) <= 0.25 * a).sum() >= 2 and s.l[i] <= lv + 0.2 * a and s.c[i] > lv:
                sl = lv - 0.7 * a
                out.append(_setup(s, "method_level_bounce", i, 1, sl, s.c[i] + 2 * (s.c[i] - sl)))
                break
        else:
            for lv in highs[-12:]:
                if (abs(highs - lv) <= 0.25 * a).sum() >= 2 and s.h[i] >= lv - 0.2 * a and s.c[i] < lv:
                    sl = lv + 0.7 * a
                    out.append(_setup(s, "method_level_bounce", i, -1, sl, s.c[i] - 2 * (sl - s.c[i])))
                    break
    return out


METHODS = {
    "method_range_both_sides": range_both_sides,
    "method_breakout20": breakout20,
    "method_ema_pullback": ema_pullback,
    "method_level_bounce": level_bounce,
}


def generate(series_list, intervals, warmup: int, cooldown_bars: int = 6, public: bool = False) -> list[dict]:
    """Run every methodology on every series of the given timeframes: broker series, and
    with public=True also long public history. Each setup carries "series" (index in
    series_list) so a public setup can be labelled on its own candles."""
    out = []
    for n, s in enumerate(series_list):
        if (s.public and not public) or s.interval not in intervals or s.atr is None:
            continue
        x = _ctx(s)
        for fn in METHODS.values():
            last = {1: -10**9, -1: -10**9}
            for st in fn(s, x, warmup):
                i = st.pop("bar")
                if i - last[st["direction"]] < cooldown_bars:
                    continue
                last[st["direction"]] = i
                st["series"] = n
                if s.public:
                    st["id"] = "pub-" + st["id"]
                out.append(st)
    return out
