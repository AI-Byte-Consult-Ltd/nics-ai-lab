"""Candidate trading rules in SHADOW: measured, not yet taught to the model.

A book methodology becomes a rule written as code here (our own parameters; no book
text is copied). Each candidate is measured on its own every night by
scripts/rules_shadow.py: on long public history (backtest) and, from the day it was
added, forward on our broker candles (the real shadow test). Only a rule that holds
up forward for 2-4 weeks is promoted into training (moved to methods.py + a new kind
in data.py, which changes the feature schema once). The full list, with results and
decisions, is docs/rules_registry.md, so no rule is added twice.

Every rule returns pending orders: {kind, symbol, direction, bar, entry, sl, tp1,
valid_bars}. The order is decided at the close of bar `bar` and fills only if price
trades through `entry` within the next `valid_bars` bars (stop entry), so results are
not flattered by entering at a price that was never offered.
"""
from __future__ import annotations

import numpy as np

from nics.market.indicators import adx, ema, rolling

CANDIDATES = {
    # id: (function name, inspired by, added on, status)
    "rule_strong_trend_pullback": ("strong_trend_pullback", "Raschke & Connors (ADX trend pullback)",
                                   "2026-11-21", "shadow"),
}


def _ctx(s):
    return {
        "a": s.atr, "e20": ema(s.c, 20), "adx": adx(s.h, s.l, s.c),
        "hi20": np.concatenate([[np.nan], rolling(s.h, 20, np.max)[:-1]]),
        "lo20": np.concatenate([[np.nan], rolling(s.l, 20, np.min)[:-1]]),
    }


def strong_trend_pullback(s, x, start):
    """Strong trend (ADX > 30, EMA20 sloping) pulls back to EMA20. Buy stop just above the
    pullback bar's high (mirror for shorts); stop beyond the last 3 bars' extreme with a
    0.4 ATR buffer; target = the high (low) of the 20 bars before, at least 1 R away."""
    out = []
    e20, ax, a = x["e20"], x["adx"], x["a"]
    for i in range(max(start, 5), len(s.c) - 1):
        if not (a[i] > 0) or np.isnan(ax[i]) or np.isnan(x["hi20"][i]) or ax[i] <= 30:
            continue
        if e20[i] > e20[i - 5] and s.l[i] <= e20[i] and s.c[i] > e20[i] - 0.25 * a[i]:
            entry = s.h[i] + 0.05 * a[i]
            sl = s.l[i - 2:i + 1].min() - 0.4 * a[i]
            risk = entry - sl
            tp = max(x["hi20"][i], entry + risk)
            out.append({"direction": 1, "bar": i, "entry": entry, "sl": sl, "tp1": tp})
        elif e20[i] < e20[i - 5] and s.h[i] >= e20[i] and s.c[i] < e20[i] + 0.25 * a[i]:
            entry = s.l[i] - 0.05 * a[i]
            sl = s.h[i - 2:i + 1].max() + 0.4 * a[i]
            risk = sl - entry
            tp = min(x["lo20"][i], entry - risk)
            out.append({"direction": -1, "bar": i, "entry": entry, "sl": sl, "tp1": tp})
    for st in out:
        st.update(kind="rule_strong_trend_pullback", symbol=s.symbol, valid_bars=3)
    return out


def run(rule_id: str, s, warmup: int, cooldown_bars: int = 6) -> list[dict]:
    fn = globals()[CANDIDATES[rule_id][0]]
    if s.atr is None:
        return []
    last, out = {1: -10**9, -1: -10**9}, []
    for st in fn(s, _ctx(s), warmup):
        if st["bar"] - last[st["direction"]] >= cooldown_bars:
            last[st["direction"]] = st["bar"]
            out.append(st)
    return out


def outcome(s, st: dict, max_bars: int):
    """(filled, label, fill_bar, end_bar). label 1 = TP1 before SL, 0 = SL first; a bar that
    touches both, or the fill bar touching the stop, counts as SL (conservative).
    label None = filled but still open."""
    d, i = st["direction"], st["bar"]
    fill = None
    for j in range(i + 1, min(i + 1 + st["valid_bars"], len(s.c))):
        if (s.h[j] >= st["entry"]) if d > 0 else (s.l[j] <= st["entry"]):
            fill = j
            break
    if fill is None:
        return False, None, None, None
    for k in range(fill, min(fill + max_bars, len(s.c))):
        hit_sl = s.l[k] <= st["sl"] if d > 0 else s.h[k] >= st["sl"]
        hit_tp = (s.h[k] >= st["tp1"] if d > 0 else s.l[k] <= st["tp1"]) and k > fill
        if hit_sl:
            return True, 0, fill, k
        if hit_tp:
            return True, 1, fill, k
    return True, None, fill, None
