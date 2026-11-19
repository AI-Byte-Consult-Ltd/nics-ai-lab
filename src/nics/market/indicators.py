"""Technical indicators and candle patterns, numpy only, strictly causal.

Every value at bar i uses bars <= i only. Outputs are scale-free (ATR units,
0..1 positions or -1..1 oscillators) so one model can learn them across
instruments and timeframes.
"""
from __future__ import annotations

import numpy as np


def ema(x: np.ndarray, period: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    k = 2.0 / (period + 1)
    acc = x[0]
    for i, v in enumerate(x):
        acc = v * k + acc * (1 - k) if i else v
        out[i] = acc
    out[: period - 1] = np.nan  # not warmed up yet
    return out


def wilder(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing (RMA), used by RSI and ADX."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) < period:
        return out
    acc = np.nanmean(x[:period])
    out[period - 1] = acc
    for i in range(period, len(x)):
        acc = (acc * (period - 1) + x[i]) / period
        out[i] = acc
    return out


def rolling(x: np.ndarray, n: int, fn) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = fn(np.lib.stride_tricks.sliding_window_view(x, n), axis=1)
    return out


def rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up, dn = wilder(np.maximum(d, 0), period), wilder(np.maximum(-d, 0), period)
    rs = np.divide(up, dn, out=np.full_like(up, np.nan), where=dn > 0)
    out = 100 - 100 / (1 + rs)
    return np.where((dn == 0) & (up > 0), 100.0, out)


def adx(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr_w = wilder(tr, period)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * wilder(plus, period) / atr_w
        mdi = 100 * wilder(minus, period) / atr_w
        dx = 100 * abs(pdi - mdi) / (pdi + mdi)
    return wilder(np.nan_to_num(dx, nan=0.0), period)


INDICATOR_FEATURES = [
    "rsi14",            # (RSI - 50) / 50
    "ema20_dist_atr",   # (close - EMA20) / ATR
    "ema50_dist_atr",
    "ema200_dist_atr",
    "ema20_slope_atr",  # EMA20 change over 5 bars / ATR
    "macd_hist_atr",    # MACD(12,26,9) histogram / ATR
    "bb_pos",           # (close - SMA20) / (2 * std20): Bollinger position
    "stoch_k",          # stochastic %K(14) scaled to -1..1
    "adx14",            # ADX / 50 - 1: trend strength (< -0.6 = range)
    "dist_high20_atr",  # distance to the 20-bar high, ATR units (levels / channel)
    "dist_low20_atr",
    "dist_high100_atr",
    "dist_low100_atr",
]
PATTERN_FEATURES = [
    "pat_bull_engulf", "pat_bear_engulf", "pat_hammer", "pat_shooting_star",
    "pat_doji", "pat_inside_bar", "pat_outside_bar",
]


def indicator_block(o, h, l, c, a) -> np.ndarray:
    """(n, len(INDICATOR_FEATURES) + len(PATTERN_FEATURES)); NaN until warmed up."""
    safe = np.where(a > 0, a, np.nan)
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    macd = ema(c, 12) - ema(c, 26)
    macd_sig = ema(np.nan_to_num(macd, nan=0.0), 9)
    sma20, std20 = rolling(c, 20, np.mean), rolling(c, 20, np.std)
    hi14, lo14 = rolling(h, 14, np.max), rolling(l, 14, np.min)
    hi20, lo20 = rolling(h, 20, np.max), rolling(l, 20, np.min)
    hi100, lo100 = rolling(h, 100, np.max), rolling(l, 100, np.min)
    e20_prev = np.concatenate([np.full(5, np.nan), e20[:-5]]) if len(c) > 5 else np.full(len(c), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ind = np.stack([
            (rsi(c) - 50) / 50,
            (c - e20) / safe, (c - e50) / safe, (c - e200) / safe,
            (e20 - e20_prev) / safe,
            (macd - macd_sig) / safe,
            np.where(std20 > 0, (c - sma20) / (2 * std20), 0.0),
            np.where(hi14 > lo14, 2 * (c - lo14) / (hi14 - lo14) - 1, 0.0),
            adx(h, l, c) / 50 - 1,
            (hi20 - c) / safe, (c - lo20) / safe, (hi100 - c) / safe, (c - lo100) / safe,
        ], axis=1)
    # e200 needs 200 bars; on short series keep it neutral instead of dropping every window
    ind[:, 3] = np.where(np.isnan(e200), 0.0, ind[:, 3])
    ind[:, 11] = np.where(np.isnan(hi100), ind[:, 9], ind[:, 11])
    ind[:, 12] = np.where(np.isnan(lo100), ind[:, 10], ind[:, 12])

    po, pc_ = np.concatenate([[o[0]], o[:-1]]), np.concatenate([[c[0]], c[:-1]])
    ph, pl = np.concatenate([[h[0]], h[:-1]]), np.concatenate([[l[0]], l[:-1]])
    body = abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    upper, lower = h - np.maximum(o, c), np.minimum(o, c) - l
    pat = np.stack([
        (c > o) & (pc_ < po) & (c >= po) & (o <= pc_),          # bullish engulfing
        (c < o) & (pc_ > po) & (c <= po) & (o >= pc_),          # bearish engulfing
        (lower >= 2 * body) & (upper <= body) & (body / rng < 0.4),  # hammer / bullish pin bar
        (upper >= 2 * body) & (lower <= body) & (body / rng < 0.4),  # shooting star / bearish pin bar
        body / rng < 0.1,                                       # doji
        (h <= ph) & (l >= pl),                                  # inside bar
        (h > ph) & (l < pl),                                    # outside bar
    ], axis=1).astype(np.float64)
    pat[0] = 0.0
    return np.concatenate([np.clip(ind, -20, 20), pat], axis=1)
