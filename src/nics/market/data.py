"""Market data for NICS Trading: snapshot loading, features, labels, splits.

A snapshot is a directory written by scripts/fetch_market_data.py:
  candles.csv   symbol,interval,candle_time,open,high,low,close,volume,bid,ask
  signals.csv   id,symbol,direction,entry_low,entry_high,stop_loss,tp1,
                confidence_score,technical_score,status,final_result,
                hit_targets,created_at,closed_at,origin_mode
  manifest.json exporter metadata (export time, row counts) + sha256 of each csv

All data comes from our own systems (Vantage MT5 bridge candles, our
scanner's signals). Nothing here downloads third-party data.

Leakage rules (docs/architecture.md, project instructions):
  - every feature at bar t uses only bars <= t;
  - labels look forward, so splits are by time with an embargo gap;
  - a signal's outcome is re-derived from candles after its creation time,
    and a bar touching both TP1 and SL counts as a loss (conservative).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ATR_PERIOD = 14
BAR_FEATURES = [
    "ret_atr",        # (close - prev close) / ATR
    "range_atr",      # (high - low) / ATR
    "body_atr",       # (close - open) / ATR
    "upper_wick_atr",
    "lower_wick_atr",
    "atr_pct",        # ATR / close * 100: volatility regime
    "vol_z",          # volume z-score over the window
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
SIGNAL_FEATURES = [
    "direction",      # +1 long, -1 short
    "entry_dist_atr", # (entry mid - close at signal) / ATR, signed by direction
    "sl_dist_atr",    # |entry mid - SL| / ATR
    "tp1_dist_atr",   # |TP1 - entry mid| / ATR
    "reward_risk",    # tp1_dist / sl_dist
    "spread_atr",     # (ask - bid) / ATR at signal time
]


def _ts(value: str) -> float:
    value = value.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(value).timestamp()


@dataclass
class Series:
    symbol: str
    t: np.ndarray       # unix seconds, ascending
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    spread: np.ndarray  # ask - bid, 0 where unknown


def load_candles(snapshot: Path, interval: str) -> dict[str, Series]:
    rows: dict[str, list] = {}
    with open(snapshot / "candles.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["interval"] != interval:
                continue
            bid = float(r["bid"]) if r.get("bid") not in (None, "") else 0.0
            ask = float(r["ask"]) if r.get("ask") not in (None, "") else 0.0
            rows.setdefault(r["symbol"], []).append((
                _ts(r["candle_time"]), float(r["open"]), float(r["high"]),
                float(r["low"]), float(r["close"]), float(r.get("volume") or 0.0),
                max(ask - bid, 0.0) if bid and ask else 0.0,
            ))
    out = {}
    for sym, rs in rows.items():
        rs.sort(key=lambda x: x[0])
        a = np.array(rs, dtype=np.float64)
        out[sym] = Series(sym, a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6])
    return out


def atr(s: Series, period: int = ATR_PERIOD) -> np.ndarray:
    prev_c = np.concatenate([[s.c[0]], s.c[:-1]])
    tr = np.maximum(s.h - s.l, np.maximum(abs(s.h - prev_c), abs(s.l - prev_c)))
    out = np.full_like(tr, np.nan)
    if len(tr) >= period:
        csum = np.cumsum(tr)
        out[period - 1:] = (csum[period - 1:] - np.concatenate([[0.0], csum[:-period]])) / period
    return out


def bar_features(s: Series) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features[n, len(BAR_FEATURES)], atr[n]). Rows before ATR is ready are NaN."""
    a = atr(s)
    safe = np.where(a > 0, a, np.nan)
    prev_c = np.concatenate([[s.c[0]], s.c[:-1]])
    top = np.maximum(s.o, s.c)
    bot = np.minimum(s.o, s.c)
    hours = (s.t % 86400) / 3600.0
    dow = ((s.t // 86400) + 3) % 7  # 1970-01-01 was a Thursday
    f = np.stack([
        (s.c - prev_c) / safe,
        (s.h - s.l) / safe,
        (s.c - s.o) / safe,
        (s.h - top) / safe,
        (bot - s.l) / safe,
        a / s.c * 100.0,
        np.zeros_like(s.c),  # vol_z filled per window
        np.sin(2 * math.pi * hours / 24), np.cos(2 * math.pi * hours / 24),
        np.sin(2 * math.pi * dow / 7), np.cos(2 * math.pi * dow / 7),
    ], axis=1)
    return np.clip(f, -20, 20), a


def window(features: np.ndarray, vol: np.ndarray, end: int, length: int) -> np.ndarray | None:
    """Window of `length` bars ending at index `end` (inclusive), or None if not available."""
    start = end - length + 1
    if start < 0:
        return None
    w = features[start:end + 1].copy()
    if np.isnan(w).any():
        return None
    v = vol[start:end + 1]
    sd = v.std()
    w[:, BAR_FEATURES.index("vol_z")] = (v - v.mean()) / sd if sd > 0 else 0.0
    return w.astype(np.float32)


def horizon_label(s: Series, a: np.ndarray, i: int, horizon: int, flat_atr: float) -> int | None:
    j = i + horizon
    if j >= len(s.c) or not a[i] > 0:
        return None
    move = (s.c[j] - s.c[i]) / a[i]
    return 0 if move < -flat_atr else (2 if move > flat_atr else 1)


def signal_outcome(s: Series, start: int, direction: int, sl: float, tp1: float, max_bars: int):
    """(label, resolving bar index): label 1 = TP1 before SL, 0 = SL first (or both in
    one bar). Returns (None, None) if neither is hit within max_bars."""
    for k in range(start, min(start + max_bars, len(s.c))):
        hit_sl = s.l[k] <= sl if direction > 0 else s.h[k] >= sl
        hit_tp = s.h[k] >= tp1 if direction > 0 else s.l[k] <= tp1
        if hit_sl:
            return 0, k
        if hit_tp:
            return 1, k
    return None, None


def load_signals(snapshot: Path) -> list[dict]:
    with open(snapshot / "signals.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def time_split(start_times: np.ndarray, label_end_times: np.ndarray, fractions=(0.7, 0.15)):
    """Masks (train, val, test) by time. Boundaries are quantiles of the sample start
    times; a sample is kept only if its label is fully known before the next split
    begins (label_end < boundary). This handles weekend and missing-bar gaps, where
    a fixed number of bars can span far more wall-clock time."""
    q1, q2 = np.quantile(start_times, [fractions[0], fractions[0] + fractions[1]])
    train = label_end_times < q1
    val = (start_times >= q1) & (label_end_times < q2)
    test = start_times >= q2
    return train, val, test, (float(q1), float(q2))


def last_closed_bar(s: Series, at: float, bar_seconds: int) -> int:
    """Index of the last candle fully closed at time `at` (candle_time = open time)."""
    return int(np.searchsorted(s.t + bar_seconds, at, side="right")) - 1
